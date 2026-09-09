
"""Constrained decoding: build the function call one piece at a time.

The answer always has the shape:
    {"name": <function>, "parameters": {<arg>: <value>, ...}}
The braces, quotes and keys never change, so we write them out directly and
only ask the model for the parts that vary: the function name and each value.
At those steps we read the model's logits and take the best token that still
keeps the JSON valid (the small _ok rules below decide what stays valid).
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

import numpy as np

from .schemas import FunctionCall, FunctionDefinition
from .tokenizer import Vocabulary

# don't let a single value loop forever
_MAX_VALUE_LEN = 256

# number patterns: FULL = a finished number, PREFIX = something that can still
# grow into one. Leading zeros are rejected so json.loads won't choke later.
_INT_FULL = re.compile(r"-?(0|[1-9][0-9]*)")
_INT_PREFIX = re.compile(r"-?(0|[1-9][0-9]*)?")
_NUM_FULL = re.compile(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?")
_NUM_PREFIX = re.compile(r"-?((0|[1-9][0-9]*)(\.[0-9]*)?)?")

_STRING_ESCAPES = set('"\\/bfnrt')


class DecodeError(RuntimeError):
    """Raised when no legal continuation exists, or a value runs away.

    Covers two situations: `_best` finds every candidate token disallowed
    by the current `allowed` predicate, or one of the piece-builders
    (`_choose`, `_string`, `_number`) exceeds `_MAX_VALUE_LEN` without
    closing, which would otherwise generate forever.
    """

    pass


def _number_ok(piece: str, allow_fraction: bool, end: str) -> bool:
    """Check whether `piece` is still a valid (possibly unfinished) number.

    `end` is the character that terminates the number in the surrounding
    JSON (',' for a non-last parameter, '}' for the last one). Once `end`
    appears in `piece`, the number is considered finished and the text
    before it must fully match a complete number; until then, `piece` only
    needs to match a prefix that could still grow into one. `allow_fraction`
    switches between integer-only matching and float-capable matching.
    """
    full = _NUM_FULL if allow_fraction else _INT_FULL
    prefix = _NUM_PREFIX if allow_fraction else _INT_PREFIX
    if end in piece:
        return bool(full.fullmatch(piece[: piece.index(end)]))
    return bool(prefix.fullmatch(piece))


def _string_close(piece: str) -> int:
    """Scan `piece` (which starts with an opening quote) for its closing quote.

    Returns the index of the closing `"` if one has been reached, -1 if the
    string is still open but everything so far is still valid, or -2 if
    `piece` can no longer become a valid JSON string. Anything after the
    closing quote is ignored, so a token that closes the string and then
    runs on into the next field is still treated as fine. Only the simple
    escapes in `_STRING_ESCAPES` are accepted (see the note below on why
    `\\uXXXX` is deliberately excluded), and raw control characters are
    rejected outright since they can't appear literally inside a JSON string.
    """
    i = 1
    while i < len(piece):
        c = piece[i]
        if c == '"':
            return i
        if c == "\\":
            # only the simple escapes are allowed. \uXXXX is intentionally
            # not: the prompts are ascii and verbatim copying never needs
            # it (a literal backslash becomes \\, a quote \"), so
            # disallowing it keeps the model from ever picking an escape
            # form it would have to spell out.
            if i + 1 >= len(piece):
                return -1
            if piece[i + 1] not in _STRING_ESCAPES:
                return -2
            i += 2
            continue
        if ord(c) < 0x20:  # raw control chars aren't valid in a JSON string
            return -2
        i += 1
    return -1


def _format_functions(functions: list[FunctionDefinition]) -> str:
    """Render `functions` as a bullet list of `name(arg: type, ...): descr`.

    Used to describe the available functions to the model inside the prompt.
    """
    lines = []
    for fn in functions:
        args = ", ".join(
            f"{name}: {jtype}" for name, jtype in fn.ordered_parameters()
        )
        lines.append(f"- {fn.name}({args}): {fn.description}")
    return "\n".join(lines)


def build_prompt(prompt: str, functions: list[FunctionDefinition]) -> str:
    """Build the full prompt the model sees for a request and function set.

    Lists the available functions and asks for a single JSON function call in
    response. Kept short on purpose: the constrained-decoding rules already
    enforce structure and types, so the prompt only has to help the model
    pick the function and the values. A longer "rules" preamble actually
    made the small model worse.
    """
    return (
        "You are a function-calling engine. "
        "Convert the user request into exactly one function call.\n"
        "Reply with only a JSON object of the form "
        '{"name": <function>, "parameters": {<arguments>}}.\n'
        "Copy any file paths, templates, or quoted text from the request "
        "character-for-character, including"
        " punctuation like \" { } / and quotes.\n\n"
        f"Available functions:\n{_format_functions(functions)}\n\n"
        f"Request: {prompt}\n"
        "Answer: "
    )


class ConstrainedDecoder:
    """Generates a single well-formed `FunctionCall` via constrained
    decoding.

    The surrounding JSON scaffolding (braces, quotes, keys) is emitted
    directly; only the function name and each parameter value are produced
    by the model, one token at a time, with every candidate token filtered
    against the grammar rules for that slot (name choice, string, number, or
    boolean).
    """

    def __init__(self, model: Any) -> None:
        """Wrap `model` and build the vocabulary's token-id-to-text lookup."""
        self._model = model
        self._id_to_text = Vocabulary.from_model(model).id_to_text

    def generate(
        self, prompt: str, functions: list[FunctionDefinition]
    ) -> FunctionCall:
        """Decode one `FunctionCall` for `prompt` out of the given `functions`.

        First picks a function name from the closed set of candidates, then
        fills in its parameters in order, appending each generated piece to
        the running context so later values are conditioned on earlier ones.
        """
        context = build_prompt(prompt, functions)
        by_name = {fn.name: fn for fn in functions}

        # the model picks the name; it can only spell a real one
        name = self._choose(context + '{"name": "', list(by_name))
        fn = by_name[name]

        # fill the parameters in order
        text = context + f'{{"name": "{name}", "parameters": {{'
        params: dict[str, Any] = {}
        items = fn.ordered_parameters()
        for index, (pname, ptype) in enumerate(items):
            text += (", " if index else "") + f'"{pname}": '
            is_last = index == len(items) - 1
            value, written = self._value(text, ptype, last=is_last)
            params[pname] = value
            text += written

        return FunctionCall(prompt=prompt, name=name, parameters=params)

    def _value(self, context: str, ptype: str, last: bool) -> tuple[Any, str]:
        """Decode a single parameter value of type `ptype`.

        Returns a `(parsed_value, raw_text)` pair. Dispatches to the
        string, boolean, or number decoder based on `ptype`. `last` tells
        the number decoder whether to terminate on '}' (final parameter)
        or ',' (more parameters follow).
        """
        if ptype == "string":
            return self._string(context)
        if ptype == "boolean":
            word = self._choose(context, ["true", "false"])
            return word == "true", word
        number = self._number(context, ptype == "number", "}" if last else ",")
        return (float(number) if ptype == "number" else int(number)), number

    def _choose(self, context: str, options: list[str]) -> str:
        """Grow text token by token until it matches one of `options`.

        At each step only tokens that keep the text a prefix of at least one
        option are allowed, so the result is guaranteed to land on a valid
        choice. Since the options are all distinct, this stops cleanly once
        `text` equals one of them.
        """
        text = ""
        while text not in options:
            def allowed(token: str) -> bool:
                return any(
                    option.startswith(text + token) for option in options)

            text += self._best(
                context + text,
                allowed,
            )
            if len(text) > _MAX_VALUE_LEN:
                raise DecodeError(f"runaway choice: {text!r}")
        return text

    def _string(self, context: str) -> tuple[str, str]:
        """Decode a JSON string value, returning (decoded_str, raw_json_text).

        Starts from an opening quote and keeps appending tokens allowed by
        `_string_close` until a closing quote is reached. Anything generated
        past the closing quote is discarded before parsing.
        """
        piece = '"'
        while _string_close(piece) < 0:
            def allowed(token: str) -> bool:
                return _string_close(piece + token) != -2

            piece += self._best(
                context + piece,
                allowed,
            )
            if len(piece) > _MAX_VALUE_LEN:
                raise DecodeError(f"runaway string: {piece!r}")
        text = piece[: _string_close(piece) + 1]  # drop text past the quote
        return json.loads(text), text

    def _number(self, context: str, allow_fraction: bool, end: str) -> str:
        """Decode a JSON number, returning its raw text (no trailing `end`).

        Keeps appending tokens allowed by `_number_ok` until the terminator
        `end` (',' or '}') appears, then returns everything generated before
        it.
        """
        piece = ""
        while end not in piece:
            def allowed(token: str) -> bool:
                return _number_ok(
                    piece + token,
                    allow_fraction,
                    end,
                )

            piece += self._best(
                context + piece,
                allowed,
            )
            if len(piece) > _MAX_VALUE_LEN:
                raise DecodeError(f"runaway number: {piece!r}")
        return piece[: piece.index(end)]

    def _best(self, context: str, allowed: Callable[[str], bool]) -> str:
        """Return the highest-logit next token whose text satisfies `allowed`.

        Encodes `context`, runs the model to get next-token logits, and
        walks the vocabulary from highest to lowest logit, returning the
        first token whose text passes `allowed`. This is equivalent to
        setting the logits of every disallowed token to -inf before
        sampling. Raises `DecodeError` if no candidate token is allowed.
        """
        input_ids = self._model.encode(context).tolist()[0]
        logits = np.asarray(self._model.get_logits_from_input_ids(input_ids))
        for token_id in np.argsort(logits)[::-1]:
            token_text = self._id_to_text.get(int(token_id))
            if token_text and allowed(token_text):
                return token_text
        raise DecodeError("no valid token to continue")


def decode_all(
    prompts: list[str],
    functions: list[FunctionDefinition],
    model: Any,
    on_result: Callable[[int, FunctionCall], None] | None = None,
) -> list[FunctionCall]:
    """Decode a `FunctionCall` for each prompt in `prompts`, in order.

    Builds one `ConstrainedDecoder` and reuses it across all prompts. If a
    prompt fails to decode, a warning is printed and a placeholder call
    (the first function, no parameters) is substituted so the returned list
    stays aligned with `prompts` index-for-index rather than raising and
    losing the whole run. If `on_result` is given, it's called with
    `(index, call)` immediately after each prompt is decoded.
    """
    decoder = ConstrainedDecoder(model)
    results: list[FunctionCall] = []
    for index, prompt in enumerate(prompts):
        try:
            call = decoder.generate(prompt, functions)
        except DecodeError as exc:
            # one bad prompt shouldn't kill the run; keep the list aligned
            print(f"warning: prompt {index + 1} could not be decoded: {exc}")
            call = FunctionCall(
                prompt=prompt, name=functions[0].name, parameters={}
            )
        results.append(call)
        if on_result is not None:
            on_result(index, call)
    return results
