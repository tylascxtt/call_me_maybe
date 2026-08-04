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
    pass


def _number_ok(piece: str, allow_fraction: bool, end: str) -> bool:
    # `end` is the char that comes after the number (',' or '}'). Once it shows
    # up the number is done, so the part before it has to be complete.
    full = _NUM_FULL if allow_fraction else _INT_FULL
    prefix = _NUM_PREFIX if allow_fraction else _INT_PREFIX
    if end in piece:
        return bool(full.fullmatch(piece[: piece.index(end)]))
    return bool(prefix.fullmatch(piece))


def _string_close(piece: str) -> int:
    # index of the closing quote, -1 if still open (but valid so far), -2 if it
    # can't become a valid string. Anything after the quote is ignored, so a
    # token that closes the string and runs into the next field is fine.
    i = 1
    while i < len(piece):
        c = piece[i]
        if c == '"':
            return i
        if c == "\\":
            # only the simple escapes are allowed. \uXXXX is intentionally not:
            # the prompts are ascii and verbatim copying never needs it (a literal
            # backslash becomes \\, a quote \"), so disallowing it keeps the model
            # from ever picking an escape form it would have to spell out.
            if i + 1 >= len(piece):
                return -1
            if piece[i + 1] not in _STRING_ESCAPES:
                return -2
            i += 2
            continue
        if ord(c) < 0x20:  # raw control chars aren't valid inside a JSON string
            return -2
        i += 1
    return -1


def _format_functions(functions: list[FunctionDefinition]) -> str:
    lines = []
    for fn in functions:
        args = ", ".join(f"{name}: {jtype}" for name, jtype in fn.ordered_parameters())
        lines.append(f"- {fn.name}({args}): {fn.description}")
    return "\n".join(lines)


def build_prompt(prompt: str, functions: list[FunctionDefinition]) -> str:
    # kept short on purpose: the rules already enforce structure and types, so
    # the prompt only has to help the model pick the function and the values.
    # a longer "rules" preamble actually made the small model worse.
    return (
        "You are a function-calling engine. "
        "Convert the user request into exactly one function call.\n"
        "Reply with only a JSON object of the form "
        '{"name": <function>, "parameters": {<arguments>}}.\n\n'
        f"Available functions:\n{_format_functions(functions)}\n\n"
        f"Request: {prompt}\n"
        "Answer: "
    )


class ConstrainedDecoder:
    def __init__(self, model: Any) -> None:
        self._model = model
        self._id_to_text = Vocabulary.from_model(model).id_to_text

    def generate(
        self, prompt: str, functions: list[FunctionDefinition]
    ) -> FunctionCall:
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
            value, written = self._value(text, ptype, last=index == len(items) - 1)
            params[pname] = value
            text += written

        return FunctionCall(prompt=prompt, name=name, parameters=params)

    def _value(self, context: str, ptype: str, last: bool) -> tuple[Any, str]:
        if ptype == "string":
            return self._string(context)
        if ptype == "boolean":
            word = self._choose(context, ["true", "false"])
            return word == "true", word
        number = self._number(context, ptype == "number", "}" if last else ",")
        return (float(number) if ptype == "number" else int(number)), number

    def _choose(self, context: str, options: list[str]) -> str:
        # grow the text token by token, staying a prefix of some option, until
        # it equals one. the options are all distinct so this stops cleanly.
        text = ""
        while text not in options:
            text += self._best(
                context + text,
                lambda t: any(o.startswith(text + t) for o in options),
            )
            if len(text) > _MAX_VALUE_LEN:
                raise DecodeError(f"runaway choice: {text!r}")
        return text

    def _string(self, context: str) -> tuple[str, str]:
        piece = '"'
        while _string_close(piece) < 0:
            piece += self._best(
                context + piece, lambda t: _string_close(piece + t) != -2
            )
            if len(piece) > _MAX_VALUE_LEN:
                raise DecodeError(f"runaway string: {piece!r}")
        text = piece[: _string_close(piece) + 1]  # drop anything past the quote
        return json.loads(text), text

    def _number(self, context: str, allow_fraction: bool, end: str) -> str:
        piece = ""
        while end not in piece:
            piece += self._best(
                context + piece, lambda t: _number_ok(piece + t, allow_fraction, end)
            )
            if len(piece) > _MAX_VALUE_LEN:
                raise DecodeError(f"runaway number: {piece!r}")
        return piece[: piece.index(end)]

    def _best(self, context: str, allowed: Callable[[str], bool]) -> str:
        # take the highest-logit token whose text is still allowed. skipping the
        # rest is the same as setting their logits to -inf.
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
    decoder = ConstrainedDecoder(model)
    results: list[FunctionCall] = []
    for index, prompt in enumerate(prompts):
        try:
            call = decoder.generate(prompt, functions)
        except DecodeError as exc:
            # one bad prompt shouldn't kill the run; keep the list aligned
            print(f"warning: prompt {index + 1} could not be decoded: {exc}")
            call = FunctionCall(prompt=prompt, name=functions[0].name, parameters={})
        results.append(call)
        if on_result is not None:
            on_result(index, call)
    return results
