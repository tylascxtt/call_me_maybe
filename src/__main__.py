"""Entry point for `python -m src`.

Loads the function definitions and the prompts, decodes a function call for
each prompt, and writes the results. Bad input or a model failure prints an
error and exits non-zero rather than crashing.

    python -m src [--functions_definition F] [--input F] [--output F] [--model ID]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from .decoder import DecodeError, decode_all
from .schemas import FunctionCall, FunctionDefinition

_FUNCTION_DEFINITIONS = TypeAdapter(list[FunctionDefinition])

_DEFAULT_FUNCTIONS = Path("data/input/functions_definition.json")
_DEFAULT_INPUT = Path("data/input/function_calling_tests.json")
_DEFAULT_OUTPUT = Path("data/output/function_calling_results.json")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="src")
    parser.add_argument("--functions_definition", type=Path, default=_DEFAULT_FUNCTIONS)
    parser.add_argument("--input", type=Path, default=_DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    return parser.parse_args(argv)


def _load_functions(path: Path) -> list[FunctionDefinition]:
    with path.open(encoding="utf-8") as handle:
        return _FUNCTION_DEFINITIONS.validate_python(json.load(handle))


def _load_prompts(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("input file must contain a JSON array of objects")
    prompts = []
    for item in data:
        if not isinstance(item, dict) or "prompt" not in item:
            raise ValueError("every input entry must be an object with a 'prompt' key")
        prompts.append(str(item["prompt"]))
    return prompts


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # load and validate the input files
    try:
        functions = _load_functions(args.functions_definition)
        prompts = _load_prompts(args.input)
    except FileNotFoundError as exc:
        print(f"error: input file not found: {exc.filename}", file=sys.stderr)
        return 1
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        print(f"error: could not parse input files: {exc}", file=sys.stderr)
        return 1

    if not functions:
        print("error: no function definitions provided", file=sys.stderr)
        return 1

    # load the model
    try:
        from llm_sdk import Small_LLM_Model

        print(f"Loading model {args.model} ...", file=sys.stderr)
        model = Small_LLM_Model(model_name=args.model)
    except Exception as exc:  # a model load can fail in many ways
        print(f"error: failed to load model '{args.model}': {exc}", file=sys.stderr)
        return 1

    # decode every prompt
    def _report(index: int, call: FunctionCall) -> None:
        print(f"[{index + 1}/{len(prompts)}] {call.name}({call.parameters})", file=sys.stderr)

    try:
        results = decode_all(prompts, functions, model, on_result=_report)
    except DecodeError as exc:
        print(f"error: constrained decoding failed: {exc}", file=sys.stderr)
        return 1

    # write the output
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(
                [call.model_dump() for call in results],
                handle,
                indent=2,
                ensure_ascii=False,
            )
            handle.write("\n")
    except OSError as exc:
        print(f"error: could not write output file: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {len(results)} function calls to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
