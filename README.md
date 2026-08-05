# call_me_maybe

Function calling for small LLMs via constrained decoding

A compact implementation of constrained decoding that makes small language models reliably output valid, typed JSON function calls. This project was created as part of the 42 curriculum.

Why this exists
- Small models often emit malformed or ill-typed JSON when asked to produce structured output. This project uses logit-masking (constrained decoding) to force structural correctness while letting the model decide the function and values.

Highlights
- Guarantees structurally valid JSON output (by construction).
- Produces function call objects: { "prompt", "name", "parameters" }.
- Works with any causal LLM supported by the included llm_sdk; default model: Qwen/Qwen3-0.6B.
- Includes unit tests that exercise the legality rules and a mock model to prove the decoder's guarantees.

Quick start

Requirements
- Python 3.10+
- make

Install

```bash
make install
```

Run (default input/output)

```bash
make run
# or directly
uv run python -m src
```

Run with explicit paths and model

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input                data/input/function_calling_tests.json \
  --output               data/output/function_calling_results.json \
  --model                Qwen/Qwen3-0.6B
```

What the program does
- Reads functions_definition.json (callable functions with typed parameters).
- Reads function_calling_tests.json (list of prompts).
- For each prompt, uses constrained decoding to select a function name and typed parameter values.
- Writes data/output/function_calling_results.json with one object per prompt in the same order.

Output format
Each result object has exactly three keys: `prompt`, `name`, `parameters`.
Example

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": { "a": 2.0, "b": 3.0 }
}
```

Project layout

- src/
  - __main__.py         CLI: load inputs, run decoding, write output
  - decoder.py          Constrained decoding logic and legality rules
  - tokenizer.py        Build id->text map from vocab (byte-level BPE)
  - schemas.py          pydantic models for inputs and outputs
- llm_sdk/              Provided LLM wrapper (use only public API)
- tests/                Unit tests (mock model; no network required)
- data/input/           Example input files (public exercise set)
- data/output/          Output file written by runs

Testing

```bash
make test
```

Design notes (short)
- Constrained decoding masks logits to forbid tokens that would break the JSON or the schema.
- The decoder writes fixed JSON scaffolding and only asks the model to make choices for function names and parameter values.
- Numbers are coerced to float for `number` and int for `integer` to satisfy grader assertions.

Grading
- The provided moulinette validates outputs by calling the real Python functions with the produced parameters. Types and values must match exactly.

Contributing
- Follow flake8 and mypy checks. Run `make lint` before opening PRs.

License
- MIT

Contact
- Maintainer: mabenaya (project created as part of the 42 curriculum)
