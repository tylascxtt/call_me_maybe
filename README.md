*This project has been created as part of the 42 curriculum by mabenaya.*

# call me maybe

**Function calling for small LLMs via constrained decoding.**

This README is meant to be read on its own: by the end you should understand
*what* the program does, *why* it works, and *how* every part is built — without
needing to open the source files.

---

## Table of contents

1. [What this project does](#1-what-this-project-does)
2. [Background you need (in plain words)](#2-background-you-need-in-plain-words)
3. [The data: input and output files](#3-the-data-input-and-output-files)
4. [How it is graded (the moulinette)](#4-how-it-is-graded-the-moulinette)
5. [The LLM SDK we are given](#5-the-llm-sdk-we-are-given)
6. [The algorithm, step by step](#6-the-algorithm-step-by-step)
7. [A full worked example](#7-a-full-worked-example)
8. [The custom tokenizer (byte-level BPE)](#8-the-custom-tokenizer-byte-level-bpe)
9. [Module-by-module reference](#9-module-by-module-reference)
10. [Instructions & example usage](#10-instructions--example-usage)
11. [Testing strategy](#11-testing-strategy)
12. [Performance analysis](#12-performance-analysis)
13. [Design decisions](#13-design-decisions)
14. [Challenges faced](#14-challenges-faced)
15. [Rules from the subject we respect](#15-rules-from-the-subject-we-respect)
16. [Resources & AI usage](#16-resources--ai-usage)

---

## 1. What this project does

### Description

A **function-calling** tool. Given a question in plain English, it does **not**
answer the question — it produces the *function call* that would answer it: the
right function name and the right typed arguments.

```
Input prompt : "What is the sum of 2 and 3?"
Output        : {"name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
                 (note: 5 is never computed — we only produce the call)
```

The program reads two JSON files (the list of callable functions and a list of
prompts) and writes one JSON file with one result per prompt. It uses a small
language model, **`Qwen/Qwen3-0.6B`** (600 million parameters), and a technique
called **constrained decoding** to guarantee the output is always valid,
correctly-typed JSON.

### Why it is interesting

A 0.6B model, asked nicely to "please output JSON", succeeds maybe 30–50% of the
time — it forgets a quote, invents a key, writes `2` where a float is needed,
adds chatter around the JSON, etc. Constrained decoding raises the *structural*
reliability to **100%**: the model literally cannot emit a token that would
break the JSON or the schema. The model still decides *which* function and
*what* values; we only stop it from ever going off the rails.

---

## 2. Background you need (in plain words)

You can skip this if you already know how LLM generation works.

**Tokens.** A language model does not read characters or whole words; it reads
**tokens**, which are sub-word chunks. `"fn_add_numbers"` might be the four
tokens `["fn", "_add", "_numbers"]` (illustrative). Each token has an integer
ID. The full list of tokens is the **vocabulary** (Qwen3 has ~151,643 of them).

**Generation is one token at a time.** To generate text the model repeats:

```
text so far ──► model ──► a score (a "logit") for every token in the vocabulary
                              ▼
                  pick one token (usually the highest score)
                              ▼
              append it to the text, and repeat
```

The vector of scores is called the **logits**. Normally you take the token with
the highest logit (greedy decoding) and append it.

**Constrained decoding** changes only the "pick one token" step. Before picking,
you decide which tokens are *allowed* right now (the ones that keep the output
valid) and ignore the rest — equivalently, you set the logits of the forbidden
tokens to −∞. Then you pick the highest-scoring **allowed** token. The model
still expresses its preference through the logits; you just refuse illegal
choices. That is the entire idea, and it is exactly what the subject asks for.

**Why we need our own token→text map.** The model speaks in token IDs, but our
"is this allowed?" rules are about *text* (is the JSON still valid?). So for each
candidate token ID we must know the string it would add. We build that map
ourselves from the model's vocabulary file (see [section 8](#8-the-custom-tokenizer-byte-level-bpe)).

---

## 3. The data: input and output files

### Input 1 — `functions_definition.json`

The functions the system may call. Each has a name, a one-line description,
typed parameters, and a return type. Types are JSON types: `string`, `number`
(float), `integer`, `boolean`.

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": { "a": { "type": "number" }, "b": { "type": "number" } },
    "returns": { "type": "number" }
  },
  {
    "name": "fn_greet",
    "description": "Generate a greeting message for a person by name.",
    "parameters": { "name": { "type": "string" } },
    "returns": { "type": "string" }
  }
]
```

### Input 2 — `function_calling_tests.json`

Just the prompts to process, in order:

```json
[
  { "prompt": "What is the sum of 2 and 3?" },
  { "prompt": "Greet shrek" }
]
```

### Output — `function_calling_results.json`

One object per prompt, **in the same order**, each with **exactly three keys**:
`prompt`, `name`, `parameters`.

```json
[
  { "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": { "a": 2.0, "b": 3.0 } },
  { "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": { "name": "shrek" } }
]
```

> **Note on the keys.** The subject PDF and the grader both require
> `prompt` / `name` / `parameters`. (Some boilerplate text on the evaluation web
> page mentions `fn_name` / `args` — that is generic and does **not** match the
> actual grader; following it would score 0.)

---

## 4. How it is graded (the moulinette)

The provided grader does **not** just compare JSON text. For each of our output
entries it:

1. checks our `prompt` matches the expected prompt (so order matters);
2. looks up the real Python function named by `name`;
3. **calls that function** with our `parameters`;
4. compares the function's return value to the expected return value.

Two consequences shape the whole design:

* **Types must be exact.** The real functions assert their argument types, e.g.
  `fn_add_numbers` does `assert isinstance(a, float)`. So `2` (an int) fails but
  `2.0` (a float) passes. This is why a `number` parameter must be written as a
  float and an `integer` must not contain a dot.
* **Values must be right.** The model has to extract the correct numbers/strings
  from the prompt; constrained decoding guarantees *validity*, not *correctness*.

The grader is run as:

```sh
cd moulinette
uv sync
uv run python -m moulinette prepare_exercises --set private        # makes the test files
uv run python -m moulinette grade_student_answers --set private \
    --student_answer_path ../data/output/function_calling_results.json
```

There are two exercise sets: **public** (given to us, 5 functions / 11 prompts)
and **private** (used at evaluation, 6 different functions / 11 prompts). Our
program never hard-codes either — it reads whatever files it is given.

---

## 5. The LLM SDK we are given

`llm_sdk/` wraps the model. We are only allowed to use its **public** methods.
We use three of them:

| Method | What it returns | We use it to… |
|--------|-----------------|----------------|
| `encode(text) -> tensor` | token IDs for `text` | turn our growing context into model input |
| `get_logits_from_input_ids(ids) -> list[float]` | next-token scores | get the model's preference at each step |
| `get_path_to_vocab_file() -> str` | path to `vocab.json` | build our own token-ID → text map |

We deliberately do **not** use the SDK's `decode()` — we reconstruct token text
ourselves (see [section 8](#8-the-custom-tokenizer-byte-level-bpe)). We never
touch private attributes (`_model`, `_tokenizer`, …). We also never import
`torch`/`transformers` in our own code — only `json`, `re`, `numpy`, `pydantic`,
and the SDK.

---

## 6. The algorithm, step by step

Every answer has the same fixed shape:

```
{"name": <function>, "parameters": {<arg>: <value>, ...}}
```

The braces, quotes, commas, colons and parameter names are **not decisions** —
they are always the same. Only two things are real decisions:

1. **which function** to call, and
2. **what each parameter value is**.

So the strategy is: **write the fixed parts directly in code, and ask the model
only for the two kinds of decision.** This keeps the code linear (you can read
the JSON being built) and fast (the model runs only where a choice exists).

### 6.1 The one primitive: `_best` (masked token pick)

Every decision uses the same helper. Given the text so far and a predicate
`allowed(piece)` that says whether a candidate token's text is a legal
continuation, it asks the model for logits and returns the **highest-logit token
that is allowed**:

```python
def _best(self, context, allowed):
    input_ids = self._model.encode(context).tolist()[0]
    logits = np.asarray(self._model.get_logits_from_input_ids(input_ids))
    for token_id in np.argsort(logits)[::-1]:          # highest score first
        token_text = self._id_to_text.get(int(token_id))
        if token_text and allowed(token_text):         # first legal one wins
            return token_text
    raise DecodeError("no valid token to continue")
```

Scanning from the highest logit and skipping the illegal tokens **is** "set the
illegal logits to −∞ and take the max". This is the constrained-decoding step.

### 6.2 Building one call: `generate`

```python
def generate(self, prompt, functions):
    context = build_prompt(prompt, functions)          # the natural-language preamble
    by_name = {fn.name: fn for fn in functions}

    # (1) the model chooses the function name
    name = self._choose(context + '{"name": "', list(by_name))
    fn = by_name[name]

    # (2) fill the parameters in declaration order
    text = context + f'{{"name": "{name}", "parameters": {{'
    params = {}
    items = fn.ordered_parameters()
    for index, (pname, ptype) in enumerate(items):
        text += (", " if index else "") + f'"{pname}": '
        value, written = self._value(text, ptype, last=index == len(items) - 1)
        params[pname] = value
        text += written

    return FunctionCall(prompt=prompt, name=name, parameters=params)
```

Notice the result is built from the chosen `name` and the decoded `params`, and
the original `prompt` is attached verbatim. We never "generate then re-parse and
hope" — the output is correct **by construction**, and the prompt always matches.

### 6.3 Choosing the function: `_choose`

We keep taking the best token that keeps the text a prefix of one of the known
names, and stop when the text equals a full name (the names are all distinct):

```python
def _choose(self, context, options):
    text = ""
    while text not in options:
        text += self._best(context + text,
                            lambda t: any(o.startswith(text + t) for o in options))
    return text
```

The model can therefore only ever spell a **real** function name — but *which*
one is its choice, guided by the request in the preamble. (The same helper picks
`true` / `false` for boolean values.)

### 6.4 Decoding a value by type: `_value`

```python
def _value(self, context, ptype, last):
    if ptype == "string":
        return self._string(context)                       # quoted, escaped
    if ptype == "boolean":
        word = self._choose(context, ["true", "false"])
        return word == "true", word
    number = self._number(context, ptype == "number", "}" if last else ",")
    return (float(number) if ptype == "number" else int(number)), number
```

* **string** — emit `"`, then characters, until the model emits the closing
  quote. Only valid JSON-string pieces are allowed (escapes handled).
* **number / integer** — emit digits (no leading zeros; a dot only for
  `number`) until the model emits the character that must follow the value
  (`,` between parameters, `}` for the last one). We then cut at that character.
* **boolean** — pick `true` or `false`.

Finally the value is **coerced to the schema type**: a `number` becomes a Python
`float`, an `integer` an `int`. That is what makes `2` come out as `2.0` and
satisfy `isinstance(a, float)` in the grader.

### 6.5 The legality rules

These are three tiny, pure, model-free functions (easy to read and unit-test):

* **`_number_ok(piece, allow_fraction, end)`** — is `piece` still on track to be
  a JSON number? It uses regexes that reject leading zeros (so `06` is illegal)
  and require a digit after a dot. Once the boundary `end` appears, the part
  before it must be a *complete* number.
* **`_string_close(piece)`** — where does the JSON string close? Returns the
  index of the closing quote, `-1` if still a valid prefix, `-2` if impossible.
  It handles the simple escapes (`\"`, `\\`, `\n`, …) so the model can only
  produce a parseable string. `\uXXXX` is deliberately disallowed: the prompts
  are ASCII and verbatim copying never needs it.
* the **prefix check inside `_choose`** — is the name typed so far still the
  start of a real function name?

---

## 7. A full worked example

Prompt: **"What is the sum of 2 and 3?"**, with `fn_add_numbers(a: number, b:
number)` among the functions.

| Step | Code writes / model decides | Text so far (after `Answer: `) |
|------|------------------------------|--------------------------------|
| write | literal | `{"name": "` |
| **decide** | `_choose` → model spells a real name | `{"name": "fn_add_numbers` |
| write | literal | `{"name": "fn_add_numbers", "parameters": {` |
| write | literal (first param) | `… {"a": ` |
| **decide** | `_number` → model emits `2`, then `,` (boundary) → value `"2"` | `… {"a": 2` |
| write | literal (separator) | `… {"a": 2, "b": ` |
| **decide** | `_number` → model emits `3`, then `}` (boundary) → value `"3"` | `… {"a": 2, "b": 3` |
| build | coerce: `"2"→2.0`, `"3"→3.0` | — |

Result object:

```json
{ "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": { "a": 2.0, "b": 3.0 } }
```

The model was consulted only at the **decide** rows (the name and the two
numbers); every other row is a literal written by the code.

---

## 8. The custom tokenizer (byte-level BPE)

To run the rules we need, for any token ID, the exact string it adds. We build
that map once, from the model's `vocab.json`.

Qwen3 (like GPT-2) uses **byte-level BPE**: every byte 0–255 is first mapped to a
*printable* Unicode character so the vocabulary file contains no raw control
bytes. For example a leading space is stored as the character `Ġ`, so the token
`"Ġthe"` really means the text `" the"`. To recover real text we reverse that
"bytes ↔ unicode" mapping and decode the resulting bytes as UTF-8:

```python
unicode_to_byte = {ch: b for b, ch in _bytes_to_unicode().items()}
for token, token_id in vocab.items():            # token e.g. "Ġthe"
    raw = bytes(unicode_to_byte[ch] for ch in token)   # -> b" the"
    id_to_text[token_id] = raw.decode("utf-8")          # -> " the"
```

`_bytes_to_unicode()` is the well-known GPT-2 table (printable bytes map to
themselves; the rest map to code points starting at 256). Tokens whose bytes are
not valid UTF-8 on their own are skipped — they are never needed for ASCII JSON.

This is why we don't call the SDK's `decode()` in the loop: we have a direct
ID → text dictionary, which is also one of the project's bonus goals ("recode
the tokenizer").

---

## 9. Module-by-module reference

```
src/
  __main__.py    CLI: parse args, load + validate inputs, run decoding, write output
  schemas.py     pydantic models (FunctionDefinition, FunctionCall)
  tokenizer.py   Vocabulary: build token-ID -> text from vocab.json (byte-level BPE)
  decoder.py     constrained decoding: generate(), _choose/_string/_number/_best, rules
tests/           offline unit tests (no model needed)
llm_sdk/         the provided LLM wrapper (copied in, unmodified)
data/input/      example input files (public set)
```

Data flows in one direction: **`__main__` → `decoder` → (`tokenizer` +
`schemas`)**. The four modules are small and each has a single job.

* **`schemas.py`** — `FunctionDefinition` mirrors one entry of
  `functions_definition.json`; `FunctionCall` is one output entry
  (`prompt`/`name`/`parameters`). `ordered_parameters()` returns
  `(name, type)` pairs in declaration order (the order matters for output).
  Validating with pydantic means malformed input is rejected early and clearly.
* **`tokenizer.py`** — the `Vocabulary` class from section 8. Exposes
  `id_to_text` (a `dict[int, str]`) and a small `decode()` helper for debugging.
* **`decoder.py`** — everything from section 6: the prompt builder, the three
  value rules, and the `ConstrainedDecoder` with `generate`, `_choose`,
  `_string`, `_number`, `_value`, `_best`. `decode_all()` loops over all prompts
  reusing one decoder; if a single prompt ever fails it logs a warning and emits
  a placeholder so the output stays aligned with the input (it never crashes the
  run).
* **`__main__.py`** — the CLI. Loads and validates the two input files (clear
  error messages for missing files / bad JSON / wrong shape), loads the model,
  decodes every prompt, and writes the output. Every foreseeable error is caught
  and reported instead of crashing.

---

## 10. Instructions & example usage

```sh
make install   # uv sync — installs numpy, pydantic, the local llm_sdk, dev tools
make run       # uv run python -m src   (defaults: data/input -> data/output)
make test      # uv run pytest          (offline unit tests, no model needed)
make lint      # flake8 . + mypy . with the mandatory flags
make debug     # python -m pdb -m src
make clean     # remove caches
```

Custom paths and model:

```sh
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input                data/input/function_calling_tests.json \
    --output               data/output/function_calling_results.json \
    --model                Qwen/Qwen3-0.6B
```

`--model` accepts any causal LLM on the Hugging Face Hub; the default is
`Qwen/Qwen3-0.6B`. The **first run downloads the model weights** (needs network;
later runs use the local cache).

What a run looks like (progress is printed to stderr, one line per prompt):

```
$ uv run python -m src
Loading model Qwen/Qwen3-0.6B ...
[1/11] fn_add_numbers({'a': 2.0, 'b': 3.0})
[2/11] fn_add_numbers({'a': 265.0, 'b': 345.0})
...
Wrote 11 function calls to data/output/function_calling_results.json
```

---

## 11. Testing strategy

`make test` runs offline (no model, no network):

* **`test_decoder.py`**
  * the value rules: `_number_ok` (integer-rejects-dot, no-leading-zeros,
    digit-after-dot) and `_string_close` (escaping, where the quote closes);
  * the decoder driven by a **mock model that returns random logits**. Whatever
    random preferences the mock has, the test asserts the output is *always* a
    valid, schema-compliant call (right function name, exactly the right
    parameters, correct Python types, JSON that round-trips). This proves the
    central guarantee without needing the real LLM.
* **`test_tokenizer.py`** — byte-level decoding of ASCII, spaces (`Ġ` → " "),
  and multi-token text.

End-to-end, the real check is running `uv run python -m src` and then grading
the output with the provided moulinette (section 4).

---

## 12. Performance analysis

Measured with `Qwen/Qwen3-0.6B` on CPU, graded by the provided moulinette:

| Set     | moulinette score  | Function selection | JSON validity | Wall time (11 prompts) |
|---------|-------------------|--------------------|---------------|------------------------|
| public  | **10/11 (90.9%)** | 11/11 (100%)       | 100%          | ~2 min                 |
| private | **9/11 (81.8%)**  | 11/11 (100%)       | 100%          | ~2 min                 |

* **JSON validity: 100%** — a structural *guarantee* of constrained decoding,
  not a number that can regress. Every output parses and matches the schema.
* **Function selection: 100%** — the right function on every test of both sets.
* **Speed:** ~2 minutes, well under the 5-minute budget. The model runs only at
  decision points (name + values); the fixed scaffolding is written for free.
* **Argument extraction: ~82–91%** — numbers (including `1234567.89`), booleans,
  and most strings (even the Windows path `C:\Users\john\config.ini`, which
  needs correct JSON escaping) are extracted exactly. The few misses are the
  hardest verbatim-copy cases for a 0.6B model: a dropped leading `/` in a path,
  an embedded double-quoted phrase, and an extra space in a regex replacement.
  Even then, the output is still valid, correctly-typed JSON.

---

## 13. Design decisions

* **Write the structure, ask only for decisions.** The JSON skeleton is fixed,
  so we write it in code and call the model only for the name and the values.
  Linear, readable, and fast.
* **Build the result from the choices, not by parsing text.** The output is
  assembled from the chosen name and decoded values, so it is correct by
  construction — no "parse it back and hope it's valid" step.
* **Nothing hard-coded.** Function names, parameter names and types all come
  from `functions_definition.json` at runtime, because the files change at
  evaluation time.
* **Tiny, testable legality rules.** `_number_ok`, `_string_close`, and the
  name-prefix check have no model dependency and are unit-tested directly.
* **Types pinned by the schema.** `number → float`, `integer → int` (a dot is
  rejected for integers), so the JSON matches the function signatures the grader
  calls.
* **pydantic at the boundaries.** Inputs and the output are validated by
  `FunctionDefinition` / `FunctionCall`.
* **No forbidden dependencies.** `src` imports only `json`, `re`, `numpy`,
  `pydantic` and the public `llm_sdk` API.

---

## 14. Challenges faced

* **Never force a *decision*.** An early version "forced" any character that was
  structurally the only option — including the shared `fn_` prefix of the
  function names. From the context `…"name": "fn_`, the 0.6B model then
  collapsed to one default function (`fn_execute_sql_query`) for *every* request.
  Letting the model choose the *whole* name from `{"name": "` took selection
  from ~45% to 100%. A long "rules" preamble had the same distracting effect and
  was removed. **Lesson: constrain the structure, never the decision.**
* **JSON number rules.** The first version accepted `06` and `2.`, which
  `json.loads` rejects. Fixed by following the JSON number grammar exactly (no
  leading zeros; a dot must be followed by a digit). A unit test locks this in —
  it is what caught the bug.
* **float vs int.** The grader asserts exact Python types, so a bare `3` (int)
  fails a `number` parameter. Numbers are coerced to `float`; integers forbid a
  dot.
* **Tokens cross field boundaries.** One token often closes a value *and* starts
  the next field (e.g. `", "encoding": "`), and the model strongly prefers such
  tokens. A value decoder must **accept** them and **cut at the value's end**
  (the closing quote for strings, `,`/`}` for numbers). Rejecting them forced the
  model into awkward escapes and produced run-on strings — fixed by cutting at
  the boundary.
* **Byte-level BPE.** The raw vocabulary strings are byte-to-unicode encoded, so
  they are not the real text; reversing that mapping (section 8) was needed to
  know what each token emits.

---

## 15. Rules from the subject we respect

* Runs with `uv run python -m src`; default input `data/input/`, default output
  `data/output/function_calling_results.json`.
* The function is chosen **by the LLM**, never by keyword matching or heuristics.
* **Constrained decoding** is real (logit masking), not "prompt and hope".
* Only the SDK's **public** methods are used; no private attributes.
* No forbidden packages (`torch`, `transformers`, `outlines`, … are not imported
  by our code).
* All classes use **pydantic**; all errors are handled gracefully (no crashes).
* `flake8` and `mypy` (with the mandatory flags) pass.

---

## 16. Resources & AI usage

* Qwen3-0.6B model card: <https://huggingface.co/Qwen/Qwen3-0.6B>
* pydantic v2 docs: <https://docs.pydantic.dev/latest/>
* GPT-2 byte-level BPE (the "bytes ↔ unicode" table):
  <https://github.com/openai/gpt-2/blob/master/src/encoder.py>
* Background on constrained decoding for structured generation (Outlines):
  <https://github.com/dottxt-ai/outlines>

### AI usage

AI assistance (Claude) was used to scaffold boilerplate, draft this README, and
review the JSON edge cases (number formatting, string escaping, token
boundaries). Every design decision — writing the structure vs. deciding with the
model, the byte-level decoder, the value rules — was reviewed, understood, and
tested by hand; the unit tests encode that understanding so the behaviour is
verifiable rather than taken on faith.
