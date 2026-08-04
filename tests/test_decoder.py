from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.decoder import ConstrainedDecoder, _number_ok, _string_close
from src.schemas import FunctionDefinition


def test_number_rules() -> None:
    # integers reject a dot; numbers accept one
    assert _number_ok("4", allow_fraction=False, end=",")
    assert _number_ok("4,", allow_fraction=False, end=",")        # done, then comma
    assert not _number_ok("4.", allow_fraction=False, end=",")    # no dot for int
    assert _number_ok("2.5}", allow_fraction=True, end="}")       # done float, then }
    assert not _number_ok("06", allow_fraction=False, end=",")    # leading zero
    assert _number_ok("0", allow_fraction=True, end=",")
    assert not _number_ok("2.,", allow_fraction=True, end=",")    # dot needs a digit


def test_string_rules() -> None:
    assert _string_close('"') == -1                # open
    assert _string_close('"hello') == -1           # still open
    assert _string_close('"hello"') == 6           # closes at index 6
    assert _string_close('"a\\nb"') == 5           # valid escape, closes
    assert _string_close('"a\\xb"') == -2          # bad escape
    assert _string_close('"a\\u0041"') == -2       # \u intentionally disallowed
    assert _string_close('"a", "b": ') == 2        # closes, rest is cut off


# a single-char-per-token ascii vocab is enough for the mock to build any value
_VOCAB = {chr(c): c - 0x21 for c in range(0x21, 0x7F)}


class _MockTensor:
    def __init__(self, ids: list[int]) -> None:
        self._ids = ids

    def tolist(self) -> list[list[int]]:
        return [self._ids]


class MockModel:
    """Stands in for Small_LLM_Model with random (seeded) logits."""

    def __init__(self, vocab_path: str, seed: int) -> None:
        self._vocab_path = vocab_path
        self._rng = np.random.default_rng(seed)
        self._size = max(_VOCAB.values()) + 1

    def get_path_to_vocab_file(self) -> str:
        return self._vocab_path

    def encode(self, text: str) -> _MockTensor:
        return _MockTensor([ord(c) % 256 for c in text] or [0])

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        logits: list[float] = self._rng.standard_normal(self._size).tolist()
        return logits


def _fn(name: str, params: dict[str, str], returns: str = "string") -> FunctionDefinition:
    return FunctionDefinition.model_validate(
        {
            "name": name,
            "description": f"test function {name}",
            "parameters": {k: {"type": v} for k, v in params.items()},
            "returns": {"type": returns},
        }
    )


FUNCTIONS = [
    _fn("fn_add_numbers", {"a": "number", "b": "number"}, "number"),
    _fn("fn_greet", {"name": "string"}),
    _fn("fn_is_even", {"n": "integer"}, "boolean"),
    _fn("fn_flag", {"on": "boolean"}, "boolean"),
]
_PY_TYPE = {"number": float, "integer": int, "boolean": bool, "string": str}


def _decoder(tmp_path: Path, seed: int) -> ConstrainedDecoder:
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps(_VOCAB), encoding="utf-8")
    return ConstrainedDecoder(MockModel(str(path), seed))


def test_always_valid_and_schema_compliant(tmp_path: Path) -> None:
    # whatever the logits, the output should be a valid, schema-correct call
    names = {fn.name for fn in FUNCTIONS}
    specs = {fn.name: dict(fn.ordered_parameters()) for fn in FUNCTIONS}

    for seed in range(8):
        call = _decoder(tmp_path, seed).generate("any request", FUNCTIONS)

        assert call.name in names
        assert set(call.parameters) == set(specs[call.name])
        for pname, jtype in specs[call.name].items():
            assert isinstance(call.parameters[pname], _PY_TYPE[jtype])

        # the written output is valid JSON and round-trips
        assert json.loads(json.dumps(call.model_dump()))["name"] == call.name
