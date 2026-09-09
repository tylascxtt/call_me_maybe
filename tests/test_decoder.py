from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.decoder import (
    ConstrainedDecoder,
    _number_ok,
    _string_close,
)
from src.schemas import FunctionDefinition


def test_number_rules() -> None:
    """Test number parsing rules."""
    # Integers reject a dot; numbers accept one.
    assert _number_ok("4", allow_fraction=False, end=",")
    assert _number_ok(
        "4,",
        allow_fraction=False,
        end=",",
    )
    assert not _number_ok(
        "4.",
        allow_fraction=False,
        end=",",
    )
    assert _number_ok(
        "2.5}",
        allow_fraction=True,
        end="}",
    )
    assert not _number_ok(
        "06",
        allow_fraction=False,
        end=",",
    )
    assert _number_ok(
        "0",
        allow_fraction=True,
        end=",",
    )
    assert not _number_ok(
        "2.,",
        allow_fraction=True,
        end=",",
    )


def test_string_rules() -> None:
    """Test string parsing rules."""
    assert _string_close('"') == -1
    assert _string_close('"hello') == -1
    assert _string_close('"hello"') == 6
    assert _string_close('"a\\nb"') == 5
    assert _string_close('"a\\xb"') == -2
    assert _string_close('"a\\u0041"') == -2
    assert _string_close('"a", "b": ') == 2


# A single-character-per-token ASCII vocabulary is enough for the mock
# decoder to construct any JSON value.
_VOCAB = {
    chr(code): code - 0x21
    for code in range(0x21, 0x7F)
}


class _MockTensor:
    def __init__(self, ids: list[int]) -> None:
        self._ids = ids

    def tolist(self) -> list[list[int]]:
        return [self._ids]


class MockModel:
    """Stand in for ``Small_LLM_Model`` with deterministic random logits."""

    def __init__(self, vocab_path: str, seed: int) -> None:
        self._vocab_path = vocab_path
        self._rng = np.random.default_rng(seed)
        self._size = max(_VOCAB.values()) + 1

    def get_path_to_vocab_file(self) -> str:
        return self._vocab_path

    def encode(self, text: str) -> _MockTensor:
        return _MockTensor(
            [ord(char) % 256 for char in text] or [0]
        )

    def get_logits_from_input_ids(
        self,
        input_ids: list[int],
    ) -> list[float]:
        del input_ids
        values = self._rng.standard_normal(self._size)
        return [float(x) for x in values]


def _fn(
    name: str,
    params: dict[str, str],
    returns: str = "string",
) -> FunctionDefinition:
    return FunctionDefinition.model_validate(
        {
            "name": name,
            "description": f"test function {name}",
            "parameters": {
                key: {"type": value}
                for key, value in params.items()
            },
            "returns": {"type": returns},
        }
    )


FUNCTIONS = [
    _fn(
        "fn_add_numbers",
        {"a": "number", "b": "number"},
        "number",
    ),
    _fn(
        "fn_greet",
        {"name": "string"},
    ),
    _fn(
        "fn_is_even",
        {"n": "integer"},
        "boolean",
    ),
    _fn(
        "fn_flag",
        {"on": "boolean"},
        "boolean",
    ),
]

_PY_TYPE = {
    "number": float,
    "integer": int,
    "boolean": bool,
    "string": str,
}


def _decoder(
    tmp_path: Path,
    seed: int,
) -> ConstrainedDecoder:
    path = tmp_path / "vocab.json"
    path.write_text(
        json.dumps(_VOCAB),
        encoding="utf-8",
    )
    return ConstrainedDecoder(
        MockModel(str(path), seed)
    )


def test_always_valid_and_schema_compliant(
    tmp_path: Path,
) -> None:
    """Generated calls should always satisfy the schema."""
    names = {
        function.name
        for function in FUNCTIONS
    }
    specs = {
        function.name: dict(
            function.ordered_parameters()
        )
        for function in FUNCTIONS
    }

    for seed in range(8):
        call = _decoder(
            tmp_path,
            seed,
        ).generate(
            "any request",
            FUNCTIONS,
        )

        assert call.name in names
        assert set(call.parameters) == set(specs[call.name])

        for parameter, json_type in specs[call.name].items():
            assert isinstance(
                call.parameters[parameter],
                _PY_TYPE[json_type],
            )

        assert (
            json.loads(
                json.dumps(call.model_dump())
            )["name"]
            == call.name
        )
