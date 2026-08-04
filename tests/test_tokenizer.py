from __future__ import annotations

import json
from pathlib import Path

from src.tokenizer import Vocabulary


def _write_vocab(tmp_path: Path, mapping: dict[str, int]) -> str:
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps(mapping), encoding="utf-8")
    return str(path)


def test_ascii_and_space_decoding(tmp_path: Path) -> None:
    # the byte-level encoding writes a space as 'G with a dot'; ascii is itself
    vocab = Vocabulary(_write_vocab(tmp_path, {"a": 0, "Ġ": 1, "Ġthe": 2, "fn_": 3}))
    assert vocab.id_to_text[0] == "a"
    assert vocab.id_to_text[1] == " "
    assert vocab.id_to_text[2] == " the"
    assert vocab.id_to_text[3] == "fn_"


def test_decode_round_trip(tmp_path: Path) -> None:
    vocab = Vocabulary(_write_vocab(tmp_path, {"Ġhello": 0, "Ġworld": 1, "!": 2}))
    assert vocab.decode([0, 1, 2]) == " hello world!"
    assert vocab.decode([99]) == ""  # unknown ids add nothing
