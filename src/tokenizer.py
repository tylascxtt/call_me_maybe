"""Build a token-id -> text table from the model's vocab.json.

Qwen uses byte-level BPE like GPT-2: the vocab stores tokens with the raw bytes
remapped to printable unicode characters (a space shows up as 'G with a dot').
To get the real text back we undo that mapping and decode the bytes as utf-8.
Doing it here means we don't need the SDK's decode() while masking logits.
"""

from __future__ import annotations

import json
from typing import Any


def _bytes_to_unicode() -> dict[int, str]:
    # the byte<->unicode table used by GPT-2 / Qwen tokenizers
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    mapping = {b: chr(b) for b in printable}
    n = 0
    for byte in range(256):
        if byte not in mapping:
            mapping[byte] = chr(256 + n)
            n += 1
    return mapping


class Vocabulary:
    def __init__(self, vocab_path: str) -> None:
        with open(vocab_path, encoding="utf-8") as handle:
            token_to_id: dict[str, int] = json.load(handle)

        unicode_to_byte = {ch: b for b, ch in _bytes_to_unicode().items()}

        # token id -> the text it adds. Tokens that aren't valid utf-8 by
        # themselves are skipped; we never need them for ascii json.
        self.id_to_text: dict[int, str] = {}
        for token, token_id in token_to_id.items():
            try:
                raw = bytes(unicode_to_byte[ch] for ch in token)
                self.id_to_text[token_id] = raw.decode("utf-8")
            except (KeyError, UnicodeDecodeError):
                continue

    @classmethod
    def from_model(cls, model: Any) -> "Vocabulary":
        return cls(model.get_path_to_vocab_file())

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.id_to_text.get(tid, "") for tid in token_ids)
