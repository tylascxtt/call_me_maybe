"""Build a token-id -> text table from the model's vocab.json.

Qwen uses byte-level BPE like GPT-2. The vocab stores tokens with the raw bytes
remapped to printable Unicode characters (for example, a space appears as
"G with a dot"). To recover the original text, we undo that mapping and decode
the bytes as UTF-8.

Doing this here means we do not need the SDK's ``decode()`` while masking
logits.
"""

from __future__ import annotations

import json
from typing import Any


def _bytes_to_unicode() -> dict[int, str]:
    """Build the byte-to-Unicode table used by GPT-2 and Qwen tokenizers.

    Printable, non-whitespace bytes map to themselves. Every remaining byte
    value is assigned an unused Unicode code point starting at 256, so every
    byte in the range 0-255 has a unique printable representation.

    Inverting this table lets ``Vocabulary`` recover the original bytes stored
    in the vocabulary file.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )

    mapping = {byte: chr(byte) for byte in printable}

    next_codepoint = 0
    for byte in range(256):
        if byte not in mapping:
            mapping[byte] = chr(256 + next_codepoint)
            next_codepoint += 1

    return mapping


class Vocabulary:
    """Map token IDs to decoded text."""

    def __init__(self, vocab_path: str) -> None:
        """Load a vocabulary file and build the token ID lookup.

        The vocabulary stores byte-level BPE tokens using the remapped Unicode
        representation. Each token is converted back into raw bytes and then
        decoded as UTF-8.

        Tokens that are not valid UTF-8 by themselves are skipped because they
        are not needed for ASCII JSON decoding.
        """
        with open(vocab_path, encoding="utf-8") as handle:
            token_to_id: dict[str, int] = json.load(handle)

        unicode_to_byte = {
            char: byte
            for byte, char in _bytes_to_unicode().items()
        }

        self.id_to_text: dict[int, str] = {}

        for token, token_id in token_to_id.items():
            try:
                raw = bytes(
                    unicode_to_byte[char]
                    for char in token
                )
                self.id_to_text[token_id] = raw.decode("utf-8")
            except (KeyError, UnicodeDecodeError):
                continue

    @classmethod
    def from_model(cls, model: Any) -> Vocabulary:
        """Build a vocabulary from a model's vocabulary file."""
        return cls(model.get_path_to_vocab_file())

    def decode(self, token_ids: list[int]) -> str:
        """Return the decoded text for a sequence of token IDs."""
        return "".join(
            self.id_to_text.get(token_id, "")
            for token_id in token_ids
        )
