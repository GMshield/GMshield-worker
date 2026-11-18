"""The keyword vector of a passage (Story 2.1, AD-10): a sparse vector whose token numbers only the owner's key gives.

A text is folded (NFKC, casefold, diacritics stripped) and cut into Unicode word tokens. Each token's number is the
first 4 bytes, big-endian, of HMAC-SHA256 under the owner's token key (`Vault.ownerTokenKey`, sent with the job), and
its value is how often it occurs; two tokens that land on one number add up. Qdrant weighs the values with its `idf`
modifier. The tokens themselves never leave this module: the vector holds numbers only, and nothing here logs.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

_WORD = re.compile(r"\w+")
KEY_BYTES = 32


@dataclass(frozen=True)
class Sparse:
    """A sparse vector: ascending token numbers and their values."""

    indices: list[int]
    values: list[float]


def fold(text: str) -> str:
    """NFKC, casefold, then every combining mark dropped (recomposed after, so Hangul and the like stay whole)."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    stripped = "".join(char for char in unicodedata.normalize("NFD", folded) if not unicodedata.combining(char))
    return unicodedata.normalize("NFC", stripped)


def tokens(text: str) -> list[str]:
    """The folded text's Unicode word tokens, in order."""
    return _WORD.findall(fold(text))


def token_number(key: bytes, token: str) -> int:
    """The first 4 bytes, big-endian, of HMAC-SHA256 under the owner's key."""
    return int.from_bytes(hmac.new(key, token.encode("utf-8"), hashlib.sha256).digest()[:4], "big")


def keyword_vector(text: str, key: bytes) -> Sparse:
    """The text's keyword vector under the owner's key: each token number with its count, ascending."""
    if len(key) != KEY_BYTES:
        raise ValueError("the token key is 32 bytes")
    counts: Counter[int] = Counter()
    for token in tokens(text):
        counts[token_number(key, token)] += 1
    numbers = sorted(counts)
    return Sparse(indices=numbers, values=[float(counts[number]) for number in numbers])
