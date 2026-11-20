"""The keyword vector of a passage (Story 2.1, AD-10): a sparse vector whose token numbers only the owner's key gives.

A text is folded (NFKC, casefold, diacritics stripped) and cut into Unicode word tokens. Each token's number is the
first 4 bytes, big-endian, of HMAC-SHA256 under the owner's token key (`Vault.ownerTokenKey`, sent with the job); two
tokens that land on one number count as one.

A passage's value for a number (`keyword_vector`, index recipe "2") is BM25's term weight without its IDF:
`tf·(k1+1) / (tf + k1·(1 - b + b·L/AVG))`, with `tf` the number's count, `L` the passage's token count, `K1` = 1.2,
`B` = 0.75 and `AVG` = `KEYWORD_AVG_TOKENS`. A word that recurs saturates, and a long passage weighs each of its words
less, so a long page full of "de" and "foi" no longer outranks a short passage on common words. A question's value
(`query_vector`) is 1 per distinct number. Qdrant multiplies the two and weighs each number with its `idf` modifier,
which together give BM25. Under index recipe "1" a passage's value was the count: while "2" re-indexes after a deploy,
the points still under "1" carry raw counts. The tokens themselves never leave this module: the vectors hold numbers
only, and nothing here logs.
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
# BM25's term-frequency saturation and length normalization.
K1 = 1.2
B = 0.75
# The average passage length, in tokens, that length normalization compares a passage with: a fixed constant, not the
# corpus's own average, so a passage's vector never depends on the other passages (each one is indexed alone, and an
# owner's library grows and shrinks). About 1,200 code points of prose (`ingestion.passages.PASSAGE_TARGET`).
KEYWORD_AVG_TOKENS = 250


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


def _counts(text: str, key: bytes) -> tuple[Counter[int], int]:
    """Each token number's count, and the number of tokens."""
    if len(key) != KEY_BYTES:
        raise ValueError("the token key is 32 bytes")
    found = tokens(text)
    return Counter(token_number(key, token) for token in found), len(found)


def keyword_vector(text: str, key: bytes) -> Sparse:
    """A passage's keyword vector under the owner's key: each token number with its BM25 term weight, ascending."""
    counts, length = _counts(text, key)
    norm = K1 * (1 - B + B * length / KEYWORD_AVG_TOKENS)
    numbers = sorted(counts)
    return Sparse(indices=numbers, values=[counts[number] * (K1 + 1) / (counts[number] + norm) for number in numbers])


def query_vector(text: str, key: bytes) -> Sparse:
    """A question's keyword vector under the owner's key: each distinct token number with the value 1, ascending."""
    counts, _length = _counts(text, key)
    numbers = sorted(counts)
    return Sparse(indices=numbers, values=[1.0] * len(numbers))
