"""The oldest backend internal contract (`info.version` of `contracts/internal.json`) this worker
works with. Against an older backend the worker stays up, unhealthy, and keeps checking.

When the worker starts using something a newer contract added, raise the constant in the same
change as the refreshed `contracts/internal.json`, unless the worker gates that use on the backend's
version (as `INDEX_RECIPE_CONTRACT_VERSION` does) and works without it against an older backend.
"""

from __future__ import annotations

import re

# 1.13.0 (Story 3.5): a question input lists the `other` tier, and a passage's tier may be `other`.
# 1.14.0 (Story 3.7): an answer's events may carry a `tier` mark.
# 1.15.0 (Story 3.8): a question input carries `modelKnowledge`, and a `tier` mark may be `model`.
# 1.16.0 (Story 3.10): a question input carries `history`, the conversation's earlier turns.
# 1.17.0 (Story 4.2): an ingestion records step (the input `passages`, `recordTypes`, a report's `records`) and the
# usage an outcome carries.
# 1.18.0 (Story 4.3 part B): a library search's input may carry `recordType`, and its result's ids may name records.
# 1.19.0 (Story 4.3 part C): an answer reads records by id (`POST …/records`) and a citation may name a `recordId`.
MINIMUM_BACKEND_CONTRACT_VERSION = "1.19.0"
# The first contract with `PUT /internal/v1/index-recipe`: the worker states its index recipe at every check only to a
# backend at least this new, and works as before with an older one (the minimum above stays).
INDEX_RECIPE_CONTRACT_VERSION = "1.12.0"

_VERSION = re.compile(r"(0|[1-9][0-9]{0,8})\.(0|[1-9][0-9]{0,8})\.(0|[1-9][0-9]{0,8})", re.ASCII)

type Version = tuple[int, int, int]


def parse_version(value: object) -> Version | None:
    """`major.minor.patch`, or `None` for anything else (pre-releases included)."""
    if not isinstance(value, str):
        return None
    match = _VERSION.fullmatch(value)
    if match is None:
        return None
    return (int(match[1]), int(match[2]), int(match[3]))


def meets_minimum(reported: object, minimum: str = MINIMUM_BACKEND_CONTRACT_VERSION) -> bool:
    """Whether a reported version is valid and at least the minimum."""
    floor = parse_version(minimum)
    if floor is None:
        raise ValueError("The minimum contract version is not major.minor.patch")
    version = parse_version(reported)
    return version is not None and version >= floor
