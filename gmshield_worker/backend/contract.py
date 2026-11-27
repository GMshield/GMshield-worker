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
# 1.20.0 (Story 4.4 part A): a records step's report may carry `proposals`.
# 1.21.0 (Story 4.4 part B): a record type is any key of `[a-z][a-z0-9_]{0,31}`, a records step's input carries
# `recordTypeNames` and a question's or a search's input `recordTypes`. The worker reads all three as optional, so the
# minimum stays at 1.20.0 (deploy order: worker, then frontend, then backend).
# 1.22.0 (Story 4.5 part B): an `index.record_removal` job deletes a deleted record's points by id. Only a backend at
# 1.22.0 creates one, so the minimum stays at 1.20.0 (deploy order: worker, then backend, then frontend).
# 1.23.0 (Story 4.6): an answer resolves its question's terms (`POST …/terms`). A backend before 1.23.0 answers 404,
# and the search then runs as before, so the minimum stays at 1.20.0 (deploy order: backend, then worker).
# 1.24.0 (Story 5.1 part B): an `index.artifact` job indexes a GM's artifact as a records run of one record (kind
# `artifact`), and an `index.artifact_removal` job deletes its points. Only a backend at 1.24.0 creates either, so the
# minimum stays at 1.20.0 (deploy order: worker, then backend, then frontend).
# 1.25.0 (Story 5.3 part A): a `content.draft` job writes a draft the GM asked for, with the same input and reads as a
# `question.answer` job. Only a backend at 1.25.0 creates one, and the worker reads nothing new, so the minimum stays at
# 1.20.0 (deploy order: backend, then worker, then frontend).
# 1.26.0 (Story 5.3 part B): a `content.draft` input may carry `recordTypeNames`, and a draft's result carries its
# `proposal` {name, recordType}. The worker reads the names as optional (each key is its own name without them), and a
# backend before 1.26.0 ignores a draft's result, so the minimum stays at 1.20.0 (deploy order: backend, then worker).
MINIMUM_BACKEND_CONTRACT_VERSION = "1.20.0"
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
