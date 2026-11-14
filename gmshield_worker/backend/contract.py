"""The oldest backend internal contract (`info.version` of `contracts/internal.json`) this worker
works with. Against an older backend the worker stays up, unhealthy, and keeps checking.

When the worker starts using something a newer contract added, raise the constant in the same
change as the refreshed `contracts/internal.json`.
"""

from __future__ import annotations

import re

MINIMUM_BACKEND_CONTRACT_VERSION = "1.0.0"

_VERSION = re.compile(r"(0|[1-9]\d{0,8})\.(0|[1-9]\d{0,8})\.(0|[1-9]\d{0,8})")

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
