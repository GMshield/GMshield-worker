from __future__ import annotations

import pytest

from gmshield_worker.backend.contract import MINIMUM_BACKEND_CONTRACT_VERSION, meets_minimum, parse_version


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1.0.0", (1, 0, 0)),
        ("10.20.30", (10, 20, 30)),
        ("1.0", None),
        ("01.0.0", None),
        ("1.0.0-rc.1", None),
        ("1\u0661.0.0", None),
        ("\u0661.0.0", None),
        (" 1.0.0", None),
        (None, None),
        (100, None),
    ],
)
def test_parse_version(value: object, expected: tuple[int, int, int] | None) -> None:
    assert parse_version(value) == expected


@pytest.mark.parametrize(
    ("reported", "minimum", "expected"),
    [
        ("1.0.0", "1.0.0", True),
        ("1.0.1", "1.0.0", True),
        ("1.10.0", "1.9.9", True),
        ("2.0.0", "1.99.99", True),
        ("0.9.0", "1.0.0", False),
        ("1.0.0", "1.0.1", False),
        ("garbage", "1.0.0", False),
    ],
)
def test_meets_minimum(reported: str, minimum: str, expected: bool) -> None:
    assert meets_minimum(reported, minimum) is expected


def test_minimum_is_a_valid_version() -> None:
    assert parse_version(MINIMUM_BACKEND_CONTRACT_VERSION) is not None
    with pytest.raises(ValueError, match=r"major\.minor\.patch"):
        meets_minimum("1.0.0", "one")
