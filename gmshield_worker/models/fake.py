"""The fake model: the same prompt always gives the same answer, and nothing leaves the process."""

from __future__ import annotations

import hashlib


class FakeModelClient:
    @property
    def kind(self) -> str:
        return "fake"

    def complete(self, system: str, prompt: str, *, max_tokens: int = 1024) -> str:
        digest = hashlib.sha256(f"{system}\x00{prompt}".encode()).hexdigest()[:16]
        return f"fake-answer-{digest}"[: max(max_tokens, 1) * 4]
