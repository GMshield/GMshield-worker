"""The text stage's child process (Story 1.16): split a text or Markdown file into pages of bounded length.

Run only through the sandbox (`sandbox.py`), as a script in isolated mode, inside the step folder:

    python -I -B text.py SOURCE OUT_DIR FORMAT

FORMAT is `txt` or `markdown`. The file is streamed, never held whole: it is decoded as UTF-8 (a leading byte order
mark dropped) or, when its bytes are not valid UTF-8, as cp1252 (the five bytes cp1252 leaves undefined become
U+FFFD). Its text is cut into pages of at most `PAGE_LIMIT` code points:

- in Markdown, every ATX heading (`#` to `######`, outside fenced code) starts a page, and each page carries its
  heading path (the titles of the headings it sits under, outermost first); plain text has no headings;
- a section longer than the limit is cut at its last blank line that fits, else at its last line end, else at its last
  whitespace, else at the limit itself.

Pages cover the text whole and in order. It imports the standard library only, holds no credential, reaches no
network, and writes only under OUT_DIR: `text-NNNNN.json` for every `PAGES_PER_FILE` pages,
`{"pages": [{"page", "headingPath", "text", "start", "end"}]}` (`start` and `end` are code-point offsets in the
decoded text, end excluded), and `manifest.json`, `{"pageTotal": N, "texts": [{"file", "firstPage", "lastPage"}]}`.
Exit status 0 means done, `EXIT_UNREADABLE` that the file holds no text at all; anything else is a crash.
"""

from __future__ import annotations

import codecs
import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import IO, Any

EXIT_UNREADABLE = 11
MANIFEST = "manifest.json"
PAGE_LIMIT = 3000
PAGES_PER_FILE = 500
READ_BYTES = 1 << 20
MARKDOWN = "markdown"
FORMATS = frozenset({"txt", MARKDOWN})

_HEADING = re.compile(r" {0,3}(#{1,6})(?:[ \t]+(.*?))??(?:[ \t]+#+)?[ \t]*\r?\n?")
_FENCE = re.compile(r" {0,3}(`{3,}|~{3,})")
_BLANK_LINE = re.compile(r"\n[ \t\r\f\v]*\n")


def text_file(index: int) -> str:
    return f"text-{index:05d}.json"


def _decode(source: Path, encoding: str) -> Iterator[str]:
    """The file's text, chunk by chunk; raises `UnicodeDecodeError` (UTF-8 only) on the first invalid byte."""
    errors = "strict" if encoding == "utf-8-sig" else "replace"
    decoder = codecs.getincrementaldecoder(encoding)(errors=errors)
    with source.open("rb") as stream:
        while chunk := stream.read(READ_BYTES):
            text = decoder.decode(chunk)
            if text:
                yield text
        rest = decoder.decode(b"", final=True)
        if rest:
            yield rest


def _valid_utf8(source: Path) -> bool:
    try:
        for _text in _decode(source, "utf-8-sig"):
            pass
    except UnicodeDecodeError:
        return False
    return True


def decoded(source: Path) -> Iterator[str]:
    """UTF-8 (BOM dropped) when every byte is valid UTF-8, else cp1252: one pass to check, one to read."""
    return _decode(source, "utf-8-sig" if _valid_utf8(source) else "cp1252")


def lines(chunks: Iterator[str], limit: int = PAGE_LIMIT) -> Iterator[tuple[str, bool]]:
    """Pieces of the text, each a line with its end, or part of a line longer than `limit`; with whether the piece
    starts a line. No piece is longer than `limit`, so memory stays bounded whatever the file."""
    pending = ""
    at_start = True
    for chunk in chunks:
        pending = pending + chunk if pending else chunk
        position, size = 0, len(pending)
        while True:
            end = pending.find("\n", position, min(position + limit, size))
            if end >= 0:
                yield pending[position : end + 1], at_start
                position, at_start = end + 1, True
            elif size - position >= limit:
                yield pending[position : position + limit], at_start
                position, at_start = position + limit, False
            else:
                break
        pending = pending[position:]
    if pending:
        yield pending, at_start


def pieces(chunks: Iterator[str], limit: int = PAGE_LIMIT) -> Iterator[str]:
    """The text in pieces of at most `limit` code points, wherever they fall (plain text needs no lines)."""
    for chunk in chunks:
        for start in range(0, len(chunk), limit):
            yield chunk[start : start + limit]


def cut(text: str, limit: int = PAGE_LIMIT) -> int:
    """Where to end a page taken from the front of `text` (longer than `limit`): after its last blank line within the
    limit, else after its last line end, else after its last whitespace, else at the limit."""
    window = text[:limit]
    blanks = list(_BLANK_LINE.finditer(window))
    if blanks:
        return blanks[-1].end()
    line_end = window.rfind("\n")
    if line_end >= 0:
        return line_end + 1
    for index in range(len(window) - 1, 0, -1):
        if window[index].isspace():
            return index + 1
    return limit


class Pager:
    """Builds the pages and writes them, `PAGES_PER_FILE` to a file."""

    def __init__(self, out: Path, limit: int = PAGE_LIMIT, per_file: int = PAGES_PER_FILE) -> None:
        self.out = out
        self.limit = limit
        self.per_file = per_file
        self.pending = ""
        self.offset = 0
        self.path: list[tuple[int, str]] = []
        self.page = 0
        self.texts: list[dict[str, Any]] = []
        self._file: IO[str] | None = None
        self._file_first = 0

    def add(self, text: str) -> None:
        self.pending += text
        while len(self.pending) > self.limit:
            self._emit(cut(self.pending, self.limit))

    def heading(self, level: int, title: str) -> None:
        """A heading starts a page: what came before ends, under the path it had."""
        self.flush()
        while self.path and self.path[-1][0] >= level:
            self.path.pop()
        self.path.append((level, title))

    def flush(self) -> None:
        while len(self.pending) > self.limit:
            self._emit(cut(self.pending, self.limit))
        if self.pending:
            self._emit(len(self.pending))

    def close(self) -> int:
        self.flush()
        self._close_file()
        return self.page

    def _emit(self, length: int) -> None:
        text, self.pending = self.pending[:length], self.pending[length:]
        self.page += 1
        if self._file is None:
            self._file_first = self.page
            self._file = (self.out / text_file(len(self.texts) + 1)).open("w", encoding="utf-8")
            self._file.write('{"pages": [')
        else:
            self._file.write(", ")
        record = {
            "page": self.page,
            "headingPath": [title for _level, title in self.path],
            "text": text,
            "start": self.offset,
            "end": self.offset + len(text),
        }
        self._file.write(json.dumps(record, ensure_ascii=False))
        self.offset += len(text)
        if self.page - self._file_first + 1 >= self.per_file:
            self._close_file()

    def _close_file(self) -> None:
        if self._file is None:
            return
        self._file.write("]}")
        self._file.close()
        self._file = None
        self.texts.append(
            {"file": text_file(len(self.texts) + 1), "firstPage": self._file_first, "lastPage": self.page}
        )


def _closes(piece: str, opened: re.Match[str], fence: str) -> bool:
    """Whether a fence line closes the open fence: the same character, at least as long, nothing after it."""
    return opened[1][0] == fence[0] and len(opened[1]) >= len(fence) and not piece[opened.end() :].strip()


def paginate(chunks: Iterator[str], markdown: bool, pager: Pager) -> int:
    """Feeds the text to `pager`, starting a page at each Markdown heading; returns the page count."""
    if not markdown:
        for piece in pieces(chunks, pager.limit):
            pager.add(piece)
        return pager.close()
    fence: str | None = None
    for piece, at_start in lines(chunks, pager.limit):
        if at_start:
            opened = _FENCE.match(piece)
            if fence is not None:
                if opened and _closes(piece, opened, fence):
                    fence = None
            elif opened:
                fence = opened[1]
            elif heading := _HEADING.fullmatch(piece):
                pager.heading(len(heading[1]), (heading[2] or "").strip())
        pager.add(piece)
    return pager.close()


def run(source: Path, out: Path, format_name: str) -> int:
    if format_name not in FORMATS:
        return 2
    out.mkdir(exist_ok=True)
    pager = Pager(out)
    total = paginate(decoded(source), format_name == MARKDOWN, pager)
    if total == 0:
        return EXIT_UNREADABLE
    (out / MANIFEST).write_text(json.dumps({"pageTotal": total, "texts": pager.texts}), encoding="utf-8")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        return 2
    return run(Path(argv[1]), Path(argv[2]), argv[3])


if __name__ == "__main__":
    sys.exit(main(sys.argv))
