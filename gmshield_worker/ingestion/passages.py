"""The passages stage's cutter (Story 1.18): a version's page records, in page order, as passages and an outline.

Pure functions over our own JSON: the stage (`step.py`) runs them in-process, with no child, one text file at a time,
and carries the heading stack from one file to the next. Two kinds of page record come in:

- **PDF** (layout's and OCR's shape, see `layout.py`): blocks in reading order. A block whose `headingLevel` is set
  closes every open section at its level or deeper and opens its own; the section path is the open headings' titles,
  outermost first, and every heading with text is an outline entry `{level, title, page}`.
- **Text** (`text.py`'s shape): each page's `headingPath` is its section path; each heading a page newly enters is an
  outline entry whose level is its depth in the path.

Blocks labelled `page_header` or `page_footer`, and blocks with no text, never enter a passage. Cutting, at
`PASSAGE_LIMIT` code points (the text stage's page limit):

- a new passage starts at a new section, at a new page (a passage lies on one page, so its page is exact), or when the
  next block would pass the limit;
- a passage that does not open its section starts with its section's heading (the leaf of its path), so each part of a
  long stat block repeats its heading; a passage that holds nothing but its own heading takes the next block anyway;
- a table (its rows one per line, cells joined by ` | `) never splits while it fits; one that does not is cut between
  rows, each part starting with the heading and the table's first row;
- any other block over the limit is cut as `text.py` cuts a page: at its last blank line, else line end, else
  whitespace, else the limit;
- a text page is cut at its blank lines.

A PDF passage carries `boxes`, its blocks' boxes on its page (`[left, top, right, bottom]`, fractions from the
top-left corner); a text passage carries `range`, `[start, end)` in code points of its page's text (the repeated
heading is not in it). A title is its heading's text with runs of whitespace as one space; a "heading" longer than
`MAX_TITLE` is read as an ordinary block, and a path never grows past `MAX_DEPTH`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gmshield_worker.ingestion import text as text_stage

PASSAGE_LIMIT = text_stage.PAGE_LIMIT
MAX_DEPTH = 32
MAX_TITLE = 1000
MAX_LEVEL = 1000
SKIPPED = frozenset({"page_header", "page_footer"})
TABLE = "table"
CELL_SEPARATOR = " | "
_SPACES = re.compile(r"\s+")
_BLANK_LINE = re.compile(r"\n[ \t\r\f\v]*\n")

type Box = list[float]
type Record = dict[str, Any]


class RecordsInvalid(ValueError):
    """A text file that is not the page records it should be."""


def title(value: object) -> str:
    """A heading's text as a title: whitespace runs as one space, trimmed; empty for anything but text."""
    return _SPACES.sub(" ", value).strip() if isinstance(value, str) else ""


def table_text(block: Record) -> str:
    """A table as text: its rows one per line, cells joined by ` | `; its text when it has no rows."""
    rows = table_rows(block)
    return "\n".join(rows) if rows else str(block.get("text") or "").strip()


def table_rows(block: Record) -> list[str]:
    """A table's rows that hold text, each a line of its cells."""
    rows = block.get("rows")
    if not isinstance(rows, list):
        return []
    lines = []
    for row in rows:
        if not isinstance(row, list):
            continue
        cells = [title(cell) for cell in row]
        if any(cells):
            lines.append(CELL_SEPARATOR.join(cells))
    return lines


def _box(value: object) -> Box | None:
    if (
        isinstance(value, list)
        and len(value) == 4
        and all(type(side) in (int, float) and 0 <= side <= 1 for side in value)
    ):
        return [float(side) for side in value]
    return None


def split(text: str, room: int) -> tuple[str, str]:
    """The front of `text` that fits in `room` code points, cut as `text.py` cuts a page, and the rest; both trimmed
    where they meet."""
    end = text_stage.cut(text, room)
    return text[:end].rstrip(), text[end:].lstrip()


@dataclass
class _Draft:
    """The passage being built: one page, one section; its prefix is the repeated heading, if any."""

    page: int
    path: tuple[str, ...]
    section: int
    prefix: str | None
    lines: list[str] = field(default_factory=list)
    boxes: list[Box] = field(default_factory=list)
    heading_only: bool = False
    # Text pages: the range of the page's text the passage holds.
    source: str = ""
    start: int = -1
    end: int = -1

    def text(self) -> str:
        body = self.source[self.start : self.end] if self.start >= 0 else "\n".join(self.lines)
        if self.prefix is None:
            return body
        return f"{self.prefix}\n{body}" if body else self.prefix

    def size(self) -> int:
        return len(self.text())

    def empty(self) -> bool:
        return not self.lines and self.start < 0


class _Cutter:
    """What both kinds share: the open sections, the outline, the passages, and the passage being built."""

    def __init__(self, limit: int = PASSAGE_LIMIT) -> None:
        self.limit = limit
        self.stack: list[tuple[int, str]] = []
        self.section = 0
        self.passages: list[dict[str, Any]] = []
        self.outline: list[dict[str, Any]] = []
        self._draft: _Draft | None = None

    def path(self) -> tuple[str, ...]:
        return tuple(name for _level, name in self.stack)

    def leaf(self) -> str | None:
        return self.stack[-1][1] if self.stack else None

    def fresh_room(self) -> int:
        """The code points a new passage that repeats the heading has left for its content."""
        leaf = self.leaf()
        return self.limit - (len(leaf) + 1 if leaf is not None else 0)

    def current(self, page: int) -> _Draft | None:
        """The passage being built, when the next content of `page` may join it."""
        draft = self._draft
        if draft is not None and draft.page == page and draft.section == self.section:
            return draft
        self.close()
        return None

    def open(self, page: int, *, opens: bool, **text_page: Any) -> _Draft:
        self.close()
        self._draft = _Draft(page, self.path(), self.section, None if opens else self.leaf(), **text_page)
        return self._draft

    def close(self) -> None:
        draft, self._draft = self._draft, None
        if draft is None or draft.empty():
            return
        passage: dict[str, Any] = {"page": draft.page, "sectionPath": list(draft.path), "text": draft.text()}
        if draft.start >= 0:
            passage["range"] = [draft.start, draft.end]
        else:
            passage["boxes"] = draft.boxes
        self.passages.append(passage)

    def finish(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """The passages in reading order and the outline; the cutter is spent."""
        self.close()
        return self.passages, self.outline

    def enter(self, level: int, name: str, page: int) -> None:
        """A heading opens a section: those at its level or deeper close, and it is an outline entry."""
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        if len(self.stack) >= MAX_DEPTH:
            self.stack.pop()
        self.stack.append((level, name))
        self.section += 1
        self.outline.append({"level": min(max(level, 0), MAX_LEVEL), "title": name, "page": page})


class PdfCutter(_Cutter):
    """Passages of PDF page records (layout's and OCR's shape)."""

    def page(self, record: Record) -> None:
        page = record["page"]
        for block in record["blocks"]:
            if not isinstance(block, dict) or block.get("label") in SKIPPED:
                continue
            box = _box(block.get("box"))
            level = block.get("headingLevel")
            name = title(block.get("text"))
            if type(level) is int and name and len(name) <= MAX_TITLE:
                self.enter(level, name, page)
                draft = self.open(page, opens=True)
                draft.lines.append(name)
                draft.heading_only = True
                draft.boxes.extend([box] if box else [])
            elif block.get("label") == TABLE:
                self._table(block, page, box)
            else:
                self._block(str(block.get("text") or "").strip(), page, box)

    def _join(self, page: int, needed: int) -> _Draft:
        """The passage the next content of `needed` code points joins: the one being built when it fits, or one that
        holds only its heading; else a new one that repeats the heading."""
        draft = self.current(page)
        if draft is not None and (draft.heading_only or draft.size() + 1 + needed <= self.limit):
            return draft
        return self.open(page, opens=False)

    def _add(self, draft: _Draft, line: str, box: Box | None) -> None:
        draft.lines.append(line)
        draft.heading_only = False
        if box is not None:
            draft.boxes.append(box)

    def _room(self, draft: _Draft) -> int:
        return self.limit - draft.size() - (1 if draft.size() else 0)

    def _block(self, text: str, page: int, box: Box | None) -> None:
        """A block joins the passage being built, or starts one; one over the room left is cut, each part after the
        first in a passage of its own."""
        if not text:
            return
        draft = self._join(page, len(text))
        while len(text) > self._room(draft):
            front, text = split(text, self._room(draft))
            if front:
                self._add(draft, front, box)
            draft = self.open(page, opens=False)
            if not text:
                return
        self._add(draft, text, box)

    def _table(self, block: Record, page: int, box: Box | None) -> None:
        """A table stays whole while it fits; else its rows are cut into parts that each start with the heading and
        the table's first row."""
        whole = table_text(block)
        if not whole:
            return
        if len(whole) <= self.fresh_room():
            self._add(self._join(page, len(whole)), whole, box)
            return
        rows = table_rows(block)
        header, body = (rows[0], rows[1:]) if rows else ("", [])
        if not body or len(header) + 1 >= self.fresh_room() // 2:
            # No rows to cut between, or a first row too long to repeat: cut it as any other block.
            self._block(whole, page, box)
            return
        draft = self.current(page)
        if draft is None or not draft.heading_only:
            draft = self.open(page, opens=False)
        part = [header]
        for row in body:
            pieces = [row]
            if len(header) + 1 + len(row) > self.fresh_room():
                pieces = self._row_pieces(row, self.fresh_room() - len(header) - 1)
            for piece in pieces:
                if len(part) > 1 and self._room(draft) < len("\n".join(part)) + 1 + len(piece):
                    self._add(draft, "\n".join(part), box)
                    draft = self.open(page, opens=False)
                    part = [header]
                part.append(piece)
        self._add(draft, "\n".join(part), box)

    @staticmethod
    def _row_pieces(row: str, room: int) -> list[str]:
        """A row longer than a part can hold, cut as a block is."""
        pieces = []
        while len(row) > room:
            front, row = split(row, room)
            if front:
                pieces.append(front)
        if row:
            pieces.append(row)
        return pieces


class TextCutter(_Cutter):
    """Passages of text page records (`text.py`'s shape)."""

    def __init__(self, limit: int = PASSAGE_LIMIT) -> None:
        super().__init__(limit)
        self._previous: list[str] = []

    def page(self, record: Record) -> None:
        page = record["page"]
        source = record["text"]
        names = [name[:MAX_TITLE] for name in (title(value) for value in record["headingPath"]) if name][:MAX_DEPTH]
        opens = False
        if names != self._previous:
            common = 0
            while common < min(len(self._previous), len(names)) and self._previous[common] == names[common]:
                common += 1
            self.stack = [(depth + 1, name) for depth, name in enumerate(names[:common])]
            for depth in range(common, len(names)):
                self.enter(depth + 1, names[depth], page)
            # A page that enters a heading opens its section; one that only backs out of headings does not.
            opens = len(names) > common
            self._previous = names
        self.close()
        for start, end in _paragraphs(source):
            opens = self._paragraph(source, page, start, end, opens=opens)
        self.close()

    def _paragraph(self, source: str, page: int, start: int, end: int, *, opens: bool) -> bool:
        """A paragraph joins the passage being built while it fits, else starts one, cut where it passes the limit.
        Answers whether a passage opened next would still open its section (only the page's first can)."""
        draft = self._draft
        while start < end:
            if draft is None:
                draft = self.open(page, opens=opens, source=source)
                opens = False
            prefix = len(draft.prefix) + 1 if draft.prefix is not None else 0
            first = draft.start if draft.start >= 0 else start
            if prefix + end - first <= self.limit:
                draft.start, draft.end = first, end
                return opens
            if draft.start >= 0:
                # The paragraph does not fit beside what the passage holds: the passage ends before it.
                self.close()
                draft = None
                continue
            cut = start + text_stage.cut(source[start:end], self.limit - prefix)
            draft.start, draft.end = start, _trim_end(source, start, cut)
            self.close()
            draft = None
            start = _trim_start(source, cut, end)
        return opens


def _trim_start(source: str, start: int, end: int) -> int:
    while start < end and source[start].isspace():
        start += 1
    return start


def _trim_end(source: str, start: int, end: int) -> int:
    while end > start and source[end - 1].isspace():
        end -= 1
    return end


def _paragraphs(source: str) -> list[tuple[int, int]]:
    """The page's text between blank lines, as ranges without their surrounding whitespace."""
    found = []
    start = 0
    for blank in [*_BLANK_LINE.finditer(source), None]:
        stop = blank.start() if blank is not None else len(source)
        first, last = _trim_start(source, start, stop), _trim_end(source, start, stop)
        if first < last:
            found.append((first, last))
        if blank is not None:
            start = blank.end()
    return found


def cutter(pdf: bool, limit: int = PASSAGE_LIMIT) -> PdfCutter | TextCutter:
    return PdfCutter(limit) if pdf else TextCutter(limit)


def read_records(path: Path, first: int, last: int, *, pdf: bool) -> list[Record]:
    """A text file's page records, checked: an object `{"pages": [...]}` whose pages lie in `first` to `last`, in
    increasing order, each in its kind's shape."""
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RecordsInvalid("unreadable") from error
    pages = body.get("pages") if isinstance(body, dict) else None
    if not isinstance(pages, list):
        raise RecordsInvalid("no pages")
    previous = first - 1
    for record in pages:
        if not isinstance(record, dict):
            raise RecordsInvalid("page")
        page = record.get("page")
        if type(page) is not int or not previous < page <= last:
            raise RecordsInvalid("page")
        previous = page
        if pdf and not isinstance(record.get("blocks"), list):
            raise RecordsInvalid("blocks")
        if not pdf and (
            not isinstance(record.get("text"), str)
            or not isinstance(record.get("headingPath"), list)
            or not all(isinstance(name, str) for name in record["headingPath"])
        ):
            raise RecordsInvalid("text")
    return pages
