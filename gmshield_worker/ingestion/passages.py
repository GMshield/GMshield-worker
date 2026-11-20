"""The passages stage's cutter (Story 1.18): a version's page records, in page order, as passages and an outline.

Pure functions over our own JSON: the stage (`step.py`) runs them in-process, one text file at a time, and carries the
section from one file to the next. Two kinds of page record come in:

- **PDF** (layout's and OCR's shape, see `layout.py`), with the file's bookmarks (`clean.py bookmarks`, pipeline "7").
  The section path is the path of the last bookmark entered (its titles in the outline, outermost first), followed by
  the last Docling heading since that bookmark, if any. Docling's heading levels never nest headings (Docling ranks a
  style per batch and per page retry, so one style gets several levels): every heading replaces the one before it under
  the current bookmark path. Bookmarks are entered in page order, in the outline's order within a page. On its page, a
  bookmark is entered just before the first block, at or after the previous bookmark's, whose text is its title
  (`folded`: NFKC, casefold, punctuation and whitespace runs as one space; a heading first, else any block); one that
  matches no block is entered with the next bookmark of its page that does, else at the page's start when none of the
  page's does, else at the page's end. A bookmark of a page that has no record is entered before the next page that
  has one. A heading whose folded title is already in the path (the current heading, as a running header repeats it,
  or any bookmark's title) is read as an ordinary block: "Tarrasque" under the bookmark "Tarrasque" is one
  "Tarrasque", and so is the stat block's own "Tarrasque" title after the monster's description. The outline has every
  bookmark at its depth (from 1) and every heading kept, one level under its bookmark path. With no bookmarks (a scan,
  an image set), the path is the last heading alone.
- **Text** (`text.py`'s shape): each page's `headingPath` is its section path; each heading a page newly enters is an
  outline entry whose level is its depth in the path.

Blocks labelled `page_header` or `page_footer`, and blocks with no text, never enter a passage. Cutting:

- a new passage starts at every bookmark and heading (a new section), at a new page (a passage lies on one page, so
  its page is exact), or when the next block would take it past `PASSAGE_TARGET` code points; a passage that opens its
  section (its first with content after a bookmark or heading, even on the page after a heading that ended a page)
  keeps taking blocks up to `PASSAGE_LIMIT` (the text stage's page limit), so a stat block or a spell stays whole;
- a passage that does not open its section starts with its section's leaf (the last title of its path), so each part
  of a long stat block repeats its heading; a passage that holds nothing but its own heading takes the next block
  anyway;
- a single block over the target is a passage of its own; one over the room a passage has (`PASSAGE_LIMIT`) is cut as
  `text.py` cuts a page: at its last blank line, else line end, else whitespace, else the limit;
- a table (its rows one per line, cells joined by ` | `) never splits while it fits; one that does not is cut between
  rows, each part starting with the heading and the table's first row;
- a text page is cut at its blank lines, its paragraphs joined by the same target and limit.

A PDF passage carries `boxes`, its blocks' boxes on its page (`[left, top, right, bottom]`, fractions from the
top-left corner); a text passage carries `range`, `[start, end)` in code points of its page's text (the repeated
heading is not in it). A title is its heading's text with runs of whitespace as one space; a "heading" longer than
`MAX_TITLE` is read as an ordinary block, and a path never grows past `MAX_DEPTH`.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gmshield_worker.ingestion import text as text_stage

PASSAGE_LIMIT = text_stage.PAGE_LIMIT
# The size passages are joined to (pipeline "7"): about a paragraph, so a page of prose is several passages.
PASSAGE_TARGET = 1200
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


def folded(value: object) -> str:
    """A title as it is compared: NFKC (ligatures, fullwidth), casefolded, every punctuation mark (quotes, dashes,
    a trailing period) as a space, whitespace runs as one space, trimmed."""
    if not isinstance(value, str):
        return ""
    normal = unicodedata.normalize("NFKC", value).casefold()
    return title("".join(" " if unicodedata.category(char).startswith("P") else char for char in normal))


@dataclass(frozen=True)
class Bookmark:
    """A bookmark of the file (`clean.py bookmarks`): its title, its depth from 0 in the outline, and its page."""

    title: str
    depth: int
    page: int


@dataclass(frozen=True)
class _Mark:
    """A bookmark ready to enter: its path (its titles in the outline, outermost first) and its page."""

    path: tuple[str, ...]
    page: int

    @property
    def key(self) -> str:
        return folded(self.path[-1])


def _marks(bookmarks: Sequence[Bookmark]) -> list[_Mark]:
    """Each bookmark with its path, worked out in the outline's order, then in page order (the outline's order within
    a page). A bookmark with no title is dropped, and its children take its place under its parent; a path is cut to
    leave room for a heading under it."""
    found: list[_Mark] = []
    # The open entries by depth: their titles, or None for one with no title (its children skip it).
    stack: list[tuple[int, str | None]] = []
    for bookmark in bookmarks:
        depth = max(bookmark.depth, 0)
        while stack and stack[-1][0] >= depth:
            stack.pop()
        name = title(bookmark.title)[:MAX_TITLE]
        stack.append((depth, name or None))
        if not name:
            continue
        names = [entry for _depth, entry in stack if entry is not None]
        path = tuple(names) if len(names) < MAX_DEPTH else (*names[: MAX_DEPTH - 2], names[-1])
        found.append(_Mark(path, bookmark.page))
    return sorted(found, key=lambda mark: mark.page)


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
    # Whether it is its section's first passage (after a bookmark or heading): it joins up to the limit.
    opens: bool = False
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

    def __init__(self, limit: int = PASSAGE_LIMIT, target: int = PASSAGE_TARGET) -> None:
        self.limit = limit
        self.target = min(target, limit)
        self.stack: list[tuple[int, str]] = []
        self.section = 0
        self.passages: list[dict[str, Any]] = []
        self.outline: list[dict[str, Any]] = []
        self._draft: _Draft | None = None

    def path(self) -> tuple[str, ...]:
        return tuple(name for _level, name in self.stack)

    def leaf(self) -> str | None:
        path = self.path()
        return path[-1] if path else None

    def cap(self, draft: _Draft) -> int:
        """The most a passage grows to by joining: the limit for one that opens its section, else the target."""
        return self.limit if draft.opens else self.target

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
        prefix = None if opens else self.leaf()
        self._draft = _Draft(page, self.path(), self.section, prefix, opens=opens, **text_page)
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
    """Passages of PDF page records (layout's and OCR's shape), under the file's bookmarks."""

    def __init__(
        self, limit: int = PASSAGE_LIMIT, target: int = PASSAGE_TARGET, bookmarks: Sequence[Bookmark] = ()
    ) -> None:
        super().__init__(limit, target)
        self._marks = _marks(bookmarks)
        self._next = 0
        self._bookmark_path: tuple[str, ...] = ()
        self._heading: str | None = None
        # The last sections that got a passage, and content beyond their heading.
        self._opened = 0
        self._filled = 0

    def path(self) -> tuple[str, ...]:
        return self._bookmark_path if self._heading is None else (*self._bookmark_path, self._heading)

    def finish(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """The passages and the outline, every bookmark in it (those of pages after the last record too)."""
        self._enter_before(None)
        return super().finish()

    def page(self, record: Record) -> None:
        page = record["page"]
        self._enter_before(page)
        blocks = record["blocks"]
        placed = self._place(page, blocks)
        for index, block in enumerate(blocks):
            for mark in placed.get(index, ()):
                self._mark(mark)
            if not isinstance(block, dict) or block.get("label") in SKIPPED:
                continue
            box = _box(block.get("box"))
            name = _heading(block)
            if name is not None and self._repeats(name):
                # A running header, or a title the path already holds: an ordinary block.
                name = None
            if name is not None:
                self._enter_heading(name, page)
                draft = self.open(page, opens=True)
                draft.lines.append(name)
                draft.heading_only = True
                draft.boxes.extend([box] if box else [])
            elif block.get("label") == TABLE:
                self._table(block, page, box)
            else:
                self._block(str(block.get("text") or "").strip(), page, box)
        for mark in placed.get(len(blocks), ()):
            self._mark(mark)

    def open(self, page: int, *, opens: bool, **text_page: Any) -> _Draft:
        if opens:
            self._opened = self.section
        return super().open(page, opens=opens, **text_page)

    def _new(self, page: int) -> _Draft:
        """A new passage: the first since a bookmark or heading opens its section, the others repeat its leaf. The
        first with content after a heading that ended a page alone still opens its section (it joins up to the limit),
        though it repeats the heading."""
        first = self.section > 0 and self._opened != self.section
        draft = self.open(page, opens=first)
        draft.opens = self.section > 0 and self._filled != self.section
        return draft

    # --- bookmarks and headings

    def _enter_before(self, page: int | None) -> None:
        """Enters every bookmark of the pages before `page` (all of them for None) not entered yet."""
        while self._next < len(self._marks) and (page is None or self._marks[self._next].page < page):
            self._mark(self._marks[self._next])
            self._next += 1

    def _place(self, page: int, blocks: list[Any]) -> dict[int, list[_Mark]]:
        """Where on the page each of its bookmarks is entered: before the block at that index (the page's end is the
        index past its last block)."""
        marks = []
        while self._next < len(self._marks) and self._marks[self._next].page == page:
            marks.append(self._marks[self._next])
            self._next += 1
        if not marks:
            return {}
        keys = [folded(block.get("text")) if isinstance(block, dict) and block.get("label") not in SKIPPED else ""
                for block in blocks]  # fmt: skip
        headings = [isinstance(block, dict) and _heading(block) is not None for block in blocks]
        placed: dict[int, list[_Mark]] = {}
        pending: list[_Mark] = []
        cursor = 0
        matched = False
        for mark in marks:
            pending.append(mark)
            found = _first(keys, mark.key, cursor, headings)
            if found is None:
                found = _first(keys, mark.key, cursor, None)
            if found is not None:
                placed.setdefault(found, []).extend(pending)
                pending = []
                cursor = found + 1
                matched = True
        if pending:
            placed.setdefault(len(blocks) if matched else 0, []).extend(pending)
        return placed

    def _mark(self, mark: _Mark) -> None:
        """A bookmark opens a section: its path replaces the path, and no heading is under it yet."""
        self.close()
        self._bookmark_path = mark.path
        self._heading = None
        self.section += 1
        self.outline.append({"level": min(len(mark.path), MAX_LEVEL), "title": mark.path[-1], "page": mark.page})

    def _repeats(self, name: str) -> bool:
        """Whether a heading's title is already in the path (the current heading, or any bookmark's title)."""
        key = folded(name)
        return any(folded(held) == key for held in self.path())

    def _enter_heading(self, name: str, page: int) -> None:
        """A heading opens a section: it replaces the heading under the bookmark path."""
        self._heading = name
        level = len(self._bookmark_path) + 1
        self.outline.append({"level": min(level, MAX_LEVEL), "title": name, "page": page})
        self.section += 1

    # --- cutting

    def _join(self, page: int, needed: int) -> _Draft:
        """The passage the next content of `needed` code points joins: the one being built while it stays within its
        cap (or holds nothing but its heading, or nothing at all); else a new one."""
        draft = self.current(page)
        if draft is not None and (draft.heading_only or draft.empty() or draft.size() + 1 + needed <= self.cap(draft)):
            return draft
        return self._new(page)

    def _add(self, draft: _Draft, line: str, box: Box | None) -> None:
        draft.lines.append(line)
        draft.heading_only = False
        self._filled = draft.section
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
            draft = self._new(page)
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
            joined = self._join(page, len(whole))
            if not joined.empty() and joined.size() + 1 + len(whole) > self.limit:
                # Measured on the passage's own text, never on its leaf's length.
                joined = self._new(page)
            self._add(joined, whole, box)
            return
        rows = table_rows(block)
        header, body = (rows[0], rows[1:]) if rows else ("", [])
        if not body or len(header) + 1 >= self.fresh_room() // 2:
            # No rows to cut between, or a first row too long to repeat: cut it as any other block.
            self._block(whole, page, box)
            return
        draft = self.current(page)
        if draft is None or not draft.heading_only:
            draft = self._new(page)
        part = [header]
        for row in body:
            pieces = [row]
            if len(header) + 1 + len(row) > self.fresh_room():
                pieces = self._row_pieces(row, self.fresh_room() - len(header) - 1)
            for piece in pieces:
                if len(part) > 1 and self._room(draft) < len("\n".join(part)) + 1 + len(piece):
                    self._add(draft, "\n".join(part), box)
                    draft = self._new(page)
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


def _heading(block: Record) -> str | None:
    """A heading block's title; None for any other block (a "heading" past `MAX_TITLE` is an ordinary block). Its
    level only says that it is a heading."""
    name = title(block.get("text"))
    return name if type(block.get("headingLevel")) is int and name and len(name) <= MAX_TITLE else None


def _first(keys: list[str], key: str, start: int, only: list[bool] | None) -> int | None:
    """The first index from `start` whose key is `key` (among those `only` marks, when given)."""
    for index in range(start, len(keys)):
        if keys[index] == key and (only is None or only[index]):
            return index
    return None


class TextCutter(_Cutter):
    """Passages of text page records (`text.py`'s shape)."""

    def __init__(self, limit: int = PASSAGE_LIMIT, target: int = PASSAGE_TARGET) -> None:
        super().__init__(limit, target)
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
        """A paragraph joins the passage being built while it stays within its cap (the limit for the passage that
        opens its section, else the target), else starts one; a paragraph alone is cut where it passes the limit.
        Answers whether a passage opened next would still open its section (only the page's first can)."""
        draft = self._draft
        while start < end:
            if draft is None:
                draft = self.open(page, opens=opens, source=source)
                opens = False
            prefix = len(draft.prefix) + 1 if draft.prefix is not None else 0
            first = draft.start if draft.start >= 0 else start
            cap = self.limit if draft.start < 0 or draft.opens else self.target
            if prefix + end - first <= cap:
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


def cutter(
    pdf: bool, limit: int = PASSAGE_LIMIT, target: int = PASSAGE_TARGET, bookmarks: Sequence[Bookmark] = ()
) -> PdfCutter | TextCutter:
    return PdfCutter(limit, target, bookmarks) if pdf else TextCutter(limit, target)


def read_bookmarks(path: Path, total: int) -> list[Bookmark]:
    """The clean child's bookmarks, checked. A body that is not JSON, or not a list of objects, is `RecordsInvalid`; an
    entry whose title is not text, whose depth is not an integer from 0, or whose page is not an integer within the
    version's is dropped, and the others kept."""
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RecordsInvalid("unreadable") from error
    if not isinstance(body, list) or not all(isinstance(item, dict) for item in body):
        raise RecordsInvalid("bookmarks")
    found = []
    for item in body:
        name, depth, page = item.get("title"), item.get("depth"), item.get("page")
        if isinstance(name, str) and type(depth) is int and depth >= 0 and type(page) is int and 1 <= page <= total:
            found.append(Bookmark(name, depth, page))
    return found


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
