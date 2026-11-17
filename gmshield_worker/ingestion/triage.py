"""The triage stage's child process (Story 1.16): say which pages of a batch have a text layer.

Run only through the sandbox (`sandbox.py`), as a script in isolated mode, inside the step folder:

    python -I -B triage.py SOURCE OUT_DIR OFFSET FIRST LAST

SOURCE is a page batch whose first page is page OFFSET of the version; FIRST to LAST (inclusive, numbered as in the
version) are the pages to look at. It imports pypdfium2 and the standard library only (never `gmshield_worker`: the
isolated child has no path to it), holds no credential, reaches no network, and writes only `OUT_DIR/manifest.json`:

    {"pages": [{"page": N, "layer": "text" | "image"}]}

A page whose text layer holds at least `MIN_CHARACTERS` characters other than whitespace has a text layer (`text`);
any other page is an image to read by OCR (`image`). Exit status 0 means done; anything else is a crash, which the
parent reports. Nothing is printed: the parent logs the status only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pypdfium2 as pdfium  # type: ignore[import-untyped]

MANIFEST = "manifest.json"
MIN_CHARACTERS = 16
TEXT = "text"
IMAGE = "image"


def layer_of(text: str) -> str:
    """`text` when the page's text layer has at least `MIN_CHARACTERS` characters other than whitespace."""
    count = 0
    for character in text:
        if not character.isspace():
            count += 1
            if count >= MIN_CHARACTERS:
                return TEXT
    return IMAGE


def run(source: Path, out: Path, offset: int, first: int, last: int) -> int:
    if offset < 1 or first < offset or last < first:
        return 2
    document = pdfium.PdfDocument(source)
    try:
        if last - offset + 1 > len(document):
            return 3
        pages = []
        for number in range(first, last + 1):
            page = document[number - offset]
            text_page = page.get_textpage()
            try:
                pages.append({"page": number, "layer": layer_of(text_page.get_text_range())})
            finally:
                text_page.close()
                page.close()
    finally:
        document.close()
    out.mkdir(exist_ok=True)
    (out / MANIFEST).write_text(json.dumps({"pages": pages}), encoding="utf-8")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) != 6:
        return 2
    try:
        offset, first, last = int(argv[3]), int(argv[4]), int(argv[5])
    except ValueError:
        return 2
    return run(Path(argv[1]), Path(argv[2]), offset, first, last)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
