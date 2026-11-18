"""The triage stage's child process (Stories 1.16 and 1.17): say which pages of a batch have a text layer.

Run only through the sandbox (`sandbox.py`), as a script in isolated mode, inside the step folder:

    python -I -B triage.py SOURCE OUT_DIR OFFSET FIRST LAST

SOURCE is a page batch whose first page is page OFFSET of the version; FIRST to LAST (inclusive, numbered as in the
version) are the pages to look at. It imports pypdfium2 and the standard library only (never `gmshield_worker`: the
isolated child has no path to it), holds no credential, reaches no network, and writes only `OUT_DIR/manifest.json`:

    {"pages": [{"page": N, "layer": "text" | "image"}]}

A page whose text layer holds at least `MIN_CHARACTERS` characters other than whitespace has a text layer (`text`),
unless it holds fewer than `FEW_CHARACTERS` and one image covers at least `IMAGE_SHARE` of the page: a scan with a
digital running header or caption, or a digital page with a big illustration and a short caption, is read by OCR
(Story 1.17). Any other page is an image to read by OCR (`image`). An image's share is its bounds clipped to the
page's crop box; an image inside a form XObject counts with the bounds PDFium gives it. Exit status 0 means done;
anything else is a crash, which the parent reports. Nothing is printed: the parent logs the status only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium  # type: ignore[import-untyped]
import pypdfium2.raw as pdfium_c  # type: ignore[import-untyped]

MANIFEST = "manifest.json"
MIN_CHARACTERS = 16
FEW_CHARACTERS = 200
IMAGE_SHARE = 0.5
TEXT = "text"
IMAGE = "image"


def characters(text: str, limit: int = FEW_CHARACTERS) -> int:
    """The characters other than whitespace in `text`, counted up to `limit`."""
    count = 0
    for character in text:
        if not character.isspace():
            count += 1
            if count >= limit:
                break
    return count


def layer_of(text: str, image_share: float = 0.0) -> str:
    """`text` when the page's text layer has at least `MIN_CHARACTERS` characters other than whitespace, unless it has
    fewer than `FEW_CHARACTERS` and an image covers at least `IMAGE_SHARE` of the page (`image_share`)."""
    return layer_for(characters(text), image_share)


def layer_for(count: int, image_share: float = 0.0) -> str:
    """`layer_of` for a text layer of `count` characters other than whitespace (counted up to `FEW_CHARACTERS`)."""
    if count < MIN_CHARACTERS:
        return IMAGE
    if count < FEW_CHARACTERS and image_share >= IMAGE_SHARE:
        return IMAGE
    return TEXT


def share(bounds: tuple[float, float, float, float], box: tuple[float, float, float, float]) -> float:
    """The share of `box` (left, bottom, right, top) that `bounds` covers once clipped to it, from 0 to 1."""
    left, bottom, right, top = box
    area = (right - left) * (top - bottom)
    if area <= 0:
        return 0.0
    width = min(bounds[2], right) - max(bounds[0], left)
    height = min(bounds[3], top) - max(bounds[1], bottom)
    if width <= 0 or height <= 0:
        return 0.0
    return min(width * height / area, 1.0)


def _box(values: Any) -> tuple[float, float, float, float]:
    left, bottom, right, top = values
    return float(left), float(bottom), float(right), float(top)


def largest_image_share(page: Any) -> float:
    """The largest share of the page's crop box that one image object covers."""
    box = _box(page.get_cropbox())
    largest = 0.0
    for image in page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE]):
        largest = max(largest, share(_box(image.get_bounds()), box))
    return largest


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
                text = text_page.get_text_range()
                # Images are looked at only when the text alone would make it a text page with few characters.
                count = characters(text)
                image_share = largest_image_share(page) if MIN_CHARACTERS <= count < FEW_CHARACTERS else 0.0
                pages.append({"page": number, "layer": layer_for(count, image_share)})
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
