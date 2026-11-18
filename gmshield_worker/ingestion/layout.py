"""The layout stage's child process (Story 1.16): read the text pages of a batch with Docling.

Run only through the sandbox (`sandbox.py`), as a script in isolated mode, inside the step folder:

    python -I -B layout.py SOURCE OUT_DIR OFFSET FIRST LAST ARTIFACTS THREADS

SOURCE is a page batch whose first page is page OFFSET of the version; FIRST to LAST (inclusive, numbered as in the
version) are the pages to read. The child's environment is empty, so everything Docling needs comes on the command
line: ARTIFACTS is the folder of its layout and table models (baked into the image: nothing is downloaded) and THREADS
the number of threads torch and Docling may use. Docling runs without OCR (`do_ocr=False`), with table structure, and
infers heading levels (from the PDF's bookmarks, numbering and font style).

Each page read is also rendered with pypdfium2 (Story 1.19) at the scale whose longer side is `PICTURE_SIDE` pixels,
never more than `MAX_PICTURE_SCALE` (the OCR stage's own scale), and written as a WebP picture of quality
`PICTURE_QUALITY`: the page view's picture, over which its lines' boxes lie.

It imports Docling, torch, pypdfium2, Pillow and the standard library only (never `gmshield_worker`), holds no
credential, reaches no network, and writes only under OUT_DIR: one `page-NNNNNNN.json` and one `page-NNNNNNN.webp`
per page read, and `manifest.json` listing them, `{"pages": [N, ...], "pictures": [{"page", "width", "height"}]}`.
A page record is

    {"page": N, "lines": [{"text", "box"}],
     "blocks": [{"label", "text", "box", "headingLevel", "rows"}]}

where a box is `[left, top, right, bottom]` as fractions of the page from its top-left corner, lines are the text
layer's lines as Docling's parser found them, and blocks are Docling's items in reading order: `headingLevel` is set
for a title (0) or a section header (its level), `rows` for a table (the text of each cell, row by row). Exit status 0
means done; anything else is a crash, which the parent reports. Nothing is printed.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pypdfium2 as pdfium  # type: ignore[import-untyped]
from PIL import Image

MANIFEST = "manifest.json"
DIGITS = 4
# A page picture's longer side, in pixels (Story 1.19): zoom stays sharp to about twice the 560-pixel source view.
PICTURE_SIDE = 2000
# The largest scale a page is rendered at: the OCR stage's (`ocr.SCALE`, 216 DPI), so a small page is never blown up.
MAX_PICTURE_SCALE = 3.0
PICTURE_QUALITY = 80


def page_file(page: int) -> str:
    return f"page-{page:07d}.json"


def picture_file(page: int) -> str:
    return f"page-{page:07d}.webp"


def picture_scale(width: float, height: float) -> float:
    """The scale that renders a page of `width` x `height` points with its longer side at `PICTURE_SIDE` pixels, never
    above `MAX_PICTURE_SCALE`."""
    return min(MAX_PICTURE_SCALE, PICTURE_SIDE / max(width, height, 1.0))


def write_webp(picture: np.ndarray, target: Path) -> tuple[int, int]:
    """A picture as WebP of quality `PICTURE_QUALITY`, shrunk (never enlarged) to a longer side of `PICTURE_SIDE`
    pixels; answers its width and height."""
    image = Image.fromarray(picture)
    factor = PICTURE_SIDE / max(image.width, image.height, 1)
    if factor < 1.0:
        size = (max(1, round(image.width * factor)), max(1, round(image.height * factor)))
        image = image.resize(size, Image.Resampling.LANCZOS)
    image.save(target, format="WEBP", quality=PICTURE_QUALITY)
    return image.width, image.height


def render_picture(document: Any, index: int, target: Path) -> tuple[int, int]:
    """Page `index` (from 0) of a pypdfium2 document, rendered at `picture_scale` and written as WebP; its size."""
    page = document[index]
    try:
        width, height = page.get_size()
        bitmap = page.render(scale=picture_scale(width, height), rev_byteorder=True)
        try:
            picture = np.array(bitmap.to_numpy()[:, :, :3], dtype=np.uint8)
        finally:
            bitmap.close()
    finally:
        page.close()
    return write_webp(picture, target)


def write_pictures(source: Path, out: Path, offset: int, pages: list[int]) -> list[dict[str, int]]:
    """Each page's picture (numbered as in the version; the batch's first page is `offset`), one page at a time."""
    document = pdfium.PdfDocument(source)
    try:
        pictures = []
        for page in pages:
            width, height = render_picture(document, page - offset, out / picture_file(page))
            pictures.append({"page": page, "width": width, "height": height})
        return pictures
    finally:
        document.close()


def fraction_box(box: tuple[float, float, float, float], width: float, height: float) -> list[float]:
    """A box `(left, top, right, bottom)` in points from the top-left corner, as fractions of the page, clamped to 0..1
    and rounded."""

    def clamp(value: float, size: float) -> float:
        return round(min(max(value / size, 0.0), 1.0), DIGITS) if size > 0 else 0.0

    left, top, right, bottom = box
    return [clamp(left, width), clamp(top, height), clamp(right, width), clamp(bottom, height)]


def top_left_box(bbox: Any, width: float, height: float) -> list[float]:
    """A Docling `BoundingBox` (either origin) as fractions from the top-left corner."""
    box = bbox.to_top_left_origin(page_height=height)
    return fraction_box((box.l, box.t, box.r, box.b), width, height)


def line_records(cells: Iterable[Any], width: float, height: float) -> list[dict[str, Any]]:
    """The text layer's lines (Docling's `textline_cells`), in the parser's order, without empty ones."""
    lines = []
    for cell in cells:
        text = cell.text
        if not isinstance(text, str) or not text.strip():
            continue
        lines.append({"text": text, "box": top_left_box(cell.rect.to_bounding_box(), width, height)})
    return lines


def block_record(item: Any, provenance: Any, width: float, height: float) -> dict[str, Any]:
    """One item of a page, in reading order: its label, text, box, heading level and table rows."""
    label = str(getattr(item.label, "value", item.label))
    text = getattr(item, "text", None)
    text = text if isinstance(text, str) else ""
    if len(item.prov) > 1 and provenance.charspan is not None:
        start, end = provenance.charspan
        text = text[start:end]
    heading: int | None = None
    if label == "title":
        heading = 0
    elif label == "section_header":
        heading = int(getattr(item, "level", 1) or 1)
    rows: list[list[str]] | None = None
    data = getattr(item, "data", None)
    if label == "table" and data is not None:
        rows = [[cell.text for cell in row] for row in data.grid]
    return {
        "label": label,
        "text": text,
        "box": top_left_box(provenance.bbox, width, height),
        "headingLevel": heading,
        "rows": rows,
    }


def page_records(result: Any, offset: int) -> list[dict[str, Any]]:
    """Every page of a Docling result, numbered as in the version (the batch's page plus `offset` - 1)."""
    blocks: dict[int, list[dict[str, Any]]] = {}
    sizes = {page.page_no: page.size for page in result.pages}
    for item, _level in result.document.iterate_items():
        for provenance in getattr(item, "prov", None) or []:
            size = sizes.get(provenance.page_no)
            if size is None:
                continue
            blocks.setdefault(provenance.page_no, []).append(block_record(item, provenance, size.width, size.height))
    records = []
    for page in result.pages:
        if page.size is None:
            continue
        records.append(
            {
                "page": page.page_no + offset - 1,
                "lines": line_records(page.cells, page.size.width, page.size.height),
                "blocks": blocks.get(page.page_no, []),
            }
        )
    return records


def convert(source: Path, first: int, last: int, artifacts: Path, threads: int) -> Any:  # pragma: no cover - image
    """Docling over pages `first` to `last` of the batch (numbered in the batch), offline. Runs only in the image."""
    # Imported here: the parent imports this module for its names only, and torch's threads are set first.
    import torch  # noqa: PLC0415

    torch.set_num_threads(threads)
    from docling.datamodel.accelerator_options import AcceleratorOptions  # noqa: PLC0415
    from docling.datamodel.base_models import InputFormat  # noqa: PLC0415
    from docling.datamodel.pipeline_options import HeadingHierarchyOptions, PdfPipelineOptions  # noqa: PLC0415
    from docling.document_converter import DocumentConverter, PdfFormatOption  # noqa: PLC0415

    options = PdfPipelineOptions(
        artifacts_path=artifacts,
        do_ocr=False,
        do_table_structure=True,
        generate_parsed_pages=True,
        # Heading levels from the bookmarks, numbering and font style: off by default, every heading is level 1.
        heading_hierarchy_options=HeadingHierarchyOptions(enabled=True),
        accelerator_options=AcceleratorOptions(num_threads=threads, device="cpu"),
    )
    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
    return converter.convert(source, page_range=(first, last))


def run(
    source: Path,
    out: Path,
    *,
    offset: int,
    first: int,
    last: int,
    artifacts: Path,
    threads: int,
    converter: Callable[[Path, int, int, Path, int], Any] = convert,
) -> int:
    if offset < 1 or first < offset or last < first or threads < 1:
        return 2
    result = converter(source, first - offset + 1, last - offset + 1, artifacts, threads)
    out.mkdir(exist_ok=True)
    written = []
    for record in page_records(result, offset):
        if first <= record["page"] <= last:
            (out / page_file(record["page"])).write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
            written.append(record["page"])
    del result
    pictures = write_pictures(source, out, offset, written)
    (out / MANIFEST).write_text(json.dumps({"pages": written, "pictures": pictures}), encoding="utf-8")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) != 8:
        return 2
    try:
        offset, first, last, threads = int(argv[3]), int(argv[4]), int(argv[5]), int(argv[7])
    except ValueError:
        return 2
    return run(
        Path(argv[1]), Path(argv[2]), offset=offset, first=first, last=last, artifacts=Path(argv[6]), threads=threads
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv))
