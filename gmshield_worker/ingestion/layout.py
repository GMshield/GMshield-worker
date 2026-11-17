"""The layout stage's child process (Story 1.16): read the text pages of a batch with Docling.

Run only through the sandbox (`sandbox.py`), as a script in isolated mode, inside the step folder:

    python -I -B layout.py SOURCE OUT_DIR OFFSET FIRST LAST ARTIFACTS THREADS

SOURCE is a page batch whose first page is page OFFSET of the version; FIRST to LAST (inclusive, numbered as in the
version) are the pages to read. The child's environment is empty, so everything Docling needs comes on the command
line: ARTIFACTS is the folder of its layout and table models (baked into the image: nothing is downloaded) and THREADS
the number of threads torch and Docling may use. Docling runs without OCR (`do_ocr=False`), with table structure, and
infers heading levels (from the PDF's bookmarks, numbering and font style).

It imports Docling, torch and the standard library only (never `gmshield_worker`), holds no credential, reaches no
network, and writes only under OUT_DIR: one `page-NNNNNNN.json` per page read, and `manifest.json` listing them,
`{"pages": [N, ...]}`. A page record is

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
from collections.abc import Iterable
from pathlib import Path
from typing import Any

MANIFEST = "manifest.json"
DIGITS = 4


def page_file(page: int) -> str:
    return f"page-{page:07d}.json"


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


def run(source: Path, out: Path, *, offset: int, first: int, last: int, artifacts: Path, threads: int) -> int:
    if offset < 1 or first < offset or last < first or threads < 1:
        return 2
    result = convert(source, first - offset + 1, last - offset + 1, artifacts, threads)
    out.mkdir(exist_ok=True)
    written = []
    for record in page_records(result, offset):
        if first <= record["page"] <= last:
            (out / page_file(record["page"])).write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
            written.append(record["page"])
    (out / MANIFEST).write_text(json.dumps({"pages": written}), encoding="utf-8")
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
