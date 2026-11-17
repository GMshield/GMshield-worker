"""The layout child's records (Story 1.16), from a Docling result built by hand: Docling itself runs only in the
production image (`tests/image`), where its models are. Boxes become fractions from the top-left corner whichever
origin Docling used; blocks keep reading order, with heading levels and table rows; pages are numbered as in the
version."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from docling_core.types.doc.base import BoundingBox, CoordOrigin

from gmshield_worker.ingestion import layout

WIDTH, HEIGHT = 600.0, 800.0


def bottom_left(left: float, top: float, right: float, bottom: float) -> BoundingBox:
    """A box given from the top-left corner, stored the way Docling's items store theirs (bottom-left origin)."""
    return BoundingBox(l=left, t=HEIGHT - top, r=right, b=HEIGHT - bottom, coord_origin=CoordOrigin.BOTTOMLEFT)


def provenance(page: int, box: BoundingBox, charspan: tuple[int, int] = (0, 0)) -> SimpleNamespace:
    return SimpleNamespace(page_no=page, bbox=box, charspan=charspan)


def item(label: str, text: str, *provenances: SimpleNamespace, **extra: Any) -> SimpleNamespace:
    return SimpleNamespace(label=SimpleNamespace(value=label), text=text, prov=list(provenances), **extra)


def cell(text: str, left: float, top: float, right: float, bottom: float) -> SimpleNamespace:
    box = BoundingBox(l=left, t=top, r=right, b=bottom, coord_origin=CoordOrigin.TOPLEFT)
    return SimpleNamespace(text=text, rect=SimpleNamespace(to_bounding_box=lambda: box))


def result(items: list[SimpleNamespace], pages: list[SimpleNamespace]) -> SimpleNamespace:
    document = SimpleNamespace(iterate_items=lambda: [(entry, 1) for entry in items])
    return SimpleNamespace(document=document, pages=pages)


def page(number: int, cells: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(page_no=number, size=SimpleNamespace(width=WIDTH, height=HEIGHT), cells=cells)


def test_a_box_is_a_fraction_of_the_page_from_its_top_left_corner_clamped_and_rounded() -> None:
    assert layout.fraction_box((60, 80, 300, 400), WIDTH, HEIGHT) == [0.1, 0.1, 0.5, 0.5]
    assert layout.fraction_box((-5, 10, 700, 900), WIDTH, HEIGHT) == [0.0, 0.0125, 1.0, 1.0]
    assert layout.fraction_box((1, 1, 2, 2), 3, 3) == [0.3333, 0.3333, 0.6667, 0.6667]
    assert layout.fraction_box((1, 1, 2, 2), 0, 0) == [0.0, 0.0, 0.0, 0.0]
    assert layout.top_left_box(bottom_left(60, 80, 300, 400), WIDTH, HEIGHT) == [0.1, 0.1, 0.5, 0.5]


def test_pages_keep_reading_order_heading_levels_table_rows_and_their_numbers_in_the_version() -> None:
    table = SimpleNamespace(grid=[[SimpleNamespace(text="Item"), SimpleNamespace(text="Cost")],
                                  [SimpleNamespace(text="Rope"), SimpleNamespace(text="1 gp")]])  # fmt: skip
    items = [
        item("title", "The Book", provenance(1, bottom_left(60, 40, 300, 80))),
        item("section_header", "Combat", provenance(1, bottom_left(60, 100, 200, 120)), level=2),
        item("text", "Left column.", provenance(1, bottom_left(60, 140, 280, 400))),
        item("text", "Right column.", provenance(1, bottom_left(320, 140, 540, 400))),
        item("table", "", provenance(2, bottom_left(60, 60, 540, 200)), data=table),
        item("picture", "", provenance(2, bottom_left(60, 300, 540, 700))),
        item("page_footer", "Ignored page", provenance(9, bottom_left(0, 0, 10, 10))),
    ]
    pages = [
        page(1, [cell("The Book", 60, 40, 300, 80), cell("  ", 0, 0, 1, 1), cell("Left column.", 60, 140, 280, 160)]),
        page(2, []),
    ]

    records = layout.page_records(result(items, pages), offset=21)

    assert [record["page"] for record in records] == [21, 22]
    first, second = records
    assert first["lines"] == [
        {"text": "The Book", "box": [0.1, 0.05, 0.5, 0.1]},
        {"text": "Left column.", "box": [0.1, 0.175, 0.4667, 0.2]},
    ]
    assert [(block["label"], block["text"], block["headingLevel"]) for block in first["blocks"]] == [
        ("title", "The Book", 0),
        ("section_header", "Combat", 2),
        ("text", "Left column.", None),
        ("text", "Right column.", None),
    ]
    assert first["blocks"][3]["box"] == [0.5333, 0.175, 0.9, 0.5]
    assert all(block["rows"] is None for block in first["blocks"])
    assert [(block["label"], block["rows"]) for block in second["blocks"]] == [
        ("table", [["Item", "Cost"], ["Rope", "1 gp"]]),
        ("picture", None),
    ]
    json.dumps(records)


def test_an_item_over_two_pages_gives_each_page_its_own_part() -> None:
    spanning = item(
        "text",
        "Ends here. Goes on.",
        provenance(1, bottom_left(60, 700, 540, 780), (0, 10)),
        provenance(2, bottom_left(60, 40, 540, 60), (11, 19)),
    )
    records = layout.page_records(result([spanning], [page(1, []), page(2, [])]), offset=1)
    assert [record["blocks"][0]["text"] for record in records] == ["Ends here.", "Goes on."]


def test_the_child_refuses_bad_arguments_without_writing(tmp_path: Path) -> None:
    assert layout.main(["layout.py"]) == 2
    assert layout.main(["layout.py", "s.pdf", str(tmp_path / "out"), "1", "x", "2", "/models", "2"]) == 2
    source, out = tmp_path / "s.pdf", tmp_path / "out"
    assert layout.run(source, out, offset=5, first=4, last=6, artifacts=tmp_path, threads=2) == 2
    assert layout.run(source, out, offset=1, first=3, last=2, artifacts=tmp_path, threads=2) == 2
    assert layout.run(source, out, offset=1, first=1, last=1, artifacts=tmp_path, threads=0) == 2
    assert not (tmp_path / "out").exists()
    assert layout.page_file(9) == "page-0000009.json"
