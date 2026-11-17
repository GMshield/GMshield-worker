"""The OCR child (Story 1.17) with Docling replaced by a fake: Docling and Tesseract run only in the production image
(`tests/image`). Each page is rendered at 216 DPI, straightened when it lies at least half a degree off (within 20),
handed over as a picture whose size Docling reads as the page's, and written in layout's shape, numbered as in the
version; a page whose OCR yields fewer than 16 characters other than whitespace is listed unreadable and not written."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pypdfium2 as pdfium  # type: ignore[import-untyped]
import pytest
from PIL import Image

from gmshield_worker.ingestion import ocr
from tests import pdfs
from tests.test_layout import bottom_left, cell, item, page, provenance, result


def gray(picture: Image.Image) -> Any:
    return np.array(picture.convert("L"))


@pytest.mark.parametrize(("angle", "expected"), [(0, 0.0), (7, -7.0), (-7, 7.0), (3.5, -3.5), (-18, 18.0)])
def test_the_skew_estimate_is_the_rotation_that_straightens_the_lines(angle: float, expected: float) -> None:
    assert ocr.skew_angle(gray(pdfs.raster_page(angle=angle))) == pytest.approx(expected, abs=0.25)


def test_an_upside_down_or_blank_page_has_no_skew_to_fix() -> None:
    assert ocr.skew_angle(gray(pdfs.raster_page(angle=180))) == pytest.approx(0.0, abs=0.25)
    assert ocr.skew_angle(gray(pdfs.raster_page(blank=True))) == 0.0


def test_a_crooked_page_is_turned_on_a_larger_white_canvas_and_a_nearly_straight_one_is_left_alone() -> None:
    crooked = np.array(pdfs.raster_page(angle=7).convert("RGB"))
    turned, angle = ocr.straighten(crooked)
    assert angle == pytest.approx(-7.0, abs=0.25)
    assert turned.shape[0] > crooked.shape[0] and turned.shape[1] > crooked.shape[1]
    assert tuple(turned[0, 0]) == (255, 255, 255) and tuple(turned[-1, -1]) == (255, 255, 255)
    # Straightened, the page has no skew left.
    assert ocr.skew_angle(cv2.cvtColor(turned, cv2.COLOR_RGB2GRAY)) == pytest.approx(0.0, abs=0.25)

    nearly = np.array(pdfs.raster_page(angle=0.3).convert("RGB"))
    same, small = ocr.straighten(nearly)
    assert same is nearly and abs(small) < ocr.MIN_SKEW
    gray_page = gray(pdfs.raster_page(angle=-5))
    assert ocr.straighten(gray_page)[0].ndim == 2


def test_characters_counts_what_ocr_read_other_than_whitespace() -> None:
    def record(*texts: str) -> dict[str, Any]:
        return {"lines": [{"text": text} for text in texts]}

    assert ocr.characters(record()) == 0
    assert ocr.characters(record("a b c", " \n ", "d e f g h i j k l m n o")) == 15
    assert ocr.characters(record("abcdefgh", "ijklmnop", "more")) == 16


def fake_reader(seen: list[tuple[tuple[int, int], tuple[float, float]]]) -> Any:
    """A Docling stand-in: it reads the picture it is given, notes its size and resolution, and finds two words on a
    page with ink and none on a blank one."""

    def make(_artifacts: Path, threads: int) -> Any:
        assert threads == 2

        def read(picture: Path) -> Any:
            with Image.open(picture) as opened:
                seen.append((opened.size, opened.info["dpi"]))
                inked = np.array(opened.convert("L")).min() < 128
            cells = [cell("Capitulo Um:", 60, 40, 300, 80), cell("As Regras", 60, 90, 300, 120)] if inked else []
            items = [item("section_header", "Capitulo Um: As Regras", provenance(1, bottom_left(60, 40, 300, 120)),
                          level=1)] if inked else []  # fmt: skip
            return result(items, [page(1, cells)])

        return read

    return make


def test_the_child_straightens_reads_and_writes_each_page_and_lists_a_blank_one_unreadable(tmp_path: Path) -> None:
    # A batch that starts at page 11 of its version: 12 is crooked, 13 blank.
    pictures = [pdfs.raster_page(), pdfs.raster_page(angle=7), pdfs.raster_page(blank=True)]
    pdfs.scanned(tmp_path / "source.pdf", pictures)
    seen: list[tuple[tuple[int, int], tuple[float, float]]] = []
    out = tmp_path / "out"

    status = ocr.run(
        tmp_path / "source.pdf",
        out,
        offset=11,
        first=11,
        last=13,
        artifacts=tmp_path,
        threads=2,
        make_reader=fake_reader(seen),
    )

    assert status == 0
    assert json.loads((out / ocr.MANIFEST).read_text(encoding="utf-8")) == {"pages": [11, 12], "unreadable": [13]}
    assert sorted(path.name for path in out.iterdir()) == ["manifest.json", "page-0000011.json", "page-0000012.json"]
    # Rendered at 216 DPI (a US Letter page: 1836 x 2376 pixels); the crooked page on a larger canvas.
    assert seen[0][0] == (1836, 2376)
    assert seen[0][1] == pytest.approx((216.0, 216.0), abs=0.01)
    assert seen[1][0][0] > 1836 and seen[1][0][1] > 2376
    record = json.loads((out / "page-0000012.json").read_text(encoding="utf-8"))
    assert record["page"] == 12
    assert set(record) == {"page", "lines", "blocks"}
    assert record["lines"][0] == {"text": "Capitulo Um:", "box": [0.1, 0.05, 0.5, 0.1]}
    assert [(block["label"], block["headingLevel"]) for block in record["blocks"]] == [("section_header", 1)]
    assert record["blocks"][0]["box"] == [0.1, 0.05, 0.5, 0.15]


def test_the_child_refuses_bad_arguments_or_pages_outside_its_batch(tmp_path: Path) -> None:
    pdfs.scanned(tmp_path / "source.pdf", [pdfs.raster_page()])
    source, out = tmp_path / "source.pdf", tmp_path / "out"
    assert ocr.main(["ocr.py"]) == 2
    assert ocr.main(["ocr.py", str(source), str(out), "1", "x", "1", "/models", "2"]) == 2
    assert ocr.run(source, out, offset=5, first=4, last=6, artifacts=tmp_path, threads=2) == 2
    assert ocr.run(source, out, offset=1, first=3, last=2, artifacts=tmp_path, threads=2) == 2
    assert ocr.run(source, out, offset=1, first=1, last=1, artifacts=tmp_path, threads=0) == 2
    assert ocr.run(source, out, offset=1, first=1, last=2, artifacts=tmp_path, threads=2) == 3
    assert not out.exists()


def test_a_reader_that_finds_no_page_leaves_the_page_unreadable(tmp_path: Path) -> None:
    pdfs.scanned(tmp_path / "source.pdf", [pdfs.raster_page()])
    document = pdfium.PdfDocument(tmp_path / "source.pdf")
    try:
        nothing = ocr.read_page(document, 0, 1, tmp_path / ocr.PICTURE, lambda _picture: SimpleNamespace(
            document=SimpleNamespace(iterate_items=list), pages=[]))  # fmt: skip
    finally:
        document.close()
    assert nothing is None
    assert not (tmp_path / ocr.PICTURE).exists()


def test_tesseracts_orientation_report_gives_the_quarter_turns_that_set_the_page_upright() -> None:
    report = "Page number: 0\nOrientation in degrees: 180\nRotate: 180\nOrientation confidence: 9.1\nScript: Latin\n"
    assert ocr.parse_orientation(report) == 180
    assert ocr.parse_orientation(report.replace("180", "90", 1)) == 90
    assert ocr.parse_orientation("Too few characters. Skipping this page\n") == 0
    assert ocr.parse_orientation("Orientation in degrees: 45\n") == 0
    picture = np.arange(6, dtype=np.uint8).reshape(2, 3)
    assert ocr.upright(picture, 0) is picture
    assert ocr.upright(picture, 180).tolist() == [[5, 4, 3], [2, 1, 0]]
    assert ocr.upright(picture, 90).shape == (3, 2)


def test_an_upside_down_page_reaches_docling_upright(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdfs.scanned(tmp_path / "source.pdf", [pdfs.raster_page(angle=180)])
    tops: list[float] = []

    def read(picture: Path) -> Any:
        with Image.open(picture) as opened:
            ink = np.nonzero(np.array(opened.convert("L")).min(axis=1) < 128)[0]
            tops.append(ink[0] / opened.height)
        return result([], [page(1, [cell("Capitulo Um: As Regras do Jogo", 60, 40, 300, 80)])])

    document = pdfium.PdfDocument(tmp_path / "source.pdf")
    try:
        record = ocr.read_page(document, 0, 4, tmp_path / ocr.PICTURE, read, orient=lambda _picture: 180)
        assert ocr.read_page(document, 0, 4, tmp_path / ocr.PICTURE, read, orient=lambda _picture: 0) is not None
    finally:
        document.close()
    assert record is not None and record["page"] == 4
    # Upright, the first line is near the top; left upside down, the first ink is in the lower half.
    assert tops[0] < 0.2
    assert tops[1] > 0.5
    # Outside the image there is no Tesseract: no turn.
    monkeypatch.setenv("PATH", str(tmp_path))
    assert ocr.orientation(tmp_path / "missing.png") == 0
