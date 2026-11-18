"""The OCR child (Story 1.17) with Docling replaced by a fake: Docling and Tesseract run only in the production image
(`tests/image`). Each page is rendered at 216 DPI, straightened when it lies at least half a degree off (within 20),
handed over as a picture whose size Docling reads as the page's, and written in layout's shape, numbered as in the
version; a page whose OCR yields fewer than 16 characters other than whitespace is listed unreadable and not written.
Every page read, the blank one too, keeps its straightened picture as WebP with a longer side of 2,000 pixels (Story
1.19), so its boxes lie over it."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pikepdf
import pypdfium2 as pdfium  # type: ignore[import-untyped]
import pytest
from PIL import Image

from gmshield_worker.ingestion import layout, ocr
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
    manifest = json.loads((out / ocr.MANIFEST).read_text(encoding="utf-8"))
    assert {key: manifest[key] for key in ("pages", "unreadable")} == {"pages": [11, 12], "unreadable": [13]}
    assert sorted(path.name for path in out.iterdir()) == ["manifest.json", "page-0000011.json", "page-0000011.webp",
                                                          "page-0000012.json", "page-0000012.webp",
                                                          "page-0000013.webp"]  # fmt: skip
    # Matrix rows "Scanned page" and "Blank scan" (Story 1.19): the straightened picture Docling read, shrunk to a
    # longer side of 2,000 pixels with its shape kept, so the boxes (fractions of it) lie over its lines.
    sizes = {item["page"]: (item["width"], item["height"]) for item in manifest["pictures"]}
    assert sorted(sizes) == [11, 12, 13]
    assert sizes[11] == (1545, 2000)
    for number, (size, _dpi) in zip((11, 12, 13), seen, strict=True):
        assert max(sizes[number]) == layout.PICTURE_SIDE
        assert sizes[number][0] / sizes[number][1] == pytest.approx(size[0] / size[1], abs=0.001)
        with Image.open(out / layout.picture_file(number)) as written:
            assert written.format == "WEBP"
            assert written.size == sizes[number]
    # The crooked page's picture is the larger straightened canvas: wider for its height than a straight page's.
    assert sizes[12][0] > sizes[11][0]
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
        nothing, size = ocr.read_page(document, 0, 1, tmp_path / ocr.PICTURE, lambda _picture: SimpleNamespace(
            document=SimpleNamespace(iterate_items=list), pages=[]))  # fmt: skip
    finally:
        document.close()
    assert nothing is None
    assert size is None
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
        record, size = ocr.read_page(
            document, 0, 4, tmp_path / ocr.PICTURE, read, orient=lambda _picture: 180, view=tmp_path / "view.webp"
        )
        assert ocr.read_page(document, 0, 4, tmp_path / ocr.PICTURE, read, orient=lambda _picture: 0)[0] is not None
    finally:
        document.close()
    assert record is not None and record["page"] == 4
    # Upright, the first line is near the top; left upside down, the first ink is in the lower half.
    assert tops[0] < 0.2
    assert tops[1] > 0.5
    # The page view's picture is the upright one: its first ink near the top too.
    with Image.open(tmp_path / "view.webp") as view:
        assert view.size == size
        ink = np.nonzero(np.array(view.convert("L")).min(axis=1) < 128)[0]
        assert ink[0] / view.height < 0.2
    # Outside the image there is no Tesseract: no turn.
    monkeypatch.setenv("PATH", str(tmp_path))
    assert ocr.orientation(tmp_path / "missing.png") == 0


TSV_HEADER = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"


def tsv(*words: tuple[float, str]) -> str:
    rows = [f"5\t1\t1\t1\t1\t{index}\t0\t0\t10\t10\t{confidence}\t{text}" for index, (confidence, text) in
            enumerate(words, 1)]  # fmt: skip
    return "\n".join([TSV_HEADER, "1\t1\t0\t0\t0\t0\t0\t0\t100\t100\t-1\t", *rows, "broken row"]) + "\n"


def test_a_word_counts_when_tesseract_read_it_with_confidence_and_it_has_two_letters_or_digits() -> None:
    report = tsv((96.5, "Universidade"), (60, "de"), (59.9, "Coimbra"), (91, "é"), (88, "—,"), (75, "12"), (80, "x"))
    assert ocr.confident_words(report) == 3
    assert ocr.confident_words("") == 0
    assert ocr.confident_words(TSV_HEADER + "\n5\t1\t1\t1\t1\t1\t0\t0\t1\t1\tnot-a-number\tpalavra\n") == 0


def test_the_band_is_the_middle_of_the_turned_and_straightened_picture() -> None:
    picture = np.full((200, 100), 255, dtype=np.uint8)
    picture[70, 10] = 100
    picture[129, 89] = 50
    straight = ocr.band(picture, 0)
    assert straight.shape == (60, 80)
    assert straight[0, 0] == 100
    # Turned a quarter, the band runs across the turned picture (100 high, 200 wide).
    assert ocr.band(picture, 90).shape == (30, 160)
    assert ocr.band(picture, 180)[0, 0] == 50
    # A crooked page is straightened before its band is cut.
    crooked = np.array(pdfs.raster_page(angle=7).convert("L"))
    assert ocr.skew_angle(ocr.band(crooked, 0)) == pytest.approx(0.0, abs=0.5)


def test_the_detected_turn_is_kept_unless_half_a_turn_further_reads_better(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    picture = tmp_path / ocr.PICTURE
    Image.new("L", (40, 60), 255).save(picture)
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/tesseract")
    detected = {"value": 180}
    words: dict[int, int] = {}
    monkeypatch.setattr(ocr, "detect", lambda _command, _picture: detected["value"])
    monkeypatch.setattr(ocr, "readable_words", lambda _command, _pixels, degrees, _target: words[degrees])

    # The 1892 book: detection says half a turn, but the page reads only as it lies.
    words.update({0: 120, 180: 3})
    assert ocr.orientation(picture) == 0
    # Detection right: kept.
    words.update({0: 2, 180: 140})
    assert ocr.orientation(picture) == 180
    # A tie (a blank page) keeps the detection, and so does a lead under `MIN_LEAD`: text read upside down still
    # yields a few words (a crooked synthetic scan: 6 against 0 before straightening).
    words.update({0: 0, 180: 0})
    assert ocr.orientation(picture) == 180
    words.update({0: ocr.MIN_LEAD - 1, 180: 0})
    assert ocr.orientation(picture) == 180
    words.update({0: ocr.MIN_LEAD, 180: 0})
    assert ocr.orientation(picture) == 0
    words.update({0: 0, 180: 0})
    # A quarter turn is checked against the other quarter.
    detected["value"] = 90
    words.update({90: 1, 270: 95})
    assert ocr.orientation(picture) == 270
    # A picture Pillow cannot open keeps the detection.
    picture.write_bytes(b"not a picture")
    assert ocr.orientation(picture) == 90


def test_reading_the_band_runs_tesseract_on_a_band_file_it_then_removes(tmp_path: Path) -> None:
    fake = tmp_path / "tesseract"
    calls = tmp_path / "calls.txt"
    fake.write_text(f'#!/bin/sh\necho "$@" >> {calls}\ncat <<EOF\n{tsv((95, "historia"), (90, "moderna"))}EOF\n')
    fake.chmod(0o755)
    picture = np.full((300, 200), 255, dtype=np.uint8)
    target = tmp_path / ocr.BAND
    assert ocr.readable_words(str(fake), picture, 0, target) == 2
    assert not target.exists()
    arguments = calls.read_text().split()
    assert arguments[0] == str(target) and arguments[1:] == ["stdout", "-l", "por+eng", "--psm", "3", "tsv"]
    # Tesseract failing reads no word, and leaves no band behind.
    fake.write_text("#!/bin/sh\nexit 1\n")
    assert ocr.readable_words(str(fake), picture, 180, target) == 0
    assert not target.exists()
    assert ocr.readable_words(str(tmp_path / "missing"), picture, 0, target) == 0


def test_a_huge_page_is_rendered_with_its_longer_side_capped_and_written_at_216_dpi(tmp_path: Path) -> None:
    # 14,400 x 7,200 points: at scale 3 that would be 43,200 x 21,600 pixels.
    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(14_400, 7_200))
    pdf.save(tmp_path / "huge.pdf")
    document = pdfium.PdfDocument(tmp_path / "huge.pdf")
    try:
        picture = ocr.render(document, 0)
    finally:
        document.close()
    assert picture.shape[:2] == (5000, ocr.MAX_SIDE)
    ocr.write_picture(picture, tmp_path / "huge.png")
    with Image.open(tmp_path / "huge.png") as written:
        assert written.info["dpi"] == pytest.approx((ocr.DPI, ocr.DPI), abs=0.01)


def test_the_picture_scale_is_the_layout_stages_and_never_above_the_ocr_scale() -> None:
    assert layout.MAX_PICTURE_SCALE == ocr.SCALE
