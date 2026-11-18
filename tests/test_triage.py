"""The triage child (Stories 1.16 and 1.17) in its sandbox: a page with at least 16 characters other than whitespace in
its text layer has one, unless it has fewer than 200 and one image covers at least half of it (a scan under a digital
running header); any other page (a scan, a blank page, a caption alone) is an image page. Pages are numbered as in
the version, read from a batch that starts at OFFSET."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pikepdf
import pytest

from gmshield_worker.ingestion import sandbox, triage
from tests import pdfs

SENTENCE = "The party rests at the inn for the night"


@pytest.mark.parametrize(
    ("text", "layer"),
    [
        ("", "image"),
        ("   \n\t  ", "image"),
        ("a b c d e f g h i j k l m n o", "image"),  # 15 characters
        ("a b c d e f g h i j k l m n o p", "text"),  # 16 characters
        (SENTENCE, "text"),
    ],
)
def test_sixteen_characters_other_than_whitespace_make_a_text_layer(text: str, layer: str) -> None:
    assert triage.layer_of(text) == layer


def run(tmp_path: Path, texts: list[str], arguments: list[str]) -> tuple[sandbox.ChildResult, Path]:
    """The child in its sandbox, as the step runs it."""
    pdfs.with_text(tmp_path / "source.pdf", texts)
    result = sandbox.run_child(
        Path(triage.__file__), ["source.pdf", "out", *arguments], folder=tmp_path, cancel=threading.Event()
    )
    return result, tmp_path / "out" / triage.MANIFEST


def run_here(tmp_path: Path, texts: list[str], arguments: list[str]) -> tuple[int, Path]:
    """The child's work in this process, so coverage sees it."""
    pdfs.with_text(tmp_path / "source.pdf", texts)
    status = triage.main(["triage.py", str(tmp_path / "source.pdf"), str(tmp_path / "out"), *arguments])
    return status, tmp_path / "out" / triage.MANIFEST


def test_the_child_marks_each_page_of_a_batch_by_its_number_in_the_version(tmp_path: Path) -> None:
    # A batch holding pages 21 to 25 of the version: 22 is blank, 23 has a page number only.
    result, manifest = run(tmp_path, [SENTENCE, "", "23", SENTENCE, SENTENCE], ["21", "21", "25"])

    assert result.returncode == 0, result
    assert json.loads(manifest.read_text(encoding="utf-8")) == {
        "pages": [
            {"page": 21, "layer": "text"},
            {"page": 22, "layer": "image"},
            {"page": 23, "layer": "image"},
            {"page": 24, "layer": "text"},
            {"page": 25, "layer": "text"},
        ]
    }


def test_one_page_of_a_batch_alone(tmp_path: Path) -> None:
    status, manifest = run_here(tmp_path, [SENTENCE, "", SENTENCE], ["41", "42", "42"])

    assert status == 0
    assert json.loads(manifest.read_text(encoding="utf-8")) == {"pages": [{"page": 42, "layer": "image"}]}


@pytest.mark.parametrize("arguments", [["1", "1", "4"], ["2", "1", "1"], ["1", "2", "1"], ["1", "x", "1"], ["1"]])
def test_pages_outside_the_batch_or_bad_arguments_end_without_a_manifest(tmp_path: Path, arguments: list[str]) -> None:
    status, manifest = run_here(tmp_path, [SENTENCE, SENTENCE, SENTENCE], arguments)

    assert status != 0
    assert not manifest.exists()


def test_a_file_that_is_not_a_pdf_crashes_the_child(tmp_path: Path) -> None:
    (tmp_path / "source.pdf").write_bytes(b"not a pdf")
    result = sandbox.run_child(
        Path(triage.__file__), ["source.pdf", "out", "1", "1", "1"], folder=tmp_path, cancel=threading.Event()
    )
    assert result.returncode != 0
    assert not (tmp_path / "out").exists()


# --- Story 1.17: few characters over a big image


@pytest.mark.parametrize(
    ("text", "image_share", "layer"),
    [
        ("H" * 30, 1.0, "image"),  # a running header over a full-page scan
        ("H" * 199, 0.5, "image"),
        ("H" * 199, 0.49, "text"),  # a small illustration beside a short text
        ("H" * 200, 1.0, "text"),  # a full text layer over a scan (a scan already read by OCR) is never OCRed again
        ("H" * 15, 0.0, "image"),
    ],
)
def test_few_characters_over_half_a_page_of_image_make_an_image_page(text: str, image_share: float, layer: str) -> None:
    assert triage.layer_of(text, image_share) == layer


def test_an_image_share_is_its_bounds_clipped_to_the_page() -> None:
    page = (0.0, 0.0, 100.0, 200.0)
    assert triage.share((0, 0, 100, 200), page) == 1.0
    assert triage.share((-50, -50, 150, 250), page) == 1.0
    assert triage.share((0, 0, 50, 200), page) == 0.5
    assert triage.share((200, 0, 300, 200), page) == 0.0
    assert triage.share((0, 0, 1, 1), (0, 0, 0, 0)) == 0.0


HEADER = "Livro do Jogador - Capitulo 3"  # 30 characters, 26 other than whitespace


def test_a_scan_under_a_digital_running_header_is_an_image_page_and_a_full_text_layer_over_a_scan_is_text(
    tmp_path: Path,
) -> None:
    blank = pdfs.raster_page(blank=True)
    pdfs.scanned(tmp_path / "header.pdf", [blank], header=HEADER)
    pdfs.scanned(tmp_path / "long.pdf", [blank], header=SENTENCE * 8)  # 248 characters other than whitespace
    pdfs.scanned(tmp_path / "caption.pdf", [blank], header=HEADER, cover=0.5)  # an image over a quarter of the page
    joined = pikepdf.Pdf.new()
    for name in ("header.pdf", "long.pdf", "caption.pdf"):
        with pikepdf.Pdf.open(tmp_path / name) as part:
            joined.pages.extend(part.pages)
    joined.save(tmp_path / "source.pdf")

    result = sandbox.run_child(
        Path(triage.__file__), ["source.pdf", "out", "1", "1", "3"], folder=tmp_path, cancel=threading.Event()
    )

    assert result.returncode == 0, result
    assert json.loads((tmp_path / "out" / triage.MANIFEST).read_text(encoding="utf-8")) == {
        "pages": [{"page": 1, "layer": "image"}, {"page": 2, "layer": "text"}, {"page": 3, "layer": "text"}]
    }
    # The same in this process, so coverage sees it.
    assert triage.main(["triage.py", str(tmp_path / "source.pdf"), str(tmp_path / "here"), "1", "1", "3"]) == 0
    assert (tmp_path / "here" / triage.MANIFEST).read_text(encoding="utf-8") == (
        tmp_path / "out" / triage.MANIFEST
    ).read_text(encoding="utf-8")
