"""The triage child (Story 1.16) in its sandbox: a page with at least 16 characters other than whitespace in its text
layer has one; any other page (a scan, a blank page, a caption alone) is an image page. Pages are numbered as in the
version, read from a batch that starts at OFFSET."""

from __future__ import annotations

import json
import threading
from pathlib import Path

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
