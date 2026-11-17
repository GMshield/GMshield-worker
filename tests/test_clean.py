"""The clean stage's child (Story 1.15): every PDF row of the matrix, run in-process for detail and through the
sandbox for the real thing, the batches, and the peak memory of a 1 GiB file."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pikepdf

from gmshield_worker.ingestion import clean, sandbox
from tests import pdfs

REMOVED_NAMES = (b"/JS", b"/JavaScript", b"/EmbeddedFile", b"/OpenAction", b"/AA", b"/Launch", b"/XFA", b"/RichMedia")


def run_clean(source: Path, out: Path) -> int:
    return clean.run(source, out)


def manifest(out: Path) -> dict[str, object]:
    loaded: dict[str, object] = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    return loaded


def links(page: pikepdf.Page) -> list[str]:
    """The kinds of the link actions left on a page (`GoTo`, `URI`)."""
    found = []
    for annotation in page.obj.get("/Annots", pikepdf.Array()):
        action = annotation.get("/A")
        if isinstance(action, pikepdf.Dictionary) and "/S" in action:
            found.append(str(action.S)[1:])
    return found


def test_an_active_pdf_keeps_none_of_its_active_content_in_the_original_or_the_batches(tmp_path: Path) -> None:
    source = pdfs.active(tmp_path / "active.pdf")
    before = pdfs.everything(source)
    for marker in (pdfs.JS_MARKER.encode(), pdfs.EMBEDDED_MARKER, pdfs.XFA_MARKER, pdfs.RICH_MEDIA_MARKER):
        assert marker in before  # the fixture really holds them

    assert run_clean(source, tmp_path / "out") == 0

    for cleaned in (tmp_path / "out" / "original.pdf", tmp_path / "out" / "batch-00001.pdf"):
        content = pdfs.everything(cleaned)
        for marker in (pdfs.JS_MARKER.encode(), pdfs.EMBEDDED_MARKER, pdfs.XFA_MARKER, pdfs.RICH_MEDIA_MARKER,
                       pdfs.LAUNCH_MARKER.encode(), b"javascript:"):  # fmt: skip
            assert marker not in content, (cleaned.name, marker)
        for name in REMOVED_NAMES:
            assert name not in content, (cleaned.name, name)
        raw = cleaned.read_bytes()
        assert pdfs.EMBEDDED_MARKER not in raw
        with pikepdf.Pdf.open(cleaned) as pdf:
            assert not pdf.is_encrypted
            assert len(pdf.attachments) == 0
            assert len(pdf.pages) == 3
            assert "/OpenAction" not in pdf.Root
            # The GoTo and the plain URI links stay; the chained URI keeps its link but not what came next.
            assert sorted(links(pdf.pages[0])) == ["GoTo", "URI", "URI"]
            subtypes = {str(annotation.Subtype) for annotation in pdf.pages[0].obj.Annots}
            assert subtypes <= {"/Link", "/Widget"}
    assert manifest(tmp_path / "out") == {
        "pageTotal": 3,
        "original": "original.pdf",
        "batches": [{"file": "batch-00001.pdf", "firstPage": 1, "lastPage": 3}],
    }


def test_an_owner_password_pdf_opens_without_one_and_is_saved_unencrypted(tmp_path: Path) -> None:
    source = pdfs.encrypted(tmp_path / "owner.pdf", user="")
    with pikepdf.Pdf.open(source) as pdf:
        assert pdf.is_encrypted

    assert run_clean(source, tmp_path / "out") == 0

    for cleaned in ("original.pdf", "batch-00001.pdf"):
        with pikepdf.Pdf.open(tmp_path / "out" / cleaned) as pdf:
            assert not pdf.is_encrypted
            assert b"PAGE-0002" in pdf.pages[1].Contents.read_bytes()


def test_a_user_password_pdf_needs_a_password(tmp_path: Path) -> None:
    source = pdfs.encrypted(tmp_path / "user.pdf", user="user-secret")
    assert run_clean(source, tmp_path / "out") == clean.EXIT_PASSWORD
    assert not (tmp_path / "out").exists()


def test_a_file_that_is_not_a_pdf_inside_is_unreadable(tmp_path: Path) -> None:
    assert run_clean(pdfs.broken(tmp_path / "broken.pdf"), tmp_path / "out") == clean.EXIT_UNREADABLE
    empty = tmp_path / "empty.pdf"
    pikepdf.Pdf.new().save(empty)
    assert run_clean(empty, tmp_path / "out2") == clean.EXIT_UNREADABLE


def test_pages_are_split_into_batches_of_twenty_that_hold_only_their_own_pages(tmp_path: Path) -> None:
    source = pdfs.linked(tmp_path / "linked.pdf", 45)

    assert run_clean(source, tmp_path / "out") == 0

    assert manifest(tmp_path / "out")["batches"] == [
        {"file": "batch-00001.pdf", "firstPage": 1, "lastPage": 20},
        {"file": "batch-00021.pdf", "firstPage": 21, "lastPage": 40},
        {"file": "batch-00041.pdf", "firstPage": 41, "lastPage": 45},
    ]
    for name, first, last in (("batch-00001.pdf", 1, 20), ("batch-00021.pdf", 21, 40), ("batch-00041.pdf", 41, 45)):
        content = pdfs.everything(tmp_path / "out" / name)
        with pikepdf.Pdf.open(tmp_path / "out" / name) as pdf:
            assert len(pdf.pages) == last - first + 1
        for number in range(1, 46):
            assert (f"PAGE-{number:04d}".encode() in content) == (first <= number <= last), (name, number)
    with pikepdf.Pdf.open(tmp_path / "out" / "original.pdf") as original:
        assert len(original.pages) == 45
        assert links(original.pages[0]) == ["GoTo"]


def test_the_child_cleans_in_the_sandbox_and_writes_only_under_its_folder(tmp_path: Path) -> None:
    folder = tmp_path / "step"
    folder.mkdir()
    pdfs.active(folder / "source.pdf")

    result = sandbox.run_child(Path(clean.__file__), ["source.pdf", "out"], folder=folder, cancel=threading.Event())

    assert result.returncode == 0
    assert sorted(path.name for path in folder.iterdir()) == ["out", "source.pdf"]
    assert sorted(path.name for path in (folder / "out").iterdir()) == [
        "batch-00001.pdf",
        "manifest.json",
        "original.pdf",
    ]
    for status, source in ((clean.EXIT_PASSWORD, pdfs.encrypted(folder / "user.pdf", user="u")),
                           (clean.EXIT_UNREADABLE, pdfs.broken(folder / "broken.pdf"))):  # fmt: skip
        assert sandbox.run_child(Path(clean.__file__), [source.name, "x"], folder=folder,
                                 cancel=threading.Event()).returncode == status  # fmt: skip


def test_the_child_takes_no_other_arguments(tmp_path: Path) -> None:
    assert clean.main(["clean.py"]) == 2


def test_a_1_gib_pdf_is_cleaned_with_a_peak_rss_under_512_mib(tmp_path: Path) -> None:
    folder = tmp_path / "step"
    folder.mkdir()
    # 64 pages of 16 MiB images: 1 GiB that the child never holds whole.
    pdfs.large(folder / "source.pdf", pages_count=64, image_bytes=16 * 1024 * 1024)
    assert (folder / "source.pdf").stat().st_size > 1024**3

    result = sandbox.run_child(Path(clean.__file__), ["source.pdf", "out"], folder=folder, cancel=threading.Event())

    assert result.returncode == 0
    assert manifest(folder / "out")["pageTotal"] == 64
    assert (folder / "out" / "original.pdf").stat().st_size > 1024**3
    print(f"1 GiB clean: {result.seconds:.1f} s, peak RSS {result.max_rss_kib / 1024:.0f} MiB")
    assert result.max_rss_kib < 512 * 1024
