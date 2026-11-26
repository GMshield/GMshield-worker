"""The clean stage's child (Story 1.15): every PDF row of the matrix, run in-process for detail and through the
sandbox for the real thing, the batches, and the peak memory of a 1 GiB file."""

from __future__ import annotations

import json
import os
import subprocess
import sys
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


def test_a_uri_is_kept_only_for_http_https_or_mailto_and_an_odd_annots_entry_never_crashes(tmp_path: Path) -> None:
    source = pdfs.uris(tmp_path / "uris.pdf")

    assert run_clean(source, tmp_path / "out") == 0

    with pikepdf.Pdf.open(tmp_path / "out" / "original.pdf") as pdf:
        kept = [
            str(annotation.A.URI)
            for annotation in pdf.pages[0].obj.Annots
            if isinstance(annotation, pikepdf.Dictionary) and "/A" in annotation
        ]
        assert sorted(kept) == sorted(pdfs.SAFE_URIS)
        # The null and the integer are still there, untouched.
        assert len(pdf.pages[0].obj.Annots) == len(pdfs.UNSAFE_URIS) + len(pdfs.SAFE_URIS) + 2


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
    assert clean.main(["clean.py", "marks", "a", "b"]) == 2


def bookmarks_of(source: Path, target: Path) -> object:
    assert clean.read_bookmarks(source, target) == 0
    return json.loads(target.read_text(encoding="utf-8"))


def test_bookmarks_are_read_with_every_kind_of_destination_and_the_title_root_dropped(tmp_path: Path) -> None:
    # Matrix row "Bad bookmark": an entry whose destination does not resolve is dropped, the others kept (its child
    # takes its place); so are an entry with no title, and the document-title root, the only entry at depth 0.
    source = pdfs.outlined(tmp_path / "outlined.pdf")
    assert bookmarks_of(source, tmp_path / "bookmarks.json") == pdfs.OUTLINED_BOOKMARKS

    # Clean keeps the outline in the original (and only there): the bookmarks read from it are the same.
    assert run_clean(source, tmp_path / "out") == 0
    assert bookmarks_of(tmp_path / "out" / "original.pdf", tmp_path / "again.json") == pdfs.OUTLINED_BOOKMARKS
    with pikepdf.Pdf.open(tmp_path / "out" / "batch-00001.pdf") as batch:
        assert "/Outlines" not in batch.Root
        assert "/Names" not in batch.Root
        assert "/Dests" not in batch.Root


def test_an_outline_that_loops_is_read_once_and_a_file_without_one_has_none(tmp_path: Path) -> None:
    looped = pdfs.outlined(tmp_path / "looped.pdf", loop=True)
    assert bookmarks_of(looped, tmp_path / "looped.json") == pdfs.OUTLINED_BOOKMARKS
    plain = tmp_path / "plain.pdf"
    pdfs.pages(3).save(plain)
    assert bookmarks_of(plain, tmp_path / "plain.json") == []

    # Two entries at depth 0: neither is a document-title root.
    with pikepdf.Pdf.open(plain, allow_overwriting_input=True) as pdf:
        with pdf.open_outline() as outline:
            outline.root.extend([pikepdf.OutlineItem("One", 0), pikepdf.OutlineItem("Two", 2)])
        pdf.save(plain)
    assert bookmarks_of(plain, tmp_path / "two.json") == [
        {"title": "One", "depth": 0, "page": 1},
        {"title": "Two", "depth": 0, "page": 3},
    ]


def test_a_lone_top_entry_without_children_is_kept_and_names_resolve_either_way(tmp_path: Path) -> None:
    source = tmp_path / "lone.pdf"
    pdf = pdfs.pages(4)
    with pdf.open_outline() as outline:
        outline.root.append(pikepdf.OutlineItem("Only Chapter", 1))
    pdf.save(source)
    assert bookmarks_of(source, tmp_path / "lone.json") == [{"title": "Only Chapter", "depth": 0, "page": 2}]

    # A Name found only in the name tree, a String found only in `/Dests`; a boolean where a page index would be.
    pdf = pdfs.pages(4)
    page = [item.obj for item in pdf.pages]
    pdf.Root.Dests = pikepdf.Dictionary(InDests=pikepdf.Array([page[2], pikepdf.Name.Fit]))
    tree = pikepdf.NameTree.new(pdf)
    tree["InTree"] = pikepdf.Array([page[3], pikepdf.Name.Fit])
    pdf.Root.Names = pikepdf.Dictionary(Dests=tree.obj)
    with pdf.open_outline() as outline:
        outline.root.extend(
            [
                pikepdf.OutlineItem("By name", destination=pikepdf.Name("/InTree")),
                pikepdf.OutlineItem("By string", destination=pikepdf.String("InDests")),
                pikepdf.OutlineItem("By boolean", destination=pikepdf.Array([True, pikepdf.Name.Fit])),
            ]
        )
    pdf.save(source)
    assert bookmarks_of(source, tmp_path / "named.json") == [
        {"title": "By name", "depth": 0, "page": 4},
        {"title": "By string", "depth": 0, "page": 3},
    ]


def test_the_bookmarks_child_runs_in_the_sandbox_and_writes_only_its_file(tmp_path: Path) -> None:
    folder = tmp_path / "step"
    folder.mkdir()
    pdfs.outlined(folder / "source.pdf")

    result = sandbox.run_child(
        Path(clean.__file__), ["bookmarks", "source.pdf", "bookmarks.json"], folder=folder, cancel=threading.Event()
    )

    assert result.returncode == 0
    assert sorted(path.name for path in folder.iterdir()) == ["bookmarks.json", "source.pdf"]
    assert json.loads((folder / "bookmarks.json").read_text(encoding="utf-8")) == pdfs.OUTLINED_BOOKMARKS
    pdfs.broken(folder / "broken.pdf")
    assert clean.main(["clean.py", "bookmarks", str(folder / "broken.pdf"), str(folder / "x.json")]) == (
        clean.EXIT_UNREADABLE
    )
    assert not (folder / "x.json").exists()


def test_a_1_gib_pdf_is_cleaned_with_a_peak_rss_under_512_mib(tmp_path: Path) -> None:
    folder = tmp_path / "step"
    folder.mkdir()
    # 64 pages of 16 MiB images: 1 GiB that the child never holds whole.
    pdfs.large(folder / "source.pdf", pages_count=64, image_bytes=16 * 1024 * 1024)
    assert (folder / "source.pdf").stat().st_size > 1024**3

    # A child's peak RSS starts from its parent's resident size at the fork (Linux keeps the high-water mark across
    # exec), so the child is launched from a fresh interpreter: from pytest, the reading would grow with the whole
    # suite's memory (528 MiB in a full run on 2026-11-26, 91 MiB alone) and not measure the child.
    launch = (
        "import json, sys, threading; from pathlib import Path; from gmshield_worker.ingestion import clean, sandbox; "
        "r = sandbox.run_child(Path(clean.__file__), ['source.pdf', 'out'], folder=Path(sys.argv[1]), "
        "cancel=threading.Event()); print(json.dumps([r.returncode, r.max_rss_kib, r.seconds]))"
    )
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent)}
    ran = subprocess.run(  # noqa: S603 - this interpreter and a fixed program, never a shell or a user's value
        [sys.executable, "-c", launch, str(folder)], capture_output=True, text=True, check=True, env=env
    )
    returncode, max_rss_kib, seconds = json.loads(ran.stdout)

    assert returncode == 0
    assert manifest(folder / "out")["pageTotal"] == 64
    assert (folder / "out" / "original.pdf").stat().st_size > 1024**3
    print(f"1 GiB clean: {seconds:.1f} s, peak RSS {max_rss_kib / 1024:.0f} MiB")
    assert max_rss_kib < 512 * 1024
