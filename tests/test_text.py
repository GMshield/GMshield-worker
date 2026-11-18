"""The text child (Story 1.16): a TXT or Markdown upload decoded as it streams (UTF-8 without its BOM, else cp1252)
and cut into pages of at most 3,000 code points at Markdown headings, then blank lines, then line ends, then
whitespace; each page carries its heading path and its code-point range, and the pages cover the text whole."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from gmshield_worker.ingestion import sandbox, text


def run(
    tmp_path: Path, content: bytes, format_name: str = "markdown"
) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
    """The child's work in this process (so coverage sees it); `test_the_child_runs_in_its_sandbox` runs it there."""
    (tmp_path / "source.txt").write_bytes(content)
    out = tmp_path / "out"
    status = text.main(["text.py", str(tmp_path / "source.txt"), str(out), format_name])
    if status != 0:
        return status, {}, []
    manifest = json.loads((out / text.MANIFEST).read_text(encoding="utf-8"))
    pages = [
        page
        for item in manifest["texts"]
        for page in json.loads((out / item["file"]).read_text(encoding="utf-8"))["pages"]
    ]
    return 0, manifest, pages


def whole(pages: list[dict[str, Any]]) -> str:
    """The pages' texts end to end, after checking their ranges follow one another from 0."""
    offset = 0
    for number, page in enumerate(pages, start=1):
        assert page["page"] == number
        assert page["start"] == offset
        assert page["end"] == offset + len(page["text"])
        assert len(page["text"]) <= text.PAGE_LIMIT
        offset = page["end"]
    return "".join(page["text"] for page in pages)


def test_markdown_sections_and_a_long_paragraph_become_bounded_pages_under_their_heading_paths(
    tmp_path: Path,
) -> None:
    paragraph = " ".join(["palavra"] * 875)  # 6,999 code points, one line
    document = f"# A\n\nUnder A.\n\n## B\n\n{paragraph}\n\n# C\nLast.\n"

    status, manifest, pages = run(tmp_path, document.encode("utf-8"))

    assert status == 0
    assert whole(pages) == document
    assert manifest["pageTotal"] == len(pages)
    assert [page["headingPath"] for page in pages] == [["A"], ["A", "B"], ["A", "B"], ["A", "B"], ["A", "B"], ["C"]]
    assert [len(page["text"]) for page in pages[2:5]] == [3000, 3000, 999 + 2]
    # The long paragraph is cut after whitespace, never inside a word.
    assert pages[2]["text"].endswith(" ") and pages[3]["text"].startswith("palavra")
    assert pages[1]["text"] == "## B\n\n"


def test_a_section_is_cut_at_blank_lines_before_line_ends_before_whitespace() -> None:
    blank = "a" * 2000 + "\n\n" + "b" * 500 + "\n" + "c" * 1000
    assert text.cut(blank) == 2002
    lines = "a" * 2000 + "\n" + "b" * 500 + " " + "c" * 1000
    assert text.cut(lines) == 2001
    words = "a" * 2000 + " " + "b" * 2000
    assert text.cut(words) == 2001
    solid = "a" * 4000
    assert text.cut(solid) == 3000


def test_headings_in_fenced_code_and_hashes_without_a_space_start_no_page(tmp_path: Path) -> None:
    document = "# Real\n```\n# not a heading\n```\n~~~\n## nor this\n~~~\n#hashtag\n  ## Indented ##\nEnd\n"

    status, _manifest, pages = run(tmp_path, document.encode("utf-8"))

    assert status == 0
    assert whole(pages) == document
    assert [page["headingPath"] for page in pages] == [["Real"], ["Real", "Indented"]]


def test_a_deeper_heading_then_a_shallower_one_resets_the_path(tmp_path: Path) -> None:
    document = "# A\n### C\n## B\n#### D\n# E\n"
    status, _manifest, pages = run(tmp_path, document.encode("utf-8"))
    assert status == 0
    assert [page["headingPath"] for page in pages] == [["A"], ["A", "C"], ["A", "B"], ["A", "B", "D"], ["E"]]


def test_plain_text_has_no_headings_and_its_byte_order_mark_goes(tmp_path: Path) -> None:
    document = "# Not a heading in plain text\n" + "linha de texto\n" * 400

    status, manifest, pages = run(tmp_path, b"\xef\xbb\xbf" + document.encode("utf-8"), "txt")

    assert status == 0
    assert whole(pages) == document
    assert all(page["headingPath"] == [] for page in pages)
    assert manifest["pageTotal"] == 3  # 6,030 code points, cut at line ends


def test_cp1252_bytes_that_are_not_utf8_are_decoded_as_cp1252(tmp_path: Path) -> None:
    document = "Ação, café e pão \u201centre aspas\u201d \u2013 20 \u20ac\n" * 10

    status, _manifest, pages = run(tmp_path, document.encode("cp1252"), "txt")

    assert status == 0
    assert whole(pages) == document


def test_bytes_cp1252_leaves_undefined_become_replacement_characters(tmp_path: Path) -> None:
    status, _manifest, pages = run(tmp_path, b"caf\xe9 \x81 fim\n", "txt")
    assert status == 0
    assert whole(pages) == "café � fim\n"


def test_every_500_pages_go_to_a_file_of_their_own(tmp_path: Path) -> None:
    document = ("x" * 2999 + "\n") * 1001

    status, manifest, pages = run(tmp_path, document.encode("ascii"), "txt")

    assert status == 0
    assert manifest == {
        "pageTotal": 1001,
        "texts": [
            {"file": "text-00001.json", "firstPage": 1, "lastPage": 500},
            {"file": "text-00002.json", "firstPage": 501, "lastPage": 1000},
            {"file": "text-00003.json", "firstPage": 1001, "lastPage": 1001},
        ],
    }
    assert whole(pages) == document


def test_a_line_longer_than_a_page_is_streamed_in_pieces(tmp_path: Path) -> None:
    pieces = list(text.lines(iter(["a" * 2500, "a" * 2500 + "\n# x\n"]), limit=3000))
    assert pieces == [("a" * 3000, True), ("a" * 2000 + "\n", False), ("# x\n", True)]
    document = "word " * 20000 + "\n# Heading\nbody\n"
    status, _manifest, pages = run(tmp_path, document.encode("ascii"))
    assert status == 0
    assert whole(pages) == document
    assert pages[-1]["headingPath"] == ["Heading"]


@pytest.mark.parametrize("content", [b"", b"\xef\xbb\xbf"])
def test_a_file_with_no_text_is_unreadable(tmp_path: Path, content: bytes) -> None:
    status, _manifest, _pages = run(tmp_path, content, "txt")
    assert status == text.EXIT_UNREADABLE


def test_an_unknown_format_or_bad_arguments_end_without_a_manifest(tmp_path: Path) -> None:
    status, _manifest, _pages = run(tmp_path, b"text\n", "pdf")
    assert status != 0
    assert text.main(["text.py"]) == 2


def test_the_child_runs_in_its_sandbox(tmp_path: Path) -> None:
    document = "# T\n" + "linha\n" * 1000
    (tmp_path / "source.txt").write_bytes(document.encode("utf-8"))

    result = sandbox.run_child(
        Path(text.__file__), ["source.txt", "out", "markdown"], folder=tmp_path, cancel=threading.Event()
    )

    assert result.returncode == 0, result
    manifest = json.loads((tmp_path / "out" / text.MANIFEST).read_text(encoding="utf-8"))
    assert manifest["pageTotal"] == 3


def test_setext_headings_start_a_page_with_their_title_under_the_new_path(tmp_path: Path) -> None:
    document = (
        "Intro.\n\nPart One\n========\n\nText one.\n\nSection A\n---------\nText A.\n\nSection B\n  ---\nText B.\n"
    )

    status, _manifest, pages = run(tmp_path, document.encode("utf-8"))

    assert status == 0
    assert whole(pages) == document
    assert [(page["headingPath"], page["text"].splitlines()[0]) for page in pages] == [
        ([], "Intro."),
        (["Part One"], "Part One"),
        (["Part One", "Section A"], "Section A"),
        (["Part One", "Section B"], "Section B"),
    ]


def test_a_dash_line_after_a_blank_line_or_in_code_is_no_heading(tmp_path: Path) -> None:
    document = "# Top\n\nText.\n\n---\n\n```\nNot a title\n---\n```\n----\n---\n    Indented\n---\nEnd\n"

    status, _manifest, pages = run(tmp_path, document.encode("utf-8"))

    assert status == 0
    assert whole(pages) == document
    assert [page["headingPath"] for page in pages] == [["Top"]]


def test_a_line_held_for_a_setext_underline_never_grows_past_a_page(tmp_path: Path) -> None:
    title = "t" * 2500
    document = "x" * 2900 + "\n" + title + "\n===\n" + "y" * 5000 + "\n"

    status, _manifest, pages = run(tmp_path, document.encode("ascii"))

    assert status == 0
    assert whole(pages) == document
    assert pages[0]["headingPath"] == []
    assert pages[1]["headingPath"] == [title]
    assert pages[1]["text"].startswith(title + "\n===\n")


def test_a_heading_drops_its_closing_hashes_only_after_whitespace() -> None:
    assert text._heading_title("Title ##  ") == "Title"
    assert text._heading_title("C# notes") == "C# notes"
    assert text._heading_title("Title#") == "Title#"
    assert text._heading_title("###") == ""
    assert text._heading_title("  ") == ""


def test_a_crafted_heading_line_is_matched_in_linear_time() -> None:
    # A marker, a long run of blanks and a line end before more text: the old pattern (lazy title, optional closing
    # run, optional blanks) took about 13 s on this line and 105 s on twice as many blanks.
    line = "# " + " " * 1500 + "\nx"
    started = time.monotonic()
    assert text._HEADING.fullmatch(line) is None
    assert time.monotonic() - started < 0.5
    heading = text._HEADING.fullmatch("# Title ##\r\n")
    assert heading is not None
    assert text._heading_title(heading[2]) == "Title"
