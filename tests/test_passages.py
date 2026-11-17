"""The passages cutter (Story 1.18) on page records built by hand: every worker row of the I/O matrix (a path across
batches, a table that fits and one that does not, a long stat block, Markdown, a page break), the outline, the blocks
that never enter a passage, the limits the backend checks, and the records a text file must hold."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from gmshield_worker.ingestion import passages, text
from gmshield_worker.ingestion.passages import PASSAGE_LIMIT, PdfCutter, TextCutter

BOX = [0.1, 0.2, 0.9, 0.3]


def block(content: str, level: int | None = None, label: str = "text", rows: Any = None) -> dict[str, Any]:
    if level is not None:
        label = "title" if level == 0 else "section_header"
    return {"label": label, "text": content, "box": BOX, "headingLevel": level, "rows": rows}


def pdf_page(page: int, *blocks: dict[str, Any]) -> dict[str, Any]:
    return {"page": page, "lines": [], "blocks": list(blocks)}


def cut_pdf(*pages: dict[str, Any], limit: int = PASSAGE_LIMIT) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cutter = PdfCutter(limit)
    for page in pages:
        cutter.page(page)
    return cutter.finish()


def test_the_limit_is_the_text_stages_page_limit() -> None:
    assert PASSAGE_LIMIT == text.PAGE_LIMIT == 3000


def test_a_section_path_crosses_batches_and_pages() -> None:
    # Matrix row "Path across batches": the heading stack carries from one text file to the next.
    cutter = PdfCutter()
    cutter.page(pdf_page(19, block("Magic Items", 1), block("Wondrous things.")))
    cutter.page(pdf_page(20, block("More of them.")))
    # The next batch's text file.
    cutter.page(pdf_page(21, block("Vorpal Sword", 2), block("A legendary blade.")))
    found, outline = cutter.finish()

    assert [(item["page"], item["sectionPath"], item["text"]) for item in found] == [
        (19, ["Magic Items"], "Magic Items\nWondrous things."),
        (20, ["Magic Items"], "Magic Items\nMore of them."),
        (21, ["Magic Items", "Vorpal Sword"], "Vorpal Sword\nA legendary blade."),
    ]
    assert outline == [
        {"level": 1, "title": "Magic Items", "page": 19},
        {"level": 2, "title": "Vorpal Sword", "page": 21},
    ]
    assert found[0]["boxes"] == [BOX, BOX]
    assert "range" not in found[0]


def test_a_heading_closes_the_sections_at_its_level_or_deeper() -> None:
    found, outline = cut_pdf(
        pdf_page(
            1,
            block("Book", 0),
            block("Chapter 1", 1),
            block("Section 1.1", 2),
            block("Body."),
            block("Chapter 2", 1),
            block("Body two."),
            block("Book Two", 0),
            block("Body three."),
        )
    )
    assert [item["sectionPath"] for item in found if item["text"].endswith(".")] == [
        ["Book", "Chapter 1", "Section 1.1"],
        ["Book", "Chapter 2"],
        ["Book Two"],
    ]
    assert [entry["level"] for entry in outline] == [0, 1, 2, 1, 0]
    # A heading with nothing under it before the next one is a passage of its own.
    assert found[0] == {"page": 1, "sectionPath": ["Book"], "text": "Book", "boxes": [BOX]}


def test_a_page_break_starts_a_passage_on_the_same_path_that_repeats_the_heading() -> None:
    # Matrix row "Page break": a paragraph runs from page 4 onto page 5 (Docling gives each page its part).
    found, _outline = cut_pdf(
        pdf_page(4, block("Grappling", 2), block("When you want to grab a creature, you can")),
        pdf_page(5, block("use the Attack action to make a special melee attack.")),
    )
    assert [(item["page"], item["sectionPath"]) for item in found] == [(4, ["Grappling"]), (5, ["Grappling"])]
    assert found[1]["text"] == "Grappling\nuse the Attack action to make a special melee attack."


def test_a_table_that_fits_stays_whole_in_one_passage() -> None:
    # Matrix row "Table fits": 40 rows, about 2,000 characters.
    rows = [["Item", "Cost", "Weight"]] + [
        [f"Adventuring item {i:02d}", f"{i:03d} gold pieces", f"{i:02d} lb."] for i in range(39)
    ]
    table = block("", label="table", rows=rows)
    found, _outline = cut_pdf(pdf_page(7, block("Gear", 2), block("x" * 1500 + "."), table))

    tables = [item for item in found if "Item | Cost | Weight" in item["text"]]
    assert len(tables) == 1
    assert 1800 <= len(passages.table_text(table)) <= 2200
    # It did not fit beside the paragraph: it starts its own passage, which repeats the heading.
    assert tables[0]["text"] == "Gear\n" + passages.table_text(table)
    assert tables[0]["text"].count("\n") == 40


def test_a_long_table_is_cut_between_rows_each_part_starting_with_the_heading_and_the_first_row() -> None:
    # Matrix row "Long table": about 5,000 characters.
    rows = [["Armor", "Cost", "AC"]] + [[f"Plate mark {i:03d}", "1,500 gp", "18"] for i in range(170)]
    table = block("", label="table", rows=rows)
    assert len(passages.table_text(table)) > 5000
    found, _outline = cut_pdf(pdf_page(12, block("Armor", 2), table))

    assert len(found) >= 2
    for item in found:
        assert item["text"].startswith("Armor\nArmor | Cost | AC\n")
        assert len(item["text"]) <= PASSAGE_LIMIT
        assert item["sectionPath"] == ["Armor"]
    body = [line for item in found for line in item["text"].split("\n")[2:]]
    assert body == [f"Plate mark {i:03d} | 1,500 gp | 18" for i in range(170)]


def test_a_table_row_too_long_for_a_part_is_cut_and_a_table_without_rows_to_cut_is_cut_as_a_block() -> None:
    long_row = [["Spell", "Description"], ["Wish", "word " * 900], ["Light", "A glow."]]
    found, _outline = cut_pdf(pdf_page(1, block("Spells", 2), block("", label="table", rows=long_row)))
    # The 4,500-character row in two pieces; the next row joins the second part.
    assert len(found) == 2
    assert all(item["text"].startswith("Spells\nSpell | Description\n") for item in found)
    assert found[0]["text"].split("\n")[2].startswith("Wish | word word")
    assert all(len(item["text"]) <= PASSAGE_LIMIT for item in found)
    assert found[-1]["text"].endswith("Light | A glow.")

    # Rows that are not lists are left out; a table with no row to cut between, or none at all, is cut as text.
    odd = block("", label="table", rows=[["Only", "row " * 1000], "not a row", [None, ""]])
    plain = block("cell " * 700, label="table", rows=None)
    found, _outline = cut_pdf(pdf_page(2, block("Tables", 2), odd, plain), limit=1000)
    assert all(len(item["text"]) <= 1000 for item in found)
    assert found[0]["text"].startswith("Tables\nOnly | row row")
    assert any(item["text"].startswith("Tables\ncell cell") for item in found)
    assert passages.table_text(block(" Plain ", label="table", rows=[[" ", None]])) == "Plain"


def test_a_block_without_a_box_of_four_fractions_adds_no_box() -> None:
    bad = [block("One."), block("Two."), block("Three.")]
    bad[0]["box"] = [0, 0, 1]
    bad[1]["box"] = [0, 0, 1, 1.5]
    bad[2]["box"] = None
    found, _outline = cut_pdf(pdf_page(1, *bad, "not a block"))  # type: ignore[arg-type]
    assert found == [{"page": 1, "sectionPath": [], "text": "One.\nTwo.\nThree.", "boxes": []}]


def test_a_long_stat_block_is_two_passages_the_second_starting_with_its_heading() -> None:
    # Matrix row "Long stat block": "Ancient Dragon", 4,500 characters, one page.
    paragraphs = [f"Trait {i}. " + "The dragon breathes fire. " * 16 for i in range(11)]
    assert 4300 <= sum(len(paragraph) + 1 for paragraph in paragraphs) <= 4700
    found, outline = cut_pdf(pdf_page(88, block("Ancient Dragon", 3), *(block(item) for item in paragraphs)))

    assert len(found) == 2
    assert found[0]["text"].startswith("Ancient Dragon\nTrait 0.")
    assert found[1]["text"].startswith("Ancient Dragon\nTrait ")
    assert all(item["sectionPath"] == ["Ancient Dragon"] and item["page"] == 88 for item in found)
    assert outline == [{"level": 3, "title": "Ancient Dragon", "page": 88}]


def test_one_block_over_the_limit_is_cut_as_the_text_stage_cuts_and_each_part_repeats_the_heading() -> None:
    sentence = "The dragon breathes fire upon the village. "
    found, _outline = cut_pdf(pdf_page(3, block("Ancient Dragon", 2), block(sentence * 110)))

    assert len(found) == 2
    assert all(len(item["text"]) <= PASSAGE_LIMIT for item in found)
    assert found[1]["text"].startswith("Ancient Dragon\n")
    # Cut at whitespace: no word is split.
    rejoined = (
        found[0]["text"].removeprefix("Ancient Dragon\n") + " " + found[1]["text"].removeprefix("Ancient Dragon\n")
    )
    assert rejoined == (sentence * 110).strip()


def test_page_headers_footers_and_blocks_without_text_never_enter_a_passage() -> None:
    found, outline = cut_pdf(
        pdf_page(
            2,
            block("Player's Handbook", label="page_header"),
            block("Rules", 1),
            block("   "),
            block("", label="picture"),
            block("Roll a d20."),
            block("42", label="page_footer"),
            block("", level=2),
        )
    )
    assert [item["text"] for item in found] == ["Rules\nRoll a d20."]
    assert outline == [{"level": 1, "title": "Rules", "page": 2}]


def test_titles_are_single_spaced_a_very_long_heading_is_a_block_and_a_path_stays_within_32() -> None:
    headings = [block(f"Level {level}", level) for level in range(1, 40)]
    found, outline = cut_pdf(
        pdf_page(1, block("A  long\nheading", 1), block("Body.")),
        pdf_page(2, block("H" * 1001, 1), *headings, block("Deep body.")),
    )
    assert found[0]["sectionPath"] == ["A long heading"]
    # The 1,001-character "heading" is read as text, under the heading before it.
    assert any(item["text"] == "A long heading\n" + "H" * 1000 + "H" for item in found)
    deepest = found[-1]
    assert deepest["text"] == "Level 39\nDeep body."
    assert len(deepest["sectionPath"]) == passages.MAX_DEPTH
    assert all(len(item["sectionPath"]) <= passages.MAX_DEPTH for item in found)
    assert len(outline) == 40


def spread(seed: int) -> Iterator[int]:
    """A fixed, varied sequence of numbers (a linear congruential generator): the same every run."""
    value = seed
    while True:
        value = (value * 1_103_515_245 + 12_345) % 2**31
        yield value


def test_every_pdf_passage_fits_the_limit_and_keeps_every_word_in_order() -> None:
    numbers = spread(18)
    words = [f"w{index}" for index in range(30_000)]
    pages = []
    position = 0
    for page in range(1, 25):
        blocks = []
        for _ in range(1 + next(numbers) % 8):
            if next(numbers) % 100 < 15:
                blocks.append(block(f"Heading {position}", 1 + next(numbers) % 4))
                continue
            size = (5, 40, 300, 900)[next(numbers) % 4]
            blocks.append(block(" ".join(words[position : position + size])))
            position += size
        pages.append(pdf_page(page, *blocks))
    found, _outline = cut_pdf(*pages)

    assert all(1 <= len(item["text"]) <= PASSAGE_LIMIT for item in found)
    pages_seen = [item["page"] for item in found]
    assert pages_seen == sorted(pages_seen)
    seen = [word for item in found for word in item["text"].split() if word.startswith("w")]
    assert seen == words[:position]


# --- text records


def text_page(page: int, path: list[str], content: str) -> dict[str, Any]:
    return {"page": page, "headingPath": path, "text": content, "start": 0, "end": len(content)}


def cut_text(*pages: dict[str, Any], limit: int = PASSAGE_LIMIT) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cutter = TextCutter(limit)
    for page in pages:
        cutter.page(page)
    return cutter.finish()


def test_markdown_pages_carry_their_heading_path_and_a_range_in_their_page() -> None:
    # Matrix row "Markdown": `# A` / `## B`.
    first = "# A\n\nUnder A.\n\n"
    second = "## B\n\nUnder B.\n"
    found, outline = cut_text(text_page(1, ["A"], first), text_page(2, ["A", "B"], second))

    assert found == [
        {"page": 1, "sectionPath": ["A"], "text": "# A\n\nUnder A.", "range": [0, 13]},
        {"page": 2, "sectionPath": ["A", "B"], "text": "## B\n\nUnder B.", "range": [0, 14]},
    ]
    assert outline == [{"level": 1, "title": "A", "page": 1}, {"level": 2, "title": "B", "page": 2}]
    assert first[0:13] == found[0]["text"]


def test_a_text_page_is_cut_at_blank_lines_and_a_part_that_does_not_open_its_section_repeats_its_heading() -> None:
    paragraphs = ["## Spells", "x" * 1500 + ".", "y" * 1500 + ".", "z" * 500 + "."]
    content = "\n\n".join(paragraphs)
    found, _outline = cut_text(text_page(1, ["Magic", "Spells"], content), text_page(2, ["Magic", "Spells"], "w."))

    assert [item["text"][:9] for item in found] == ["## Spells", "Spells\nyy", "Spells\nw."]
    for item in found:
        start, end = item["range"]
        page_text = content if item["page"] == 1 else "w."
        assert item["text"].removeprefix("Spells\n") == page_text[start:end]
        assert len(item["text"]) <= PASSAGE_LIMIT
    # The first passage ends at a blank line: whole paragraphs only.
    assert found[0]["text"].endswith("x" * 1500 + ".")
    assert found[1]["text"].endswith("z" * 500 + ".")


def test_a_page_of_the_limit_that_continues_a_section_is_cut_so_its_heading_fits() -> None:
    content = "\n".join(["x" * 299] * 10)
    assert len(content) == 2999
    found, _outline = cut_text(text_page(1, ["Regras"], "# Regras\n\nAbertura."), text_page(2, ["Regras"], content))

    continued = [item for item in found if item["page"] == 2]
    assert len(continued) == 2
    assert all(item["text"].startswith("Regras\n") and len(item["text"]) <= PASSAGE_LIMIT for item in continued)
    assert continued[0]["range"][0] == 0
    assert continued[1]["range"][1] == len(content.rstrip())


def test_plain_text_has_no_path_and_backing_out_of_a_heading_enters_none() -> None:
    found, outline = cut_text(text_page(1, [], "Notas.\n\nMais notas."), text_page(2, [], "Fim."))
    assert [(item["sectionPath"], item["text"]) for item in found] == [([], "Notas.\n\nMais notas."), ([], "Fim.")]
    assert outline == []

    found, outline = cut_text(
        text_page(1, ["A", "B"], "## B\n\nCorpo."), text_page(2, ["A"], "Volta a A."), text_page(3, ["C"], "# C")
    )
    assert [(item["sectionPath"], item["text"]) for item in found] == [
        (["A", "B"], "## B\n\nCorpo."),
        (["A"], "A\nVolta a A."),
        (["C"], "# C"),
    ]
    assert [entry["title"] for entry in outline] == ["A", "B", "C"]


def test_an_empty_or_blank_page_gives_no_passage_and_empty_titles_leave_the_path() -> None:
    found, outline = cut_text(text_page(1, ["", "A"], "  \n\n \n"), text_page(2, ["", "A"], "Corpo."))
    assert found == [{"page": 2, "sectionPath": ["A"], "text": "A\nCorpo.", "range": [0, 6]}]
    assert outline == [{"level": 1, "title": "A", "page": 1}]


def test_every_text_passage_fits_and_its_range_is_its_text() -> None:
    numbers = spread(181)
    pages = []
    for page in range(1, 40):
        paragraphs = [" ".join("p" * (1 + next(numbers) % 12) for _ in range(1 + next(numbers) % 500))]
        while sum(len(item) + 2 for item in paragraphs) < 2800 and next(numbers) % 10 < 8:
            paragraphs.append(" ".join("q" * (1 + next(numbers) % 9) for _ in range(1 + next(numbers) % 80)))
        content = "\n\n".join(paragraphs)[:3000]
        pages.append(text_page(page, ["Livro", f"Parte {page // 7}"], content))
    found, _outline = cut_text(*pages)

    by_page = {page["page"]: page["text"] for page in pages}
    for item in found:
        assert 1 <= len(item["text"]) <= PASSAGE_LIMIT
        start, end = item["range"]
        assert 0 <= start < end <= len(by_page[item["page"]])
        leaf = item["sectionPath"][-1]
        body = item["text"] if item["text"] == by_page[item["page"]][start:end] else item["text"][len(leaf) + 1 :]
        assert body == by_page[item["page"]][start:end]


# --- the text files


def test_records_are_read_in_page_order_within_their_file(tmp_path: Path) -> None:
    path = tmp_path / "texts.json"
    path.write_text(json.dumps({"pages": [pdf_page(3), pdf_page(5)]}), encoding="utf-8")
    assert [record["page"] for record in passages.read_records(path, 3, 5, pdf=True)] == [3, 5]
    path.write_text(json.dumps({"pages": [text_page(1, ["A"], "a")]}), encoding="utf-8")
    assert passages.read_records(path, 1, 1, pdf=False)[0]["text"] == "a"

    for body, pdf in (
        ({"pages": [pdf_page(5), pdf_page(3)]}, True),
        ({"pages": [pdf_page(6)]}, True),
        ({"pages": [pdf_page(2)]}, True),
        ({"pages": [{"page": 3}]}, True),
        ({"pages": [pdf_page(3)]}, False),
        ({"pages": [text_page(3, [1], "a")]}, False),  # type: ignore[list-item]
        ({"pages": "none"}, True),
        ([], True),
    ):
        path.write_text(json.dumps(body), encoding="utf-8")
        with pytest.raises(passages.RecordsInvalid):
            passages.read_records(path, 3, 5, pdf=pdf)
    path.write_text("{", encoding="utf-8")
    with pytest.raises(passages.RecordsInvalid):
        passages.read_records(path, 3, 5, pdf=True)
