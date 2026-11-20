"""The passages cutter (Story 1.18) on page records built by hand: every worker row of the I/O matrix (a path across
batches, a table that fits and one that does not, a long stat block, Markdown, a page break), the outline, the blocks
that never enter a passage, the limits the backend checks, and the records a text file must hold. Since pipeline "7",
the section paths from the file's bookmarks plus one heading, and passages joined to about `PASSAGE_TARGET` (the rows
of the retrieval-quality fix's matrix)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from gmshield_worker.ingestion import passages, text
from gmshield_worker.ingestion.passages import PASSAGE_LIMIT, PASSAGE_TARGET, Bookmark, PdfCutter, TextCutter

BOX = [0.1, 0.2, 0.9, 0.3]


def block(content: str, level: int | None = None, label: str = "text", rows: Any = None) -> dict[str, Any]:
    if level is not None:
        label = "title" if level == 0 else "section_header"
    return {"label": label, "text": content, "box": BOX, "headingLevel": level, "rows": rows}


def pdf_page(page: int, *blocks: dict[str, Any]) -> dict[str, Any]:
    return {"page": page, "lines": [], "blocks": list(blocks)}


def cut_pdf(
    *pages: dict[str, Any], limit: int = PASSAGE_LIMIT, bookmarks: tuple[Bookmark, ...] = ()
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cutter = PdfCutter(limit, bookmarks=bookmarks)
    for page in pages:
        cutter.page(page)
    return cutter.finish()


def paths(found: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Each passage's path, by the passage's first line."""
    return {item["text"].split("\n")[0]: item["sectionPath"] for item in found}


def test_the_limit_is_the_text_stages_page_limit_and_the_target_about_a_paragraph() -> None:
    assert PASSAGE_LIMIT == text.PAGE_LIMIT == 3000
    assert PASSAGE_TARGET == 1200


def test_a_section_path_crosses_batches_and_pages() -> None:
    # Matrix row "Path across batches": the bookmark's path carries from one text file to the next, and the heading
    # under it takes its place below it.
    cutter = PdfCutter(bookmarks=[Bookmark("Magic Items", 0, 19)])
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


def test_a_heading_replaces_the_one_before_it_whatever_its_level() -> None:
    # Docling's levels never nest: every heading replaces the one before it (with no bookmark, the path is the heading).
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
        ["Section 1.1"],
        ["Chapter 2"],
        ["Book Two"],
    ]
    assert [entry["level"] for entry in outline] == [1, 1, 1, 1, 1]
    # A heading with nothing under it before the next one is a passage of its own.
    assert found[0] == {"page": 1, "sectionPath": ["Book"], "text": "Book", "boxes": [BOX]}


def test_the_same_heading_style_ranked_differently_in_two_batches_gives_the_same_path_depth() -> None:
    # Matrix row "Across batches": Docling ranks a style per batch, so one spell-name style is level 6 in one batch
    # and level 1 in the next.
    spells = (Bookmark("Spells", 0, 101), Bookmark("Spell Descriptions", 1, 101))
    found, _outline = cut_pdf(
        pdf_page(101, block("Acid Arrow", 6), block("A shimmering green arrow.")),
        pdf_page(121, block("Fire Bolt", 1), block("You hurl a mote of fire.")),
        bookmarks=spells,
    )
    assert paths(found) == {
        "Acid Arrow": ["Spells", "Spell Descriptions", "Acid Arrow"],
        "Fire Bolt": ["Spells", "Spell Descriptions", "Fire Bolt"],
    }


def test_a_spell_after_a_stat_block_title_is_never_under_that_title() -> None:
    # Matrix row "SRD spell": on page 131, Find Steed's stat block "Otherworldly Steed" (a larger style, level 1)
    # comes before Fireball (level 6), which Docling would nest under it.
    srd = (
        Bookmark("System Reference Document 5.2.1", 0, 1),
        Bookmark("Spells", 0, 103),
        Bookmark("Gaining Spells", 1, 103),
        Bookmark("Spell Descriptions", 1, 106),
        Bookmark("Rules Glossary", 0, 176),
    )
    found, outline = cut_pdf(
        pdf_page(106, block("Spell Descriptions", 1), block("The spells are presented in alphabetical order.")),
        pdf_page(
            131,
            block("Find Steed", 6),
            block("Level 2 Conjuration (Paladin)"),
            block("Otherworldly Steed", 1),
            block("Large Celestial, Fey, or Fiend (Your Choice), Neutral"),
            block("Fireball", 6),
            block("Level 3 Evocation (Sorcerer, Wizard)"),
        ),
        pdf_page(132, block("Fire Bolt", 2), block("Evocation Cantrip (Sorcerer, Wizard)")),
        bookmarks=srd,
    )
    found_paths = paths(found)
    assert found_paths["Spell Descriptions"] == ["Spells", "Spell Descriptions"]
    assert found_paths["Find Steed"] == ["Spells", "Spell Descriptions", "Find Steed"]
    assert found_paths["Otherworldly Steed"] == ["Spells", "Spell Descriptions", "Otherworldly Steed"]
    assert found_paths["Fireball"] == ["Spells", "Spell Descriptions", "Fireball"]
    assert found_paths["Fire Bolt"] == ["Spells", "Spell Descriptions", "Fire Bolt"]
    # The outline follows the same paths: the bookmark of a page with no record (106 is read, 1 and 103 are not) is
    # entered before the next record, and one after the last record still has its entry.
    assert [(entry["level"], entry["title"], entry["page"]) for entry in outline] == [
        (1, "System Reference Document 5.2.1", 1),
        (1, "Spells", 103),
        (2, "Gaining Spells", 103),
        (2, "Spell Descriptions", 106),
        (3, "Find Steed", 131),
        (3, "Otherworldly Steed", 131),
        (3, "Fireball", 131),
        (3, "Fire Bolt", 132),
        (1, "Rules Glossary", 176),
    ]


def test_a_monster_bookmark_enters_at_its_heading_and_its_traits_sit_under_it() -> None:
    # Matrix row "Stat block": the "Tarrasque" bookmark and its heading are one "Tarrasque"; "Traits" sits under it.
    # Two monsters share page 331, so each bookmark enters at its own heading, not at the page's start.
    monsters = (
        Bookmark("Monsters A-Z", 0, 258),
        Bookmark("Succubus", 1, 331),
        Bookmark("Tarrasque", 1, 331),
        Bookmark("Toughs", 1, 332),
        Bookmark("Tough", 2, 332),
    )
    found, outline = cut_pdf(
        pdf_page(
            331,
            block("The rest of the stormy giant's actions."),
            block("SUCCUBUS", 3),
            block("Medium Fiend, Neutral Evil"),
            block("Tarrasque", 2),
            block("The tarrasque is a reptilian juggernaut."),
            block("TARRASQUE", 3),
            block("Gargantuan Monstrosity (Titan), Unaligned"),
            block("Traits", 4),
            block("Legendary Resistance (6/Day)."),
        ),
        pdf_page(332, block("Tough", 3), block("Medium or Small Humanoid, Any Alignment")),
        bookmarks=monsters,
    )
    found_paths = paths(found)
    assert found_paths["The rest of the stormy giant's actions."] == ["Monsters A-Z"]
    assert found_paths["SUCCUBUS"] == ["Monsters A-Z", "Succubus"]
    assert found_paths["Tarrasque"] == ["Monsters A-Z", "Tarrasque"]
    assert found_paths["Traits"] == ["Monsters A-Z", "Tarrasque", "Traits"]
    # "Toughs" matches no block of its page: it enters with "Tough", the next bookmark of the page that does.
    assert found_paths["Tough"] == ["Monsters A-Z", "Toughs", "Tough"]
    assert [entry["title"] for entry in outline] == [
        "Monsters A-Z",
        "Succubus",
        "Tarrasque",
        "Traits",
        "Toughs",
        "Tough",
    ]
    assert outline[3] == {"level": 3, "title": "Traits", "page": 331}
    # The texts are the page's own: a bookmark adds no line. The stat block's own title, after the description, is
    # already in the path: an ordinary block, and the passage stays under the bookmark.
    assert [(item["text"], item["sectionPath"]) for item in found if "Gargantuan" in item["text"]] == [
        (
            "Tarrasque\nThe tarrasque is a reptilian juggernaut.\nTARRASQUE\nGargantuan Monstrosity (Titan), Unaligned",
            ["Monsters A-Z", "Tarrasque"],
        )
    ]


def test_a_heading_already_in_the_path_is_an_ordinary_block() -> None:
    # A running header repeats the current heading on every page; an outer bookmark's title comes back as a heading.
    found, outline = cut_pdf(
        pdf_page(106, block("Spells", 1), block("Casting rules.")),
        pdf_page(107, block("Spell Descriptions", 1), block("Acid Arrow", 2), block("A green arrow.")),
        pdf_page(108, block("Acid Arrow", 2), block("More of the arrow.")),
        bookmarks=(Bookmark("Spells", 0, 106), Bookmark("Spell Descriptions", 1, 107)),
    )
    assert [(item["page"], item["sectionPath"], item["text"]) for item in found] == [
        (106, ["Spells"], "Spells\nCasting rules."),
        (107, ["Spells", "Spell Descriptions"], "Spell Descriptions"),
        (107, ["Spells", "Spell Descriptions", "Acid Arrow"], "Acid Arrow\nA green arrow."),
        (108, ["Spells", "Spell Descriptions", "Acid Arrow"], "Acid Arrow\nAcid Arrow\nMore of the arrow."),
    ]
    assert [entry["title"] for entry in outline] == ["Spells", "Spell Descriptions", "Acid Arrow"]

    # An outer bookmark's title as a heading under an inner one: never "Spells > Spell Descriptions > Spells".
    found, _outline = cut_pdf(
        pdf_page(110, block("SPELLS.", 1), block("Body.")),
        bookmarks=(Bookmark("Spells", 0, 109), Bookmark("Spell Descriptions", 1, 109)),
    )
    assert [item["sectionPath"] for item in found] == [["Spells", "Spell Descriptions"]]


@pytest.mark.parametrize(
    ("mark", "heading"),
    [
        ("Hill Giant's Club", "Hill Giant\u2019s Club"),
        ("Fire Shield", "\ufb01re shield"),
        ("Monsters A-Z", "Monsters A\u2013Z"),
        ("Equipment", "\uff25\uff51\uff55\uff49\uff50\uff4d\uff45\uff4e\uff54."),
        ("Weapons", "  WEAPONS  "),
    ],
)
def test_a_bookmark_matches_its_heading_across_quotes_ligatures_dashes_width_and_punctuation(
    mark: str, heading: str
) -> None:
    assert passages.folded(mark) == passages.folded(heading)
    found, outline = cut_pdf(
        pdf_page(5, block("Before."), block(heading, 2), block("Body.")), bookmarks=(Bookmark(mark, 0, 5),)
    )
    # The bookmark entered at its heading, which is not repeated under it.
    assert [item["sectionPath"] for item in found] == [[], [mark]]
    assert [entry["title"] for entry in outline] == [mark]


def test_a_bookmark_that_matches_no_block_enters_at_its_pages_start_or_end() -> None:
    # No bookmark of page 5 matches a block: they enter at its start. On page 6 "Weapons" matches, so "Notes", after
    # it, enters at the page's end, for the pages that follow.
    bookmarks = (
        Bookmark("Chapter 3", 0, 5),
        Bookmark("Armor", 1, 5),
        Bookmark("Weapons", 1, 6),
        Bookmark("Notes", 1, 6),
    )
    found, _outline = cut_pdf(
        pdf_page(4, block("End of chapter two.")),
        pdf_page(5, block("CAPITULO III", 1), block("Armor protects.")),
        pdf_page(6, block("More on armor."), block("weapons ", 2), block("Swords cut.")),
        pdf_page(7, block("A note.")),
        bookmarks=bookmarks,
    )
    assert [(item["text"], item["sectionPath"]) for item in found] == [
        ("End of chapter two.", []),
        ("CAPITULO III\nArmor protects.", ["Chapter 3", "Armor", "CAPITULO III"]),
        ("CAPITULO III\nMore on armor.", ["Chapter 3", "Armor", "CAPITULO III"]),
        # "weapons " is the bookmark's title, case- and space-folded: not repeated.
        ("weapons\nSwords cut.", ["Chapter 3", "Weapons"]),
        ("A note.", ["Chapter 3", "Notes"]),
    ]


def test_a_bookmark_matches_a_heading_before_a_plain_block_and_a_plain_block_when_no_heading_does() -> None:
    found, _outline = cut_pdf(
        pdf_page(2, block("Intro text."), block("Combat"), block("Rules here."), block("Combat", 1), block("Body.")),
        pdf_page(3, block("Lore"), block("Old tales.")),
        bookmarks=(Bookmark("Combat", 0, 2), Bookmark("Lore", 0, 3)),
    )
    assert [(item["text"], item["sectionPath"]) for item in found] == [
        ("Intro text.\nCombat\nRules here.", []),
        ("Combat\nBody.", ["Combat"]),
        # No heading matches "Lore": the plain block that does is where it enters, and it opens the passage.
        ("Lore\nOld tales.", ["Lore"]),
    ]


def test_bookmarks_are_entered_in_page_order_their_paths_from_the_outline_and_capped() -> None:
    # A bookmark listed before an earlier page's keeps its outline path; a bookmark with no title is left out; a path
    # 40 deep keeps room for a heading.
    deep = tuple(Bookmark(f"Level {depth}", depth, 9) for depth in range(40))
    found, outline = cut_pdf(
        pdf_page(2, block("Early.")),
        pdf_page(8, block("Late.")),
        pdf_page(9, block("Deep.")),
        bookmarks=(
            Bookmark("Part B", 0, 8),
            Bookmark("B1", 1, 8),
            Bookmark("Part A", 0, 2),
            Bookmark("  ", 0, 2),
            Bookmark("A  long\ntitle", 1, 2),
            *deep,
        ),
    )
    # The untitled entry is dropped, and its child takes its place under its parent (none: the top).
    assert [item["sectionPath"] for item in found[:2]] == [["A long title"], ["Part B", "B1"]]
    assert [entry["title"] for entry in outline[:4]] == ["Part A", "A long title", "Part B", "B1"]
    deepest = found[-1]["sectionPath"]
    assert len(deepest) == passages.MAX_DEPTH - 1
    assert deepest[-1] == "Level 39"
    assert deepest[0] == "Level 0"


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


def test_a_long_stat_block_opens_with_up_to_the_limit_and_its_other_parts_repeat_its_heading() -> None:
    # Matrix row "Long stat block": "Ancient Dragon", 4,500 characters, one page. The passage that opens the section
    # takes blocks up to the limit; the rest is joined to the target, each part starting with the heading.
    paragraphs = [f"Trait {i}. " + "The dragon breathes fire. " * 16 for i in range(11)]
    assert 4300 <= sum(len(paragraph) + 1 for paragraph in paragraphs) <= 4700
    found, outline = cut_pdf(pdf_page(88, block("Ancient Dragon", 3), *(block(item) for item in paragraphs)))

    assert found[0]["text"].startswith("Ancient Dragon\nTrait 0.")
    assert PASSAGE_TARGET < len(found[0]["text"]) <= PASSAGE_LIMIT
    assert len(found) == 3
    assert all(item["text"].startswith("Ancient Dragon\nTrait ") for item in found[1:])
    assert all(len(item["text"]) <= PASSAGE_TARGET for item in found[1:])
    assert all(item["sectionPath"] == ["Ancient Dragon"] and item["page"] == 88 for item in found)
    assert outline == [{"level": 1, "title": "Ancient Dragon", "page": 88}]


def test_a_stat_block_of_2400_characters_under_its_heading_is_one_passage() -> None:
    # Matrix row "Stat block whole": the passage opens its section, so it keeps joining up to the limit.
    blocks = [block(f"Action {i}. " + "x" * 290) for i in range(8)]
    assert 2350 <= sum(len(item["text"]) + 1 for item in blocks) <= 2450
    found, _outline = cut_pdf(
        pdf_page(7, block("Ogre", 2), *blocks), bookmarks=(Bookmark("Monsters", 0, 7), Bookmark("Ogre", 1, 7))
    )
    assert len(found) == 1
    assert found[0]["sectionPath"] == ["Monsters", "Ogre"]
    assert found[0]["text"].startswith("Ogre\nAction 0.")

    # A bookmark with no heading under it opens its section too: its first passage takes up to the limit, unprefixed.
    found, _outline = cut_pdf(pdf_page(9, *blocks), bookmarks=(Bookmark("Monsters", 0, 9),))
    assert len(found) == 1
    assert found[0]["text"].startswith("Action 0.")


def test_a_heading_at_the_foot_of_a_page_still_opens_its_section_on_the_next() -> None:
    # The heading ends page 7 alone; its stat block, on page 8, is still the section's opening passage: whole up to
    # the limit, repeating the heading.
    blocks = [block(f"Action {i}. " + "x" * 290) for i in range(8)]
    found, _outline = cut_pdf(pdf_page(7, block("Intro."), block("Ogre", 2)), pdf_page(8, *blocks, block("y" * 900)))
    assert [(item["page"], item["text"][:14]) for item in found] == [
        (7, "Intro."),
        (7, "Ogre"),
        (8, "Ogre\nAction 0."),
        (8, "Ogre\nyyyyyyyyy"),
    ]
    assert PASSAGE_TARGET < len(found[2]["text"]) <= PASSAGE_LIMIT


def test_a_table_joins_a_heading_only_when_the_headings_own_text_leaves_room() -> None:
    # "STRASSE" folds to the bookmark "Straße", a code point shorter: a table that just fits beside the leaf never
    # takes the heading's passage past the limit.
    leaf = "Straße"
    rows = [["Name", "Value"]] + [[f"row {i:03d}", "x" * 20] for i in range(95)]
    room = PASSAGE_LIMIT - len(leaf) - 1
    rows.append(["last", "z" * (room - len(passages.table_text(block("", label="table", rows=rows))) - 1 - 7)])
    table = block("", label="table", rows=rows)
    assert len(passages.table_text(table)) == room
    found, _outline = cut_pdf(
        pdf_page(3, block("STRASSE", 2), table), bookmarks=(Bookmark("Weg", 0, 3), Bookmark(leaf, 1, 3))
    )
    assert all(len(item["text"]) <= PASSAGE_LIMIT for item in found)
    assert [item["text"].split("\n")[0] for item in found] == ["STRASSE", leaf]


def test_a_page_of_paragraphs_is_cut_to_the_target_and_a_larger_block_stands_alone() -> None:
    # Matrix row "Paragraph page": Coimbra page 94, five blocks and no heading.
    sizes = [1466, 547, 47, 362, 397]
    labels = ["text", "text", "footnote", "footnote", "footnote"]
    blocks = [block("p" * (size - 1) + ".", label=label) for size, label in zip(sizes, labels, strict=True)]
    found, _outline = cut_pdf(pdf_page(94, *blocks))
    assert [len(item["text"]) for item in found] == [1466, 958, 397]
    assert all(item["sectionPath"] == [] for item in found)

    # The same page under a heading of an earlier page: each passage repeats it, and none takes past the target
    # (but the larger block).
    found, _outline = cut_pdf(pdf_page(93, block("CAPITULO III", 1), block("Intro.")), pdf_page(94, *blocks))
    sizes_94 = [len(item["text"]) for item in found if item["page"] == 94]
    assert sizes_94 == [13 + 1466, 13 + 547 + 1 + 47 + 1 + 362, 13 + 397]
    assert all(item["sectionPath"] == ["CAPITULO III"] for item in found)


def test_a_scan_without_bookmarks_takes_its_path_from_its_last_heading() -> None:
    # Matrix row "No bookmarks": the Coimbra scan.
    found, outline = cut_pdf(pdf_page(80, block("CAPITULO III", 2), block("A Universidade foi fundada em 1290.")))
    assert [item["sectionPath"] for item in found] == [["CAPITULO III"]]
    assert outline == [{"level": 1, "title": "CAPITULO III", "page": 80}]


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


def test_titles_are_single_spaced_and_a_very_long_heading_is_a_block() -> None:
    found, outline = cut_pdf(
        pdf_page(1, block("A  long\nheading", 1), block("Body.")),
        pdf_page(2, block("H" * 1001, 1), block("Level 39", 39), block("Deep body.")),
    )
    assert found[0]["sectionPath"] == ["A long heading"]
    # The 1,001-character "heading" is read as text, under the heading before it.
    assert any(item["text"] == "A long heading\n" + "H" * 1000 + "H" for item in found)
    assert found[-1] == {"page": 2, "sectionPath": ["Level 39"], "text": "Level 39\nDeep body.", "boxes": [BOX, BOX]}
    assert len(outline) == 2


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
    bookmarks = tuple(Bookmark(f"Part {page}", page % 3, page) for page in range(1, 25, 5))
    found, _outline = cut_pdf(*pages, bookmarks=bookmarks)

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

    # The page opens "Spells": its first passage joins up to the limit; the others join up to the target.
    assert [item["text"][:9] for item in found] == ["## Spells", "Spells\nyy", "Spells\nzz", "Spells\nw."]
    for item in found:
        start, end = item["range"]
        page_text = content if item["page"] == 1 else "w."
        assert item["text"].removeprefix("Spells\n") == page_text[start:end]
        assert len(item["text"]) <= PASSAGE_LIMIT
    # The first passage ends at a blank line: whole paragraphs only.
    assert found[0]["text"].endswith("x" * 1500 + ".")
    assert found[1]["text"].endswith("y" * 1500 + ".")


def test_text_paragraphs_join_to_the_target_unless_their_passage_opens_its_section() -> None:
    paragraphs = ["p" * 499 + "." for _ in range(6)]
    content = "\n\n".join(paragraphs)
    # A page that enters no heading: paragraphs of 500 join two at a time (1,002 code points), never three.
    found, _outline = cut_text(text_page(1, [], content))
    assert [len(item["text"]) for item in found] == [1002, 1002, 1002]
    # A page that enters a heading: its first passage takes up to the limit (its heading and five paragraphs, 2,518).
    found, _outline = cut_text(text_page(1, ["Regras"], "# Regras\n\n" + content))
    assert [len(item["text"]) for item in found] == [2518, len("Regras\n") + 500]


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


@pytest.mark.parametrize(
    "body",
    ["{", "{}", '["x"]', '[{"title": "A", "depth": 0, "page": 1}, 3]', '"bookmarks"'],
)
def test_bookmarks_that_are_not_a_list_of_objects_are_invalid(tmp_path: Path, body: str) -> None:
    path = tmp_path / "bookmarks.json"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(passages.RecordsInvalid):
        passages.read_bookmarks(path, 10)
    with pytest.raises(passages.RecordsInvalid):
        passages.read_bookmarks(tmp_path / "missing.json", 10)


@pytest.mark.parametrize(
    "entry",
    [
        {"title": 7, "depth": 0, "page": 3},
        {"depth": 0, "page": 3},
        {"title": "B", "depth": -1, "page": 3},
        {"title": "B", "depth": True, "page": 3},
        {"title": "B", "depth": "0", "page": 3},
        {"title": "B", "depth": 0, "page": "3"},
        {"title": "B", "depth": 0, "page": 0},
        {"title": "B", "depth": 0, "page": 11},
        {"title": "B", "depth": 0, "page": True},
    ],
)
def test_a_bad_bookmark_entry_is_dropped_and_the_others_kept(tmp_path: Path, entry: dict[str, Any]) -> None:
    path = tmp_path / "bookmarks.json"
    good = [{"title": "A", "depth": 0, "page": 1}, {"title": "C", "depth": 1, "page": 10}]
    path.write_text(json.dumps([good[0], entry, good[1]]), encoding="utf-8")
    assert passages.read_bookmarks(path, 10) == [Bookmark("A", 0, 1), Bookmark("C", 1, 10)]


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
