"""Checks that run only inside the production image, with no network:

    docker run --rm --network none --read-only --tmpfs /tmp <image> python /app/tests/image/test_image.py

Tesseract lists Portuguese and English, and Docling converts a small generated PDF with the
layout and table models baked into the image (HF_HUB_OFFLINE=1, DOCLING_ARTIFACTS_PATH), with
Tesseract as its OCR engine. Since Story 1.15 the clean stage's child runs in its sandbox
(isolated, with pikepdf from the image's environment and no network socket). Since Story 1.16 the
triage child and the layout child run there too: Docling reads a generated two-column page with a
table offline, with its models given on the command line (the child's environment is empty). Since Story 1.17 the
OCR child reads generated scans there (straight, turned 7 degrees, upside down, and blank) with Docling and Tesseract;
the scans come from `tests/pdfs.py`, which the image carries for these checks. Since Story 1.21 the images child
runs there too: it decodes a HEIC with the image's pi-heif, turns a photographed page upright by its EXIF
orientation, and assembles them into a PDF whose pages the OCR child reads. The file runs as a plain script (the
image has no pytest); pytest collects it too, but `pyproject.toml` leaves `tests/image` out of the ordinary test run.

`two-colours.heic` (528 bytes, 96 x 64, red on its left half and blue on its right) was made for these tests on
2026-11-17: drawn with Pillow 13.0.0 and encoded with pillow-heif 1.8.1 (libheif 1.23.6, x265 4.3) in a throwaway
environment outside the project. It holds nothing but those two rectangles; pillow-heif is never a dependency here.

`coimbra-1892-p2.jpg` (Story 1.22; 1802 x 2706, grey, JPEG quality 30, 216 DPI) is page 2 of Theophilo Braga,
"Historia da universidade de Coimbra", volume 1 (Lisbon, 1892), from the Internet Archive item
`historiadauniver01braguoft` (University of Toronto, marked NOT_IN_COPYRIGHT; public domain): its raw 600 DPI scan,
`historiadauniver01braguoft_raw_0026.jp2`, rendered at 216 DPI and turned upright, the scanner's dark background
kept. Tesseract's orientation detection is half a turn wrong on it whichever way it lies, as on every page of that
book; the OCR child must still set it upright.

`python /app/tests/image/test_image.py --ocr-timing PAGES` instead times the OCR child alone on a generated scan of
PAGES pages (Story 1.17's timing check on roberto-server) and prints its time and peak memory.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

SAMPLE_TEXT = "GMshield sample page"


def sample_pdf() -> bytes:
    """A one-page PDF with one line of text, written by hand: no PDF library needed."""
    stream = f"BT /F1 24 Tf 72 720 Td ({SAMPLE_TEXT}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R"
        b" /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    return _pdf(objects)


def _pdf(objects: list[bytes]) -> bytes:
    """A PDF of `objects` (object 1 the catalog), with its cross-reference table."""
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def test_tesseract_lists_portuguese_and_english() -> None:
    listed = subprocess.run(
        ["tesseract", "--list-langs"],  # noqa: S607 - the image's own Tesseract
        capture_output=True,
        text=True,
        check=True,
    )
    languages = set(listed.stdout.split())
    assert {"por", "eng"} <= languages, languages


def test_docling_converts_offline() -> None:
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    artifacts = Path(os.environ["DOCLING_ARTIFACTS_PATH"])
    assert artifacts.is_dir()

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import OcrMode, PdfPipelineOptions, TesseractCliOcrOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions(
        artifacts_path=artifacts,
        do_ocr=True,
        ocr_options=TesseractCliOcrOptions(lang=["por", "eng"], mode=OcrMode.FULL_PAGE),
        do_table_structure=True,
    )
    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
    with tempfile.TemporaryDirectory(dir="/tmp") as folder:
        source = Path(folder) / "sample.pdf"
        source.write_bytes(sample_pdf())
        result = converter.convert(source)
    text = result.document.export_to_markdown()
    assert "GMshield" in text, text


def test_worker_folders_are_private_and_writable() -> None:
    """The work and health folders: owned by the worker, mode 0700, writable with a read-only root."""
    for folder in (Path("/var/lib/gmshield-worker/work"), Path("/run/gmshield-worker")):
        info = folder.stat()
        assert (info.st_uid, info.st_mode & 0o777) == (10001, 0o700), f"{folder}: {info.st_uid} {info.st_mode:o}"
        probe = folder / ".probe"
        probe.write_text("x", encoding="ascii")
        probe.unlink()


def test_the_clean_child_runs_in_its_sandbox() -> None:
    import json
    import threading

    from gmshield_worker.ingestion import clean, sandbox

    with tempfile.TemporaryDirectory(dir="/tmp") as name:
        folder = Path(name)
        (folder / "source.pdf").write_bytes(sample_pdf())
        result = sandbox.run_child(Path(clean.__file__), ["source.pdf", "out"], folder=folder, cancel=threading.Event())
        assert result.returncode == 0, result
        assert json.loads((folder / "out" / "manifest.json").read_text(encoding="utf-8"))["pageTotal"] == 1
        (folder / "dial.py").write_text(
            "import socket\ntry:\n    socket.socket(socket.AF_INET, socket.SOCK_STREAM)\nexcept PermissionError:\n"
            "    raise SystemExit(13)\n",
            encoding="ascii",
        )
        assert sandbox.run_child(folder / "dial.py", [], folder=folder, cancel=threading.Event()).returncode == 13


def two_column_pdf() -> bytes:
    """One page: a heading, two columns of text, a second heading and a ruled table of four rows, written by hand."""
    operations = []

    def text(x: int, y: int, words: str, size: int = 10, font: str = "F1") -> None:
        operations.append(f"BT /{font} {size} Tf {x} {y} Td ({words}) Tj ET")

    text(72, 740, "Chapter One: The Rules", 20, "F2")
    for index, line in enumerate(LEFT_COLUMN):
        text(72, 700 - index * 14, line)
    for index, line in enumerate(RIGHT_COLUMN):
        text(320, 700 - index * 14, line)
    text(72, 590, "Equipment Costs", 14, "F2")
    top = 570
    for row_index, row in enumerate(TABLE_ROWS):
        for column, cell in enumerate(row):
            text(80 + column * 120, top - row_index * 20 - 14, cell, 10, "F2" if row_index == 0 else "F1")
    for row_index in range(len(TABLE_ROWS) + 1):
        operations.append(f"72 {top - row_index * 20} m 432 {top - row_index * 20} l S")
    for column in range(4):
        operations.append(f"{72 + column * 120} {top} m {72 + column * 120} {top - len(TABLE_ROWS) * 20} l S")
    stream = "\n".join(operations).encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R"
        b" /Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
    ]
    return _pdf(objects)


LEFT_COLUMN = [
    "Combat begins when the party meets",
    "a band of goblins on the road. Each",
    "creature rolls for initiative and acts",
    "in order from highest to lowest.",
    "Roll a twenty sided die and add your",
    "Dexterity modifier to the result.",
]
RIGHT_COLUMN = [
    "Spells require components, which are",
    "listed in each spell description. A",
    "caster must have a free hand for the",
    "somatic components of a spell and a",
    "focus or pouch for the material ones.",
    "Some spells need concentration.",
]
TABLE_ROWS = [["Item", "Cost", "Weight"], ["Longsword", "15 gp", "3 lb"], ["Shield", "10 gp", "6 lb"],
              ["Rope", "1 gp", "10 lb"]]  # fmt: skip


def test_the_triage_and_layout_children_read_two_columns_and_a_table_offline_in_their_sandbox() -> None:
    import json
    import threading

    from gmshield_worker.ingestion import layout, sandbox, triage

    artifacts = os.environ.get("DOCLING_ARTIFACTS_PATH", "/opt/docling-models")
    with tempfile.TemporaryDirectory(dir="/tmp") as name:
        folder = Path(name)
        (folder / "source.pdf").write_bytes(two_column_pdf())
        # The page is page 7 of its version, in a batch that starts at page 7.
        triaged = sandbox.run_child(
            Path(triage.__file__), ["source.pdf", "triage", "7", "7", "7"], folder=folder, cancel=threading.Event()
        )
        assert triaged.returncode == 0, triaged
        assert json.loads((folder / "triage" / "manifest.json").read_text(encoding="utf-8")) == {
            "pages": [{"page": 7, "layer": "text"}]
        }
        result = sandbox.run_child(
            Path(layout.__file__),
            ["source.pdf", "out", "7", "7", "7", artifacts, "2"],
            folder=folder,
            cancel=threading.Event(),
        )
        assert result.returncode == 0, result
        manifest = json.loads((folder / "out" / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["pages"] == [7], manifest
        # Story 1.19: the page's WebP picture, its longer side 2,000 pixels (a US Letter page: 1546 x 2000).
        assert manifest["pictures"] == [{"page": 7, "width": 1546, "height": 2000}], manifest
        assert (folder / "out" / "page-0000007.webp").read_bytes()[8:12] == b"WEBP"
        record = json.loads((folder / "out" / "page-0000007.json").read_text(encoding="utf-8"))
    print(f"layout child: {result.seconds:.1f} s, peak {result.max_rss_kib // 1024} MiB")
    assert record["page"] == 7
    blocks = record["blocks"]
    labels = [(block["label"], block["text"][:20]) for block in blocks]
    assert [block["label"] for block in blocks] == ["section_header", "text", "text", "section_header", "table"], labels
    heading, left, right, second, table = blocks
    assert heading["text"] == "Chapter One: The Rules"
    assert heading["headingLevel"] == 1
    # The 14-point sub-heading sits under the 20-point heading: Docling infers its level from the font size.
    assert second["headingLevel"] > heading["headingLevel"], (heading, second)
    # Reading order: the whole left column, then the whole right one.
    assert left["text"] == " ".join(LEFT_COLUMN)
    assert right["text"] == " ".join(RIGHT_COLUMN)
    assert left["box"][0] < 0.2 and right["box"][0] > 0.5
    assert second["text"] == "Equipment Costs"
    assert table["rows"] == TABLE_ROWS
    for block in blocks:
        left_edge, top_edge, right_edge, bottom_edge = block["box"]
        assert 0 <= left_edge < right_edge <= 1 and 0 <= top_edge < bottom_edge <= 1, block
    assert heading["box"][1] < left["box"][1] < second["box"][1] < table["box"][1]
    assert record["lines"][0]["text"] == "Chapter One: The Rules"
    assert {line["text"] for line in record["lines"]} >= set(LEFT_COLUMN) | set(RIGHT_COLUMN)


# Words the OCR check looks for: one Portuguese and one English line of `pdfs.SCAN_LINES`, by their longest words.
OCR_WORDS = ("estalagem", "initiative")


def scan_pdf(folder: Path, pages: int | None = None) -> Path:
    """A scan of two paragraphs (Portuguese, then English): straight, turned 7 degrees, upside down, blank, and turned
    90 degrees (a landscape picture of a portrait page); or `pages` pages straight, crooked and upside down in turn."""
    from tests import pdfs

    lines = [*pdfs.SCAN_LINES[:3], "", "", "", *pdfs.SCAN_LINES[3:]]
    if pages is None:
        pictures = [pdfs.raster_page(lines), pdfs.raster_page(lines, angle=7), pdfs.raster_page(lines, angle=180),
                    pdfs.raster_page(blank=True), pdfs.raster_page(lines, angle=90)]  # fmt: skip
    else:
        pictures = [pdfs.raster_page(lines, angle=(0, 7, -4, 180)[index % 4]) for index in range(pages)]
    return pdfs.scanned(folder / "source.pdf", pictures)


def run_ocr(folder: Path, first: int, last: int) -> tuple[Any, dict[str, Any]]:
    import json
    import threading

    from gmshield_worker.ingestion import ocr, sandbox

    artifacts = os.environ.get("DOCLING_ARTIFACTS_PATH", "/opt/docling-models")
    result = sandbox.run_child(
        Path(ocr.__file__),
        ["source.pdf", "out", "1", str(first), str(last), artifacts, "2"],
        folder=folder,
        cancel=threading.Event(),
    )
    assert result.returncode == 0, result
    manifest = json.loads((folder / "out" / "manifest.json").read_text(encoding="utf-8"))
    return result, manifest


def test_the_ocr_child_straightens_and_reads_scans_offline_in_its_sandbox() -> None:
    import json

    from gmshield_worker.ingestion import layout
    from tests import pdfs

    with tempfile.TemporaryDirectory(dir="/tmp") as name:
        folder = Path(name)
        scan_pdf(folder)
        result, manifest = run_ocr(folder, 1, 5)
        assert {key: manifest[key] for key in ("pages", "unreadable")} == {"pages": [1, 2, 3, 5], "unreadable": [4]}
        records = [json.loads((folder / "out" / layout.page_file(page)).read_text(encoding="utf-8")) for page in
                   (1, 2, 3, 5)]  # fmt: skip
        written = sorted(path.name for path in (folder / "out").iterdir())
        expected = ["manifest.json", *(layout.page_file(page) for page in (1, 2, 3, 5))]
        expected += [layout.picture_file(page) for page in range(1, 6)]
        assert written == sorted(expected), written
        # Story 1.19: every page read has its straightened picture, the blank one too, its longer side 2,000 pixels.
        assert [item["page"] for item in manifest["pictures"]] == [1, 2, 3, 4, 5], manifest
        assert all(max(item["width"], item["height"]) == layout.PICTURE_SIDE for item in manifest["pictures"])
    print(f"ocr child: 5 pages in {result.seconds:.1f} s, peak {result.max_rss_kib // 1024} MiB")
    for record in records:
        text = " ".join(block["text"] for block in record["blocks"])
        assert set(record) == {"page", "lines", "blocks"}, record.keys()
        for word in OCR_WORDS:
            assert word in text, (record["page"], text)
        # Reading order: the Portuguese paragraph comes before the English one, also upside down and sideways.
        assert text.index("estalagem") < text.index("initiative"), (record["page"], text)
        assert pdfs.SCAN_LINES[0].split()[0] in text, (record["page"], text)
        for item in [*record["lines"], *record["blocks"]]:
            left, top, right, bottom = item["box"]
            assert 0 <= left <= right <= 1 and 0 <= top <= bottom <= 1, item
        # The first line sits near the top of the straightened picture.
        first = min(record["blocks"], key=lambda block: block["box"][1])
        assert first["box"][1] < 0.3, (record["page"], first)


COIMBRA = Path(__file__).with_name("coimbra-1892-p2.jpg")


def test_an_1892_page_is_set_upright_whichever_way_it_lies() -> None:
    import shutil

    import numpy as np
    from PIL import Image

    from gmshield_worker.ingestion import ocr

    command = shutil.which("tesseract")
    assert command is not None
    with Image.open(COIMBRA) as opened:
        page = np.array(opened.convert("RGB"))
    with tempfile.TemporaryDirectory(dir="/tmp") as name:
        picture = Path(name) / ocr.PICTURE
        found = []
        for lying in (0, 90, 180, 270):  # the page lies turned this far counter-clockwise
            ocr.write_picture(ocr.upright(page, lying), picture)
            found.append((lying, ocr.detect(command, picture), ocr.orientation(picture)))
    print(f"1892 page (lying, detected, confirmed): {found}")
    assert [confirmed for _lying, _detected, confirmed in found] == [0, 270, 180, 90], found


def test_the_ocr_child_reads_an_1892_page_upright_whichever_way_it_lies() -> None:
    import json

    from PIL import Image

    from gmshield_worker.ingestion import layout
    from tests import pdfs

    with tempfile.TemporaryDirectory(dir="/tmp") as name, Image.open(COIMBRA) as opened:
        folder = Path(name)
        page = opened.convert("L")
        pdfs.scanned(folder / "source.pdf", [page, page.rotate(90, expand=True)])
        result, manifest = run_ocr(folder, 1, 2)
        assert manifest["pages"] == [1, 2], manifest
        texts = [
            " ".join(block["text"] for block in json.loads((folder / "out" / layout.page_file(number)).read_text(
                encoding="utf-8"))["blocks"])
            for number in (1, 2)
        ]  # fmt: skip
    print(f"ocr child: the 1892 page twice in {result.seconds:.1f} s, peak {result.max_rss_kib // 1024} MiB")
    for text in texts:
        # Read upright, in reading order: the first paragraph, then the second (the running head is not a block).
        for words in ("grande crise", "synthese positiva", "Universidades na Europa"):
            assert words in text, text[:300]
        assert text.index("synthese positiva") < text.index("Universidades na Europa"), text[:300]


HEIC = Path(__file__).with_name("two-colours.heic")


def test_the_images_child_assembles_photos_offline_in_its_sandbox_and_ocr_reads_them_upright() -> None:
    import json
    import threading

    from PIL import Image

    from gmshield_worker.ingestion import images, ocr, sandbox
    from tests import pdfs

    with tempfile.TemporaryDirectory(dir="/tmp") as name:
        folder = Path(name)
        page = pdfs.raster_page(list(pdfs.SCAN_LINES)).convert("RGB")
        # A photo taken with the phone turned: its pixels lie on their side, its EXIF orientation (6) says so.
        exif = Image.Exif()
        exif[0x0112] = 6
        page.rotate(90, expand=True).save(folder / "image-00001", format="JPEG", quality=92, exif=exif.tobytes())
        page.save(folder / "image-00002", format="PNG")
        (folder / "image-00003").write_bytes(HEIC.read_bytes())
        (folder / "image-00004").write_bytes(b"\x00\x00\x00\x18ftypheic" + bytes(400))
        result = sandbox.run_child(
            Path(images.__file__),
            ["out", "image-00001", "image-00002", "image-00003", "image-00004"],
            folder=folder,
            cancel=threading.Event(),
        )
        assert result.returncode == 0, result
        manifest = json.loads((folder / "out" / images.MANIFEST).read_text(encoding="utf-8"))
        assert manifest["pageTotal"] == 4 and manifest["unreadable"] == [4], manifest
        import pypdfium2 as pdfium  # type: ignore[import-untyped]

        document = pdfium.PdfDocument((folder / "out" / images.ORIGINAL).read_bytes())
        try:
            heic = document[2].render(scale=3).to_pil().convert("RGB")
        finally:
            document.close()
        assert heic.size == (96, 64), heic.size
        left, right = heic.getpixel((10, 30)), heic.getpixel((80, 30))
        assert isinstance(left, tuple) and isinstance(right, tuple) and left[0] > 180 and right[2] > 180, (left, right)
        (folder / "source.pdf").write_bytes((folder / "out" / "batch-00001.pdf").read_bytes())
        for path in (folder / "out").iterdir():
            path.unlink()
        (folder / "out").rmdir()
        _, read = run_ocr(folder, 1, 2)
        assert read["pages"] == [1, 2], read
        for number in (1, 2):
            record = json.loads((folder / "out" / ocr.layout.page_file(number)).read_text(encoding="utf-8"))
            text = " ".join(block["text"] for block in record["blocks"])
            for word in OCR_WORDS:
                assert word in text, (number, text)
    print(f"images child: 4 images in {result.seconds:.1f} s, peak {result.max_rss_kib // 1024} MiB")


def ocr_timing(pages: int) -> int:
    """The OCR child alone over a generated scan of `pages` pages: its time, peak memory and outcome."""
    with tempfile.TemporaryDirectory(dir="/tmp") as name:
        folder = Path(name)
        scan_pdf(folder, pages)
        result, manifest = run_ocr(folder, 1, pages)
    print(
        f"ocr timing: {pages} pages in {result.seconds:.1f} s ({result.seconds / pages:.1f} s per page), "
        f"peak {result.max_rss_kib // 1024} MiB; read {len(manifest['pages'])}, "
        f"unreadable {len(manifest['unreadable'])}"
    )
    return 0 if len(manifest["pages"]) == pages else 1


# The HEALTHCHECK's command in the Dockerfile, word for word: an emptied environment, so no secret of the worker's sits
# in the environment of a process a parsing child could read.
HEALTHCHECK = [
    "/usr/bin/env",
    "-i",
    "PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin",
    "PYTHONPATH=/app",
    "PYTHONDONTWRITEBYTECODE=1",
    "python",
    "-m",
    "gmshield_worker.health",
]


def test_the_health_check_works_with_an_emptied_environment() -> None:
    from gmshield_worker import health

    environment = subprocess.run(  # noqa: S603 - the HEALTHCHECK's own command, with a probe in place of the module
        [*HEALTHCHECK[:5], "python", "-c", "import os; print(sorted(os.environ))"],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "GMSHIELD_WORKER_CREDENTIAL": "secret-in-the-worker"},
    )
    assert "GMSHIELD" not in environment.stdout, environment.stdout
    health.mark_healthy(health.HEALTH_FILE)
    try:
        assert subprocess.run(HEALTHCHECK, check=False).returncode == 0  # noqa: S603 - the HEALTHCHECK itself
    finally:
        health.mark_unhealthy(health.HEALTH_FILE)
    assert subprocess.run(HEALTHCHECK, check=False).returncode == 1  # noqa: S603 - the HEALTHCHECK itself


def main() -> int:
    failures = 0
    checks = (
        test_worker_folders_are_private_and_writable,
        test_the_health_check_works_with_an_emptied_environment,
        test_the_clean_child_runs_in_its_sandbox,
        test_the_triage_and_layout_children_read_two_columns_and_a_table_offline_in_their_sandbox,
        test_the_ocr_child_straightens_and_reads_scans_offline_in_its_sandbox,
        test_an_1892_page_is_set_upright_whichever_way_it_lies,
        test_the_ocr_child_reads_an_1892_page_upright_whichever_way_it_lies,
        test_the_images_child_assembles_photos_offline_in_its_sandbox_and_ocr_reads_them_upright,
        test_tesseract_lists_portuguese_and_english,
        test_docling_converts_offline,
    )
    for check in checks:
        try:
            check()
        except Exception as error:
            failures += 1
            print(f"FAIL {check.__name__}: {type(error).__name__}: {error}", file=sys.stderr)
        else:
            print(f"PASS {check.__name__}")
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--ocr-timing":
        sys.exit(ocr_timing(int(sys.argv[2])))
    sys.exit(main())
