"""Checks that run only inside the production image, with no network:

    docker run --rm --network none --read-only --tmpfs /tmp <image> python /app/tests/image/test_image.py

Tesseract lists Portuguese and English, and Docling converts a small generated PDF with the
layout and table models baked into the image (HF_HUB_OFFLINE=1, DOCLING_ARTIFACTS_PATH), with
Tesseract as its OCR engine. The file runs as a plain script (the image has no pytest); pytest
collects it too, but `pyproject.toml` leaves `tests/image` out of the ordinary test run.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

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


def main() -> int:
    failures = 0
    for check in (test_tesseract_lists_portuguese_and_english, test_docling_converts_offline):
        try:
            check()
        except Exception as error:
            failures += 1
            print(f"FAIL {check.__name__}: {type(error).__name__}: {error}", file=sys.stderr)
        else:
            print(f"PASS {check.__name__}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
