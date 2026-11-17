"""Checks that run only inside the production image, with no network:

    docker run --rm --network none --read-only --tmpfs /tmp <image> python /app/tests/image/test_image.py

Tesseract lists Portuguese and English, and Docling converts a small generated PDF with the
layout and table models baked into the image (HF_HUB_OFFLINE=1, DOCLING_ARTIFACTS_PATH), with
Tesseract as its OCR engine. Since Story 1.15 the clean stage's child runs in its sandbox
(isolated, with pikepdf from the image's environment and no network socket). The file runs as a
plain script (the image has no pytest); pytest collects it too, but `pyproject.toml` leaves
`tests/image` out of the ordinary test run.
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
    sys.exit(main())
