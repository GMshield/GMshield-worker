"""The OCR stage's child process (Story 1.17): read the image pages of a batch with Docling and Tesseract.

Run only through the sandbox (`sandbox.py`), as a script in isolated mode, inside the step folder:

    python -I -B ocr.py SOURCE OUT_DIR OFFSET FIRST LAST ARTIFACTS THREADS

SOURCE is a page batch whose first page is page OFFSET of the version; FIRST to LAST (inclusive, numbered as in the
version) are the pages to read. As in `layout.py`, everything Docling needs comes on the command line: ARTIFACTS is the
folder of its layout and table models (baked into the image: nothing is downloaded) and THREADS the number of threads
torch, Docling and Tesseract (`OMP_THREAD_LIMIT`) may use.

Each page is rendered with pypdfium2 at `SCALE` (216 DPI). A page lying at 90, 180 or 270 degrees is turned upright
first, by Tesseract's orientation detection (the `tesseract` command, `--psm 0 -l osd`, as Docling runs it): Docling
runs the same detection, but it turns only the words it reads, not the picture its layout model and reading order
see, so an upside-down page would come out with its blocks in reverse (and since Story 1.22 Docling's own detection
is switched off: the page reaches it upright, and it reads it as it lies). The detection can be half a turn wrong on
old typefaces (Story 1.22: every page of an 1892 Portuguese book, upright ones included, came out as upside down), so
its answer is checked by reading: Tesseract (`por+eng`) reads a band across the middle of the page turned as
detected and turned half a turn further, and the turn with more words read with confidence wins; a tie keeps the
detection. Its skew is then estimated within `MAX_SKEW`
degrees with OpenCV (the angle whose rotation gives the sharpest profile of ink rows), and the page is rotated, with
a white fill on a canvas that keeps every corner, when the angle is at least `MIN_SKEW`. The straightened picture
goes to Docling as an image (`InputFormat.IMAGE`, at 216 DPI, so Docling reads it at its own size) with Tesseract
(`por` and `eng`) reading the full page, table structure and heading levels on. The converter is built once per
child.

It imports Docling, OpenCV, numpy, Pillow, pypdfium2 and the standard library, and `layout.py` beside it by path
(never `gmshield_worker`: the isolated child has no path to it). It holds no credential, reaches no network, and
writes only under OUT_DIR: one `page-NNNNNNN.json` per page read, in exactly `layout.py`'s shape (its boxes are
fractions of the straightened picture, and its lines are the words Tesseract read), the picture of the page being
read while Docling reads it, and `manifest.json`, `{"pages": [N, ...], "unreadable": [N, ...], "pictures": [{"page",
"width", "height"}]}`. A page whose OCR yields fewer than `MIN_CHARACTERS` characters other than whitespace (a blank
scan) is listed `unreadable` and gets no page file. Every page read, unreadable ones included, also keeps its
straightened picture as `page-NNNNNNN.webp` (Story 1.19), shrunk to layout's `PICTURE_SIDE` and written by
`layout.write_webp`: its boxes are fractions of that same picture. Exit status 0 means done; anything else is a crash,
which the parent reports. Nothing is printed.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2
import numpy as np
import pypdfium2 as pdfium  # type: ignore[import-untyped]
from PIL import Image

MANIFEST = "manifest.json"
PICTURE = ".page.png"
# 3 x 72 DPI: the resolution Docling's Tesseract reader works at, so the picture reaches it unscaled.
SCALE = 3.0
# The PNG's resolution, whatever scale rendered it: Docling reads the picture at 3 x its size in points, that is at
# its own size in pixels, never scaled up. Boxes are fractions of the picture, so its size in points does not matter.
DPI = 72.0 * SCALE
# A page is rendered at `SCALE` unless its longer side would pass this many pixels (PDF pages reach 14,400 points):
# then at the scale that keeps it there, so a huge page never takes gigabytes.
MAX_SIDE = 10_000
MIN_CHARACTERS = 16
# The skew search: whole degrees within MAX_SKEW, then tenths around the best; smaller than MIN_SKEW is left alone.
MAX_SKEW = 20.0
MIN_SKEW = 0.5
COARSE_STEP = 1.0
FINE_STEP = 0.1
# The skew is estimated on a copy whose longer side is at most this many pixels.
SKEW_SIDE = 1000
# Less ink than this share of the copy (a blank scan) has no skew to estimate.
MIN_INK = 0.001
WHITE = (255, 255, 255)
# How long Tesseract's orientation detection may take before the page is read as it lies: well under a page-by-page
# retry's share of the step (540 s / 10 pages = 54 s).
OSD_SECONDS = 20
# The line of Tesseract's `--psm 0` report that says how many degrees, counter-clockwise, turn the page upright.
_ORIENTATION = re.compile(r"^Orientation in degrees: *(0|90|180|270) *$", re.MULTILINE | re.ASCII)
# Story 1.22: the detected turn is checked by reading a band of the page (from `BAND_TOP` to `BAND_BOTTOM` of the
# turned and straightened page's height, `BAND_SIDE` in from each side) as detected and half a turn further. A word
# counts when Tesseract gives it at least `CONFIDENT_WORD` (0 to 100) and it has two letters or digits. Half a turn
# further wins only by `MIN_LEAD` words or more: text read upside down still yields a few such words (4 to 9 on the
# test scans). Each read may take `READ_SECONDS`, like the detection.
BAND_TOP = 0.35
BAND_BOTTOM = 0.65
BAND_SIDE = 0.1
CONFIDENT_WORD = 60.0
MIN_LEAD = 10
READ_SECONDS = 20
BAND = ".band.png"
# Tesseract's TSV report: twelve columns, the confidence in the eleventh and the word in the last.
_TSV_COLUMNS = 12

type Reader = Callable[[Path], Any]
type Orienter = Callable[[Path], int]


def _sibling(name: str) -> ModuleType:
    """A module of this folder, loaded by path: the isolated child has no path to `gmshield_worker`."""
    spec = importlib.util.spec_from_file_location(f"_gmshield_{name}", Path(__file__).with_name(f"{name}.py"))
    if spec is None or spec.loader is None:  # pragma: no cover - the file sits beside this one
        raise ImportError(name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


layout = _sibling("layout")


def _rotation(angle: float, width: int, height: int, *, expand: bool) -> tuple[np.ndarray, tuple[int, int]]:
    """The affine matrix that turns a `width` x `height` picture by `angle` degrees (counter-clockwise) about its
    centre, and the size of the result: the same, or (`expand`) large enough to keep every corner."""
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    if not expand:
        return matrix, (width, height)
    cosine, sine = abs(math.cos(math.radians(angle))), abs(math.sin(math.radians(angle)))
    new_width = math.ceil(height * sine + width * cosine)
    new_height = math.ceil(height * cosine + width * sine)
    matrix[0, 2] += new_width / 2.0 - width / 2.0
    matrix[1, 2] += new_height / 2.0 - height / 2.0
    return matrix, (new_width, new_height)


def _sharpness(ink: np.ndarray, angle: float) -> float:
    """How sharply the rows of `ink` turned by `angle` separate lines of text from the gaps between them."""
    height, width = ink.shape
    matrix, size = _rotation(angle, width, height, expand=False)
    turned = cv2.warpAffine(ink, matrix, size, flags=cv2.INTER_NEAREST, borderValue=0)
    rows = turned.sum(axis=1, dtype=np.float64)
    return float(np.sum(np.diff(rows) ** 2))


def skew_angle(gray: np.ndarray) -> float:
    """The rotation, in degrees within `MAX_SKEW` (counter-clockwise positive), that straightens the lines of a
    grayscale page; 0 for a page with almost no ink. On a tie the smaller rotation wins."""
    height, width = gray.shape
    factor = min(1.0, SKEW_SIDE / max(height, width))
    small = cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA) if factor < 1.0 else gray
    _threshold, ink = cv2.threshold(small, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if cv2.countNonZero(ink) < MIN_INK * ink.size:
        return 0.0

    def best(angles: list[float]) -> float:
        return max(angles, key=lambda angle: (_sharpness(ink, angle), -abs(angle)))

    steps = round(MAX_SKEW / COARSE_STEP)
    coarse = best([index * COARSE_STEP for index in range(-steps, steps + 1)])
    fine_steps = round(COARSE_STEP / FINE_STEP)
    fine = [round(coarse + index * FINE_STEP, 1) for index in range(-fine_steps, fine_steps + 1)]
    return best([angle for angle in fine if abs(angle) <= MAX_SKEW])


def straighten(picture: np.ndarray) -> tuple[np.ndarray, float]:
    """The page turned by its skew angle with a white fill, when the angle is at least `MIN_SKEW`, and the angle."""
    gray = cv2.cvtColor(picture, cv2.COLOR_RGB2GRAY) if picture.ndim == 3 else picture
    angle = skew_angle(gray)
    if abs(angle) < MIN_SKEW:
        return picture, angle
    height, width = gray.shape
    matrix, size = _rotation(angle, width, height, expand=True)
    fill = WHITE if picture.ndim == 3 else 255
    turned = cv2.warpAffine(picture, matrix, size, flags=cv2.INTER_CUBIC, borderValue=fill)
    return turned, angle


def parse_orientation(report: str) -> int:
    """The quarter turns, in degrees counter-clockwise, that set a page upright, from Tesseract's `--psm 0` report; 0
    when it gives none."""
    found = _ORIENTATION.search(report)
    return int(found.group(1)) if found else 0


def detect(command: str, picture: Path) -> int:
    """Tesseract's orientation detection on a page picture; 0 when it finds too little text to tell (a blank page)."""
    try:
        done = subprocess.run(  # noqa: S603 - the image's own Tesseract on our own picture, never a shell
            [command, str(picture), "stdout", "--psm", "0", "-l", "osd"],
            capture_output=True,
            stdin=subprocess.DEVNULL,
            check=True,
            timeout=OSD_SECONDS,
        )
    except OSError, subprocess.SubprocessError:
        return 0
    return parse_orientation(done.stdout.decode("utf-8", "replace"))


def confident_words(report: str) -> int:
    """The words of Tesseract's TSV report read with at least `CONFIDENT_WORD` confidence and holding two letters or
    digits."""
    count = 0
    for row in report.splitlines()[1:]:
        columns = row.split("\t")
        if len(columns) != _TSV_COLUMNS:
            continue
        try:
            confidence = float(columns[10])
        except ValueError:
            continue
        if confidence >= CONFIDENT_WORD and sum(1 for character in columns[11] if character.isalnum()) >= 2:
            count += 1
    return count


def band(picture: np.ndarray, degrees: int) -> np.ndarray:
    """The band across the middle of the picture turned `degrees` counter-clockwise and straightened, which
    `readable_words` reads."""
    turned = straighten(upright(picture, degrees))[0]
    height, width = turned.shape[:2]
    top, bottom = int(height * BAND_TOP), int(height * BAND_BOTTOM)
    side = int(width * BAND_SIDE)
    return np.ascontiguousarray(turned[top:bottom, side : width - side])


def readable_words(command: str, picture: np.ndarray, degrees: int, target: Path) -> int:
    """How many words Tesseract reads with confidence in the page's band turned `degrees`; 0 when it fails."""
    write_picture(band(picture, degrees), target)
    try:
        done = subprocess.run(  # noqa: S603 - the image's own Tesseract on our own picture, never a shell
            [command, str(target), "stdout", "-l", "por+eng", "--psm", "3", "tsv"],
            capture_output=True,
            stdin=subprocess.DEVNULL,
            check=True,
            timeout=READ_SECONDS,
        )
    except OSError, subprocess.SubprocessError:
        return 0
    finally:
        target.unlink(missing_ok=True)
    return confident_words(done.stdout.decode("utf-8", "replace"))


def orientation(picture: Path) -> int:
    """The quarter turns, in degrees counter-clockwise, that set the page picture upright: Tesseract's detection,
    unless half a turn further reads at least `MIN_LEAD` more words with confidence. 0 when Tesseract is missing
    (outside the image)."""
    command = shutil.which("tesseract")
    if command is None:
        return 0
    detected = detect(command, picture)
    opposite = (detected + 180) % 360
    try:
        with Image.open(picture) as opened:
            pixels = np.array(opened.convert("L"))
    except OSError:
        return detected
    target = picture.with_name(BAND)
    as_detected = readable_words(command, pixels, detected, target)
    return opposite if readable_words(command, pixels, opposite, target) >= as_detected + MIN_LEAD else detected


def upright(picture: np.ndarray, degrees: int) -> np.ndarray:
    """The picture turned `degrees` (a multiple of 90) counter-clockwise."""
    turns = (degrees // 90) % 4
    return np.ascontiguousarray(np.rot90(picture, turns)) if turns else picture


def render(document: Any, index: int) -> np.ndarray:
    """Page `index` (from 0) of a pypdfium2 document as an RGB picture at `SCALE`, or smaller for a page whose longer
    side would pass `MAX_SIDE` pixels."""
    page = document[index]
    try:
        width, height = page.get_size()
        scale = min(SCALE, MAX_SIDE / max(width, height, 1.0))
        bitmap = page.render(scale=scale, rev_byteorder=True)
        try:
            return np.array(bitmap.to_numpy()[:, :, :3], dtype=np.uint8)
        finally:
            bitmap.close()
    finally:
        page.close()


def write_picture(picture: np.ndarray, target: Path) -> None:
    """The picture as a PNG at `DPI`, which Docling reads as the page's size."""
    Image.fromarray(picture).save(target, format="PNG", dpi=(DPI, DPI))


def characters(record: dict[str, Any]) -> int:
    """The characters other than whitespace that OCR read on a page (its words), counted up to `MIN_CHARACTERS`."""
    count = 0
    for line in record["lines"]:
        count += sum(1 for character in line["text"] if not character.isspace())
        if count >= MIN_CHARACTERS:
            break
    return count


def reader(artifacts: Path, threads: int) -> Reader:  # pragma: no cover - runs only in the image
    """Docling with Tesseract over one straightened page picture, offline, built once. Runs only in the image."""
    import torch  # noqa: PLC0415

    torch.set_num_threads(threads)
    from docling.datamodel.accelerator_options import AcceleratorOptions  # noqa: PLC0415
    from docling.datamodel.base_models import InputFormat  # noqa: PLC0415
    from docling.datamodel.pipeline_options import (  # noqa: PLC0415
        HeadingHierarchyOptions,
        OcrMode,
        PdfPipelineOptions,
        TesseractCliOcrOptions,
    )
    from docling.document_converter import DocumentConverter, ImageFormatOption  # noqa: PLC0415
    from docling.models.stages.ocr import tesseract_ocr_cli_model  # noqa: PLC0415

    # Story 1.22: the picture reaches Docling already upright (`orientation`, checked by reading). Docling 2.145.0's
    # Tesseract model runs the same detection again and turns the picture by it, with no option to stop it; on the
    # 1892 scan that detection is half a turn wrong and undid the turn. So the model reads the picture as it lies:
    # no detection (one Tesseract run fewer per page). It names its languages (`lang`), so it never needs the
    # detection's script either. `tests/image` reads the 1892 page through this child.
    setattr(tesseract_ocr_cli_model.TesseractOcrCliModel, "_perform_osd", lambda _self, _file: None)  # noqa: B010
    setattr(tesseract_ocr_cli_model, "_parse_orientation", lambda _osd: 0)  # noqa: B010

    options = PdfPipelineOptions(
        artifacts_path=artifacts,
        do_ocr=True,
        ocr_options=TesseractCliOcrOptions(lang=["por", "eng"], mode=OcrMode.FULL_PAGE),
        do_table_structure=True,
        generate_parsed_pages=True,
        heading_hierarchy_options=HeadingHierarchyOptions(enabled=True),
        accelerator_options=AcceleratorOptions(num_threads=threads, device="cpu"),
    )
    converter = DocumentConverter(
        allowed_formats=[InputFormat.IMAGE],
        format_options={InputFormat.IMAGE: ImageFormatOption(pipeline_options=options)},
    )

    def read(picture: Path) -> Any:
        return converter.convert(picture)

    return read


type PageRead = tuple[dict[str, Any] | None, tuple[int, int] | None]


def read_page(
    document: Any,
    index: int,
    number: int,
    picture: Path,
    read: Reader,
    *,
    orient: Orienter = orientation,
    view: Path | None = None,
) -> PageRead:
    """Page `index` of the batch, set upright, straightened and read, as page `number` of the version: its record, or
    None when unreadable; and, when `view` is given, the size of the straightened picture written there as WebP."""
    size: tuple[int, int] | None = None
    try:
        rendered = render(document, index)
        write_picture(rendered, picture)
        straight = straighten(upright(rendered, orient(picture)))[0]
        del rendered
        write_picture(straight, picture)
        if view is not None:
            size = layout.write_webp(straight, view)
        del straight
        result = read(picture)
    finally:
        picture.unlink(missing_ok=True)
    records = layout.page_records(result, number)
    if len(records) != 1:
        return None, size
    record: dict[str, Any] = records[0]
    return (record if characters(record) >= MIN_CHARACTERS else None), size


def run(
    source: Path,
    out: Path,
    *,
    offset: int,
    first: int,
    last: int,
    artifacts: Path,
    threads: int,
    make_reader: Callable[[Path, int], Reader] = reader,
    orient: Orienter = orientation,
) -> int:
    if offset < 1 or first < offset or last < first or threads < 1:
        return 2
    document = pdfium.PdfDocument(source)
    try:
        if last - offset + 1 > len(document):
            return 3
        read = make_reader(artifacts, threads)
        out.mkdir(exist_ok=True)
        written: list[int] = []
        unreadable: list[int] = []
        pictures: list[dict[str, int]] = []
        for number in range(first, last + 1):
            record, size = read_page(
                document,
                number - offset,
                number,
                out / PICTURE,
                read,
                orient=orient,
                view=out / layout.picture_file(number),
            )
            if size is not None:
                pictures.append({"page": number, "width": size[0], "height": size[1]})
            if record is None:
                unreadable.append(number)
                continue
            (out / layout.page_file(number)).write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
            written.append(number)
    finally:
        document.close()
    manifest = {"pages": written, "unreadable": unreadable, "pictures": pictures}
    (out / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) != 8:
        return 2
    try:
        offset, first, last, threads = int(argv[3]), int(argv[4]), int(argv[5]), int(argv[7])
    except ValueError:
        return 2
    if threads >= 1:
        # Tesseract's threads, set before this child or Docling starts it.
        os.environ["OMP_THREAD_LIMIT"] = str(threads)
    return run(
        Path(argv[1]), Path(argv[2]), offset=offset, first=first, last=last, artifacts=Path(argv[6]), threads=threads
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv))
