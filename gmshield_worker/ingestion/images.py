"""The images stage's child process (Story 1.21): assemble an ordered set of images into one image-only PDF.

Run only through the sandbox (`sandbox.py`), as a script in isolated mode, inside the step folder, in one of three
ways:

    python -I -B images.py OUT_DIR IMAGE...              every image, then the PDF (a step's first attempt)
    python -I -B images.py --decode IMAGE OUT_DIR        one image's pages (a retry: one child per image)
    python -I -B images.py --assemble OUT_DIR DECODED... the PDF from the decoded folders (a retry, last)

The images are the set's, in its order. Each one gives one page, and a multi-frame TIFF one page per frame, in
place. Each picture is decoded with Pillow (HEIC through pi-heif), turned upright by its EXIF orientation, laid on
white where it is transparent, kept grey or made RGB, shrunk (never enlarged) to at most `MAX_PIXELS` pixels and
`MAX_SIDE` on its longer side, and encoded as JPEG of quality `QUALITY`, which also drops every other metadata
(the place a photo was taken, among others). Its page measures its pixels at `SCALE` pixels per point (216 DPI), the
resolution the OCR stage renders at, so the photo reaches OCR at its own size. An image (or a frame) that does not
decode is a blank page at its place, listed `unreadable` (OCR then reports it so, as any blank scan): it never fails
the step. The OCR stage then turns each page
upright and straightens it as it does a scan.

It writes, under OUT_DIR, what the clean stage writes: `original.pdf`, `batch-NNNNN.pdf` (every `BATCH_PAGES` pages)
and `manifest.json`, `{"pageTotal": N, "original": "original.pdf", "batches": [{"file", "firstPage", "lastPage"}],
"unreadable": [N, ...]}`. A decoding child writes `pages.json`, `{"pages": [{"file", "width", "height", "gray"} or
{"unreadable": true}]}`, and one `page-NNNNN.jpg` per page; the assembling child reads them (a folder without a valid
`pages.json`, its child having crashed, is one unreadable page) and removes each JPEG once its batch is written.

It imports Pillow, pi-heif, pikepdf and the standard library only (never `gmshield_worker`: the isolated child has no
path to it), holds no credential, reaches no network, and writes only under its output folder. Exit status 0 means
done; anything else is a crash, which the parent reports. Nothing is printed.
"""

from __future__ import annotations

import io
import json
import math
import re
import sys
import warnings
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeIs

from PIL import Image, ImageOps

MANIFEST = "manifest.json"
PAGES = "pages.json"
ORIGINAL = "original.pdf"
BATCH_PAGES = 20
# Pixels per point: 216 DPI, the OCR stage's `SCALE`.
SCALE = 3.0
# The most pixels a page keeps, and its longest side (the OCR stage renders no side past 10,000 pixels).
MAX_PIXELS = 24_000_000
MAX_SIDE = 10_000
# A picture past this many pixels is refused before it is decoded (a decompression bomb): an unreadable page.
MAX_DECODED_PIXELS = 200_000_000
# The most pages a set gives, multi-frame TIFFs included: frames past it are left out.
MAX_PAGES = 10_000
QUALITY = 90
# An unreadable page: blank, A4 in points.
BLANK = (595.0, 842.0)
WHITE = (255, 255, 255)
_PAGE_FILE = re.compile(r"page-[0-9]{5}\.jpg", re.ASCII)

Image.MAX_IMAGE_PIXELS = MAX_DECODED_PIXELS
# Past the limit Pillow only warns up to twice it: every warning is a refusal here.
warnings.simplefilter("error", Image.DecompressionBombWarning)


def _register_heif() -> None:
    """HEIC through pi-heif's Pillow plugin; without it a HEIC is an unreadable page, never a crash."""
    try:
        from pi_heif import register_heif_opener  # type: ignore[import-untyped,unused-ignore]  # noqa: PLC0415
    except ImportError:  # pragma: no cover - pi-heif is always installed in the image
        return
    register_heif_opener()


_register_heif()


@dataclass(frozen=True)
class Page:
    """One page's picture: its JPEG and its size in pixels, grey or RGB; `jpeg` is None for an unreadable page."""

    jpeg: bytes | None
    width: int = 0
    height: int = 0
    gray: bool = False


UNREADABLE = Page(None)


def flattened(image: Image.Image) -> Image.Image:
    """The picture as grey (`L`) or RGB, laid on white where it is transparent; 16-bit and float grey scaled to 8."""
    if image.mode in ("I;16", "I;16L", "I;16B", "I;16N"):
        image = image.convert("I")
    if image.mode in ("I", "F"):
        image = image.point(lambda value: value / 256).convert("L")
    if image.mode == "1":
        return image.convert("L")
    if image.mode == "P":
        image = image.convert("RGBA")
    if image.mode in ("RGBA", "LA", "PA", "RGBa", "La"):
        alpha = image.getchannel("A")
        colour = image.convert("RGB")
        background = Image.new("RGB", image.size, WHITE)
        background.paste(colour, mask=alpha)
        return background
    if image.mode == "L":
        return image
    return image.convert("RGB")


def points(pixels: int) -> float:
    """A side in points at `SCALE`, cut (never rounded up) to four decimals: a renderer at 216 DPI rounds a side up, so
    the page renders at the picture's own pixels."""
    return math.floor(pixels / SCALE * 10_000) / 10_000


def capped(width: int, height: int) -> tuple[int, int]:
    """The size a picture keeps: shrunk, never enlarged, to at most `MAX_PIXELS` pixels and `MAX_SIDE` a side."""
    factor = min(1.0, math.sqrt(MAX_PIXELS / max(width * height, 1)), MAX_SIDE / max(width, height, 1))
    if factor >= 1.0:
        return width, height
    return max(1, math.floor(width * factor)), max(1, math.floor(height * factor))


def prepared(frame: Image.Image) -> Image.Image:
    """A decoded frame upright, flattened and capped."""
    upright = ImageOps.exif_transpose(frame)
    image = flattened(upright if upright is not None else frame)
    size = capped(image.width, image.height)
    if size != image.size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return image


def encoded(image: Image.Image) -> Page:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=QUALITY)
    return Page(buffer.getvalue(), image.width, image.height, image.mode == "L")


def _frame_count(image: Image.Image) -> int:
    """The pages an image gives: every frame of a TIFF, the primary picture of anything else."""
    if image.format != "TIFF":
        return 1
    return max(1, int(getattr(image, "n_frames", 1)))


def pages(source: Path) -> Iterator[Page]:
    """The pages of one image, in order: one per frame of a TIFF, else one; an unreadable one where it fails."""
    try:
        image = Image.open(source)
    except OSError, ValueError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning:
        yield UNREADABLE
        return
    with image:
        try:
            count = _frame_count(image)
        except OSError, ValueError, SyntaxError, EOFError:
            count = 1
        for index in range(count):
            try:
                image.seek(index)
                if image.width * image.height > MAX_DECODED_PIXELS:
                    # Pillow checks a decompression bomb when it opens a file, never at a TIFF's later frames.
                    yield UNREADABLE
                    continue
                if image.width * image.height > MAX_PIXELS:
                    # A JPEG decodes straight to a smaller size (1/2, 1/4 or 1/8); others decode whole.
                    image.draft(None, capped(image.width, image.height))
                page = encoded(prepared(image))
            except (
                OSError,
                ValueError,
                SyntaxError,
                EOFError,
                MemoryError,
                Image.DecompressionBombError,
                Image.DecompressionBombWarning,
            ):
                page = UNREADABLE
            yield page


class Assembler:
    """Writes pages, in order, into batches of `BATCH_PAGES` and then the original made of them."""

    def __init__(self, out: Path) -> None:
        import pikepdf  # noqa: PLC0415 - only the assembling children need it

        self._pikepdf = pikepdf
        self.out = out
        self.total = 0
        self.unreadable: list[int] = []
        self.batches: list[dict[str, Any]] = []
        self._current: Any = None
        self._first = 1
        self._done: list[Path] = []

    @property
    def full(self) -> bool:
        """Whether `MAX_PAGES` are written: nothing more is added."""
        return self.total >= MAX_PAGES

    def add(self, page: Page, done: Path | None = None) -> bool:
        """Adds one page; False once `MAX_PAGES` are written (it is left out). `done` is removed with its batch."""
        if self.full:
            return False
        pikepdf = self._pikepdf
        if self._current is None:
            self._current = pikepdf.Pdf.new()
            self._first = self.total + 1
        pdf = self._current
        self.total += 1
        if page.jpeg is None:
            self.unreadable.append(self.total)
            width, height = BLANK
            pdf.pages.append(
                pikepdf.Page(pikepdf.Dictionary(Type=pikepdf.Name.Page, MediaBox=pikepdf.Array([0, 0, width, height])))
            )
        else:
            width, height = points(page.width), points(page.height)
            picture = pdf.make_stream(page.jpeg)
            picture.Type = pikepdf.Name.XObject
            picture.Subtype = pikepdf.Name.Image
            picture.Width = page.width
            picture.Height = page.height
            picture.ColorSpace = pikepdf.Name.DeviceGray if page.gray else pikepdf.Name.DeviceRGB
            picture.BitsPerComponent = 8
            picture.Filter = pikepdf.Name.DCTDecode
            content = pdf.make_stream(f"q {width} 0 0 {height} 0 0 cm /Im0 Do Q".encode("ascii"))
            pdf.pages.append(
                pikepdf.Page(
                    pikepdf.Dictionary(
                        Type=pikepdf.Name.Page,
                        MediaBox=pikepdf.Array([0, 0, width, height]),
                        Resources=pikepdf.Dictionary(XObject=pikepdf.Dictionary(Im0=picture)),
                        Contents=content,
                    )
                )
            )
        if done is not None:
            self._done.append(done)
        if self.total - self._first + 1 == BATCH_PAGES:
            self._flush()
        return True

    def _flush(self) -> None:
        if self._current is None:
            return
        name = f"batch-{self._first:05d}.pdf"
        self.out.mkdir(mode=0o700, exist_ok=True)
        self._current.save(self.out / name, encryption=False)
        self._current.close()
        self._current = None
        self.batches.append({"file": name, "firstPage": self._first, "lastPage": self.total})
        for path in self._done:
            path.unlink(missing_ok=True)
        self._done = []

    def finish(self) -> None:
        """The last batch, the original (every batch's pages, in order) and the manifest."""
        self._flush()
        pikepdf = self._pikepdf
        parts = [pikepdf.Pdf.open(self.out / batch["file"]) for batch in self.batches]
        try:
            with pikepdf.Pdf.new() as original:
                for part in parts:
                    original.pages.extend(part.pages)
                original.save(self.out / ORIGINAL, encryption=False)
        finally:
            for part in parts:
                part.close()
        manifest = {
            "pageTotal": self.total,
            "original": ORIGINAL,
            "batches": self.batches,
            "unreadable": self.unreadable,
        }
        (self.out / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")


def run(out: Path, sources: Iterable[Path]) -> int:
    """Every image's pages, in order, into the PDF; the exit status."""
    assembler = Assembler(out)
    for source in sources:
        if assembler.full:
            break
        for page in pages(source):
            if not assembler.add(page):
                break
    if assembler.total == 0:
        return 2
    assembler.finish()
    return 0


def decode(source: Path, out: Path) -> int:
    """One image's pages as JPEG files and `pages.json`; the exit status."""
    out.mkdir(mode=0o700, exist_ok=True)
    listed: list[dict[str, Any]] = []
    for index, page in enumerate(pages(source), start=1):
        if index > MAX_PAGES:
            break
        if page.jpeg is None:
            listed.append({"unreadable": True})
            continue
        name = f"page-{index:05d}.jpg"
        (out / name).write_bytes(page.jpeg)
        listed.append({"file": name, "width": page.width, "height": page.height, "gray": page.gray})
    (out / PAGES).write_text(json.dumps({"pages": listed}), encoding="utf-8")
    return 0


def _side(value: object) -> TypeIs[int]:
    return type(value) is int and 1 <= value <= MAX_SIDE


def decoded(folder: Path) -> list[tuple[Page, Path | None]]:
    """A decoding child's pages, checked: without a valid `pages.json` (its child crashed or ran out of time), one
    unreadable page; an entry that is not a plain JPEG file of that folder with a valid size, an unreadable page."""
    try:
        listed = json.loads((folder / PAGES).read_text(encoding="utf-8"))["pages"]
        if not isinstance(listed, list) or not listed:
            raise ValueError(PAGES)
    except OSError, ValueError, KeyError, TypeError:
        return [(UNREADABLE, None)]
    found: list[tuple[Page, Path | None]] = []
    for item in listed:
        if not isinstance(item, dict) or item.get("unreadable") is True:
            found.append((UNREADABLE, None))
            continue
        name, width, height, gray = item.get("file"), item.get("width"), item.get("height"), item.get("gray")
        path = folder / name if isinstance(name, str) and _PAGE_FILE.fullmatch(name) else None
        if path is None or path.is_symlink() or not path.is_file() or not _side(width) or not _side(height):
            found.append((UNREADABLE, None))
            continue
        found.append((Page(path.read_bytes(), width, height, gray is True), path))
    return found


def assemble(out: Path, folders: Iterable[Path]) -> int:
    """The PDF from decoding children's folders, in order; the exit status."""
    assembler = Assembler(out)
    for folder in folders:
        if assembler.full:
            break
        for page, path in decoded(folder):
            if not assembler.add(page, path):
                break
    if assembler.total == 0:
        return 2
    assembler.finish()
    return 0


def main(argv: list[str]) -> int:
    if len(argv) == 4 and argv[1] == "--decode":
        return decode(Path(argv[2]), Path(argv[3]))
    if len(argv) >= 4 and argv[1] == "--assemble":
        return assemble(Path(argv[2]), [Path(folder) for folder in argv[3:]])
    if len(argv) >= 3 and not argv[1].startswith("--"):
        return run(Path(argv[1]), [Path(source) for source in argv[2:]])
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
