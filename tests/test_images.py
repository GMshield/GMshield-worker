"""The images stage's child (Story 1.21), run in-process: the pages of each kind of image, upright and flattened, a
picture past the pixel cap, an image that does not decode, the batches and the original, and a retry's decoding and
assembling children. `tests/test_ingestion_step.py` runs it through the sandbox, as the step does."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pikepdf
import pytest
from PIL import Image, ImageDraw

from gmshield_worker.ingestion import images

HEIC = Path(__file__).parent / "image" / "two-colours.heic"


def write(path: Path, picture: Image.Image, fmt: str, **options: Any) -> Path:
    picture.save(path, format=fmt, **options)
    return path


def page_sizes(pdf_path: Path) -> list[tuple[float, float]]:
    with pikepdf.Pdf.open(pdf_path) as pdf:
        return [(float(page.mediabox[2]), float(page.mediabox[3])) for page in pdf.pages]


def manifest(out: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((out / images.MANIFEST).read_text(encoding="utf-8"))
    return loaded


def test_each_image_is_a_page_at_216_dpi_in_order_and_one_that_does_not_decode_is_a_blank_unreadable_page(
    tmp_path: Path,
) -> None:
    sources = [
        write(tmp_path / "a.png", Image.new("RGB", (30, 60), (10, 20, 30)), "PNG"),
        tmp_path / "broken.jpg",
        HEIC,
        write(tmp_path / "c.bmp", Image.new("RGB", (9, 6), (1, 2, 3)), "BMP"),
    ]
    sources[1].write_bytes(b"\xff\xd8\xff\xe0" + bytes(100))
    out = tmp_path / "out"

    assert images.run(out, sources) == 0

    body = manifest(out)
    assert body == {
        "pageTotal": 4,
        "original": "original.pdf",
        "batches": [{"file": "batch-00001.pdf", "firstPage": 1, "lastPage": 4}],
        "unreadable": [2],
    }
    assert page_sizes(out / "original.pdf") == [(10.0, 20.0), images.BLANK, (32.0, 21.3333), (3.0, 2.0)]
    assert page_sizes(out / "batch-00001.pdf") == page_sizes(out / "original.pdf")
    with pikepdf.Pdf.open(out / "original.pdf") as pdf:
        picture = pdf.pages[0].Resources.XObject.Im0
        assert picture.Filter == pikepdf.Name.DCTDecode
        assert (int(picture.Width), int(picture.Height)) == (30, 60)
        assert "/Contents" not in pdf.pages[1].obj
        assert not pdf.is_encrypted


def test_pages_are_cut_into_batches_of_twenty_and_the_original_holds_them_all(tmp_path: Path) -> None:
    frames = [Image.new("L", (8, 8), shade) for shade in range(45)]
    tiff = write(tmp_path / "many.tif", frames[0], "TIFF", save_all=True, append_images=frames[1:])
    out = tmp_path / "out"

    assert images.run(out, [tiff]) == 0

    body = manifest(out)
    assert body["pageTotal"] == 45
    assert [(batch["firstPage"], batch["lastPage"]) for batch in body["batches"]] == [(1, 20), (21, 40), (41, 45)]
    assert len(page_sizes(out / "original.pdf")) == 45
    with pikepdf.Pdf.open(out / "batch-00041.pdf") as pdf:
        assert len(pdf.pages) == 5
        assert pdf.pages[0].Resources.XObject.Im0.ColorSpace == pikepdf.Name.DeviceGray


def test_a_picture_is_turned_upright_by_its_exif_orientation_and_laid_on_white(tmp_path: Path) -> None:
    upright = Image.new("RGB", (20, 40), (255, 255, 255))
    ImageDraw.Draw(upright).rectangle([0, 0, 9, 9], fill=(255, 0, 0))
    exif = Image.Exif()
    exif[0x0112] = 8  # turn 90 degrees counter-clockwise to show it
    lying = write(tmp_path / "lying.jpg", upright.rotate(-90, expand=True), "JPEG", quality=95, exif=exif.tobytes())
    see_through = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    pages = [*images.pages(lying), *images.pages(write(tmp_path / "clear.png", see_through, "PNG"))]

    assert [(page.width, page.height) for page in pages] == [(20, 40), (10, 10)]
    shown = [Image.open(io.BytesIO(page.jpeg or b"")).convert("RGB") for page in pages]
    red = shown[0].getpixel((3, 3))
    assert isinstance(red, tuple)
    assert red[0] > 200 and red[1] < 60 and red[2] < 60
    assert shown[0].getpixel((15, 35)) == pytest.approx((255, 255, 255), abs=8)
    assert shown[1].getpixel((5, 5)) == pytest.approx((255, 255, 255), abs=2)


@pytest.mark.parametrize(
    ("mode", "colour", "gray"),
    [("1", 1, True), ("L", 90, True), ("I;16", 65535, True), ("F", 25600.0, True), ("P", 3, False),
     ("LA", (40, 255), False), ("CMYK", (0, 0, 0, 0), False)],
)  # fmt: skip
def test_every_mode_is_flattened_to_grey_or_rgb(mode: str, colour: Any, gray: bool) -> None:
    flat = images.flattened(Image.new(mode, (4, 4), colour))
    assert flat.mode == ("L" if gray else "RGB")


def test_a_picture_past_the_pixel_cap_is_shrunk_and_never_enlarged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert images.capped(4000, 3000) == (4000, 3000)
    assert images.capped(8000, 6000) == (5656, 4242)
    assert images.capped(20_000, 100) == (10_000, 50)
    monkeypatch.setattr(images, "MAX_PIXELS", 100)
    big = write(tmp_path / "big.jpg", Image.new("RGB", (40, 40), (0, 0, 255)), "JPEG")
    [page] = list(images.pages(big))
    assert (page.width, page.height) == (10, 10)


def test_a_decompression_bomb_or_a_frame_that_fails_is_an_unreadable_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 50)
    bomb = write(tmp_path / "bomb.png", Image.new("RGB", (20, 20)), "PNG")
    assert list(images.pages(bomb)) == [images.UNREADABLE]
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", images.MAX_DECODED_PIXELS)
    assert list(images.pages(tmp_path / "missing.heic")) == [images.UNREADABLE]
    frames = [Image.new("L", (8, 8), 0), Image.new("L", (8, 8), 255)]
    tiff = write(tmp_path / "two.tif", frames[0], "TIFF", save_all=True, append_images=frames[1:])

    def failing(image: Image.Image) -> Image.Image:
        if image.tell() == 1:
            raise OSError("frame")
        return image.convert("L")

    monkeypatch.setattr(images, "prepared", failing)
    found = list(images.pages(tiff))
    assert found[1] == images.UNREADABLE
    assert found[0].jpeg is not None


def test_pages_past_the_most_a_set_gives_are_left_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(images, "MAX_PAGES", 2)
    sources = [write(tmp_path / f"{index}.png", Image.new("L", (4, 4)), "PNG") for index in range(3)]
    assert images.run(tmp_path / "out", sources) == 0
    assert manifest(tmp_path / "out")["pageTotal"] == 2
    assert images.decode(sources[0], tmp_path / "one") == 0
    assert images.run(tmp_path / "none", []) == 2


def test_a_retry_decodes_each_image_alone_then_assembles_and_a_crashed_decoding_is_one_unreadable_page(
    tmp_path: Path,
) -> None:
    frames = [Image.new("L", (6, 4), 0), Image.new("L", (6, 4), 200)]
    tiff = write(tmp_path / "two.tif", frames[0], "TIFF", save_all=True, append_images=frames[1:])
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(50))
    for index, source in enumerate([tiff, broken, HEIC], start=1):
        assert images.main(["images.py", "--decode", str(source), str(tmp_path / f"decoded-{index}")]) == 0
    listed = json.loads((tmp_path / "decoded-1" / images.PAGES).read_text(encoding="utf-8"))["pages"]
    assert listed == [
        {"file": "page-00001.jpg", "width": 6, "height": 4, "gray": True},
        {"file": "page-00002.jpg", "width": 6, "height": 4, "gray": True},
    ]
    assert json.loads((tmp_path / "decoded-2" / images.PAGES).read_text(encoding="utf-8")) == {
        "pages": [{"unreadable": True}]
    }
    (tmp_path / "decoded-4").mkdir()  # its child crashed before writing anything
    folders = [str(tmp_path / f"decoded-{index}") for index in (1, 2, 4, 3)]

    assert images.main(["images.py", "--assemble", str(tmp_path / "out"), *folders]) == 0

    body = manifest(tmp_path / "out")
    assert body["pageTotal"] == 5
    assert body["unreadable"] == [3, 4]
    assert page_sizes(tmp_path / "out" / "original.pdf")[4] == (32.0, 21.3333)
    # Each JPEG goes once its batch is written.
    assert not list((tmp_path / "decoded-1").glob("*.jpg"))


def test_a_decoded_list_that_is_not_ours_is_read_as_unreadable_pages(tmp_path: Path) -> None:
    folder = tmp_path / "decoded"
    folder.mkdir()
    (folder / "page-00001.jpg").write_bytes(b"\xff\xd8\xff")
    (folder / "link.jpg").symlink_to(folder / "page-00001.jpg")
    entries: list[object] = [
        {"file": "page-00001.jpg", "width": 1, "height": 1, "gray": False},
        {"file": "../page-00001.jpg", "width": 1, "height": 1},
        {"file": "link.jpg", "width": 1, "height": 1},
        {"file": "page-00002.jpg", "width": 1, "height": 1},
        {"file": "page-00001.jpg", "width": 0, "height": 1},
        {"file": "page-00001.jpg", "width": 1, "height": True},
        "page",
    ]
    (folder / images.PAGES).write_text(json.dumps({"pages": entries}), encoding="utf-8")
    found = images.decoded(folder)
    assert [page.jpeg is not None for page, _path in found] == [True, False, False, False, False, False, False]
    bodies: list[object] = [{"pages": []}, {"pages": "x"}, {"other": 1}, []]
    for body in bodies:
        (folder / images.PAGES).write_text(json.dumps(body), encoding="utf-8")
        assert images.decoded(folder) == [(images.UNREADABLE, None)]


def test_the_command_line_takes_one_of_its_three_forms() -> None:
    for argv in (
        ["images.py"],
        ["images.py", "out"],
        ["images.py", "--decode", "a"],
        ["images.py", "--other", "a", "b"],
    ):
        assert images.main(argv) == 2


def test_a_tiff_frame_past_the_decoded_pixel_cap_is_an_unreadable_page_never_decoded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = [Image.new("L", (8, 8), 0), Image.new("L", (20, 20), 255), Image.new("L", (8, 8), 128)]
    tiff = write(tmp_path / "bomb.tif", frames[0], "TIFF", save_all=True, append_images=frames[1:])
    monkeypatch.setattr(images, "MAX_DECODED_PIXELS", 100)
    decoded: list[tuple[int, int]] = []
    real = images.prepared

    def recording(frame: Image.Image) -> Image.Image:
        decoded.append(frame.size)
        return real(frame)

    monkeypatch.setattr(images, "prepared", recording)
    found = list(images.pages(tiff))
    assert [page == images.UNREADABLE for page in found] == [False, True, False]
    assert decoded == [(8, 8), (8, 8)]


def test_once_the_most_pages_are_written_no_other_image_or_folder_is_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(images, "MAX_PAGES", 2)
    sources = [write(tmp_path / f"{index}.png", Image.new("L", (4, 4)), "PNG") for index in range(4)]
    opened: list[Path] = []
    real_pages, real_decoded = images.pages, images.decoded

    def pages(source: Path) -> Any:
        opened.append(source)
        return real_pages(source)

    def decoded(folder: Path) -> Any:
        opened.append(folder)
        return real_decoded(folder)

    monkeypatch.setattr(images, "pages", pages)
    monkeypatch.setattr(images, "decoded", decoded)
    assert images.run(tmp_path / "out", sources) == 0
    assert opened == sources[:2]
    opened.clear()
    folders = [tmp_path / f"decoded-{index}" for index in range(4)]
    assert images.assemble(tmp_path / "again", folders) == 0
    assert opened == folders[:2]
