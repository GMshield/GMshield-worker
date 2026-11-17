"""PDFs built for the tests with pikepdf (Story 1.15): active content of every kind the clean stage removes, owner
and user passwords, a file that is not a PDF inside, many linked pages, and a large image PDF written as a stream so
that the test itself never holds it in memory."""

from __future__ import annotations

from pathlib import Path

import pikepdf
from pikepdf import Array, Dictionary, Name, Pdf, String

# Markers the cleaned files must not hold anywhere, in any stream or string.
JS_MARKER = "MARKER-JAVASCRIPT"
EMBEDDED_MARKER = b"MARKER-EMBEDDED-FILE"
LAUNCH_MARKER = "MARKER-LAUNCH"
XFA_MARKER = b"MARKER-XFA"
RICH_MEDIA_MARKER = b"MARKER-RICH-MEDIA"


def _page_text(pdf: Pdf, text: str) -> pikepdf.Stream:
    return pikepdf.Stream(pdf, f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode("ascii"))


def pages(count: int) -> Pdf:
    """`count` pages, each with its number as text in its content stream (`PAGE-0001`...)."""
    pdf = Pdf.new()
    font = pdf.make_indirect(Dictionary(Type=Name.Font, Subtype=Name.Type1, BaseFont=Name.Helvetica))
    for number in range(1, count + 1):
        pdf.add_blank_page(page_size=(200, 200))
        page = pdf.pages[-1]
        page.obj.Resources = Dictionary(Font=Dictionary(F1=font))
        page.obj.Contents = pdf.make_indirect(_page_text(pdf, f"PAGE-{number:04d}"))
    return pdf


def _javascript(pdf: Pdf, text: str) -> Dictionary:
    return pdf.make_indirect(Dictionary(Type=Name.Action, S=Name.JavaScript, JS=String(f"app.alert('{text}')")))


def _link(pdf: Pdf, action: Dictionary) -> Dictionary:
    return pdf.make_indirect(
        Dictionary(Type=Name.Annot, Subtype=Name.Link, Rect=Array([0, 0, 50, 50]), Border=Array([0, 0, 0]), A=action)
    )


def _attachment(pdf: Pdf) -> pikepdf.AttachedFileSpec:
    return pikepdf.AttachedFileSpec(
        pdf,
        EMBEDDED_MARKER,
        description="",
        filename="notes.txt",
        mime_type="text/plain",
        creation_date="",
        mod_date="",
    )


def active(path: Path, page_count: int = 3) -> Path:
    """Every kind of active content the clean stage removes, beside a GoTo and a URI link it keeps."""
    pdf = pages(page_count)
    script = _javascript(pdf, JS_MARKER)
    pdf.Root.OpenAction = script
    pdf.Root.AA = Dictionary(WC=_javascript(pdf, JS_MARKER))
    pdf.Root.Names = Dictionary(JavaScript=Dictionary(Names=Array([String("init"), script])))
    pdf.attachments["notes.txt"] = _attachment(pdf)
    xfa = pikepdf.Stream(pdf, b"<xdp>" + XFA_MARKER + b"</xdp>")
    pdf.Root.AcroForm = Dictionary(Fields=Array([]), XFA=pdf.make_indirect(xfa), NeedsRendering=True)
    first = pdf.pages[0].obj
    first.AA = Dictionary(O=_javascript(pdf, JS_MARKER))
    goto = Dictionary(S=Name.GoTo, D=Array([pdf.pages[-1].obj, Name.Fit]))
    uri = Dictionary(S=Name.URI, URI=String("https://example.test/rules"))
    launch = Dictionary(S=Name.Launch, F=String(LAUNCH_MARKER))
    chained = Dictionary(S=Name.URI, URI=String("https://example.test/next"), Next=_javascript(pdf, JS_MARKER))
    javascript_uri = Dictionary(S=Name.URI, URI=String(f"javascript:app.alert('{JS_MARKER}')"))
    media = pikepdf.Stream(pdf, RICH_MEDIA_MARKER)
    attachment = _attachment(pdf)
    first.Annots = Array(
        [
            _link(pdf, goto),
            _link(pdf, uri),
            _link(pdf, launch),
            _link(pdf, chained),
            _link(pdf, javascript_uri),
            _link(pdf, _javascript(pdf, JS_MARKER)),
            pdf.make_indirect(
                Dictionary(
                    Type=Name.Annot,
                    Subtype=Name.RichMedia,
                    Rect=Array([0, 0, 10, 10]),
                    RichMediaContent=Dictionary(Assets=Dictionary(Names=Array([String("m"), media]))),
                )
            ),
            pdf.make_indirect(
                Dictionary(Type=Name.Annot, Subtype=Name.FileAttachment, Rect=Array([0, 0, 10, 10]), FS=attachment.obj)
            ),
            pdf.make_indirect(Dictionary(Type=Name.Annot, Subtype=Name.Screen, Rect=Array([0, 0, 10, 10]))),
            pdf.make_indirect(Dictionary(Type=Name.Annot, Subtype=Name("/3D"), Rect=Array([0, 0, 10, 10]))),
            pdf.make_indirect(Dictionary(Type=Name.Annot, Subtype=Name.Movie, Rect=Array([0, 0, 10, 10]))),
            pdf.make_indirect(Dictionary(Type=Name.Annot, Subtype=Name.Sound, Rect=Array([0, 0, 10, 10]))),
            pdf.make_indirect(
                Dictionary(Type=Name.Annot, Subtype=Name.Widget, Rect=Array([0, 0, 10, 10]), AA=Dictionary(
                    K=_javascript(pdf, JS_MARKER)))
            ),
        ]
    )  # fmt: skip
    pdf.save(path)
    return path


UNSAFE_URIS = (
    "javascript:app.alert(1)",
    "\x01javascript:app.alert(1)",
    "java\tscript:app.alert(1)",
    " JavaScript :app.alert(1)",
    "file:///etc/passwd",
    "smb://host/share",
    "data:text/html,<script>alert(1)</script>",
    "vbscript:msgbox(1)",
    "relative/page.html",
    "",
)
SAFE_URIS = ("https://example.test/a", "http://example.test/b", "mailto:gm@example.test", " HTTPS://example.test/c")


def uris(path: Path) -> Path:
    """One page with a URI link for each unsafe and each safe address, a null and an integer in its /Annots."""
    pdf = pages(1)
    links = [_link(pdf, Dictionary(S=Name.URI, URI=String(uri))) for uri in (*UNSAFE_URIS, *SAFE_URIS)]
    pdf.pages[0].obj.Annots = Array([*links, None, 7])
    pdf.save(path)
    return path


def encrypted(path: Path, *, user: str) -> Path:
    """Three pages under an owner password, and a user password when `user` is not empty."""
    pages(3).save(path, encryption=pikepdf.Encryption(owner="owner-secret", user=user, R=6))
    return path


def broken(path: Path) -> Path:
    """A PDF header, then bytes that are not a PDF."""
    path.write_bytes(b"%PDF-1.7\n" + bytes(range(256)) * 64)
    return path


def linked(path: Path, page_count: int) -> Path:
    """`page_count` pages; the first links to the last, and the last back to the first."""
    pdf = pages(page_count)
    first, last = pdf.pages[0].obj, pdf.pages[-1].obj
    first.Annots = Array([_link(pdf, Dictionary(S=Name.GoTo, D=Array([last, Name.Fit])))])
    last.Annots = Array([_link(pdf, Dictionary(S=Name.GoTo, D=Array([first, Name.Fit])))])
    pdf.save(path)
    return path


def large(path: Path, *, pages_count: int, image_bytes: int) -> Path:
    """`pages_count` pages, each drawing an image of `image_bytes` bytes, written as it goes: the test never holds the
    file. The images are opaque DCT data (never decoded by the clean stage), so qpdf copies them as they are."""
    block = (b"\xff\xd8" + bytes(range(256)) * 4096)[: 1024 * 1024]
    offsets: list[int] = []
    with path.open("wb") as out:

        def start_object(number: int) -> None:
            offsets.append(out.tell())
            out.write(f"{number} 0 obj\n".encode("ascii"))

        out.write(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
        start_object(1)
        out.write(b"<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
        kids = " ".join(f"{3 + index * 3} 0 R" for index in range(pages_count))
        start_object(2)
        out.write(f"<< /Type /Pages /Count {pages_count} /Kids [{kids}] >>\nendobj\n".encode("ascii"))
        for index in range(pages_count):
            page, content, image = 3 + index * 3, 4 + index * 3, 5 + index * 3
            start_object(page)
            out.write(
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content} 0 R"
                f" /Resources << /XObject << /Im0 {image} 0 R >> >> >>\nendobj\n".encode("ascii")
            )
            drawing = b"q 612 0 0 792 0 0 cm /Im0 Do Q"
            start_object(content)
            out.write(f"<< /Length {len(drawing)} >>\nstream\n".encode("ascii") + drawing + b"\nendstream\nendobj\n")
            start_object(image)
            out.write(
                f"<< /Type /XObject /Subtype /Image /Width 4096 /Height 4096 /ColorSpace /DeviceGray"
                f" /BitsPerComponent 8 /Filter /DCTDecode /Length {image_bytes} >>\nstream\n".encode("ascii")
            )
            written = 0
            while written < image_bytes:
                chunk = block[: image_bytes - written]
                out.write(chunk)
                written += len(chunk)
            out.write(b"\nendstream\nendobj\n")
        xref = out.tell()
        out.write(f"xref\n0 {len(offsets) + 1}\n0000000000 65535 f \n".encode("ascii"))
        for offset in offsets:
            out.write(f"{offset:010d} 00000 n \n".encode("ascii"))
        out.write(f"trailer\n<< /Size {len(offsets) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii"))
    return path


def everything(path: Path) -> bytes:
    """Every string, name and decoded stream of a PDF, as one byte string to search for markers."""
    found: list[bytes] = []
    with Pdf.open(path) as pdf:
        for item in pdf.objects:
            if isinstance(item, pikepdf.Stream):
                try:
                    found.append(item.read_bytes())
                except pikepdf.PdfError:
                    found.append(item.read_raw_bytes())
            found.append(repr(item).encode("utf-8", "replace"))
    return b"\n".join(found)


def with_text(path: Path, texts: list[str]) -> Path:
    """One page per entry of `texts`, each drawing that text (an empty one draws nothing: a page with no text layer,
    as a scan has)."""
    pdf = Pdf.new()
    font = pdf.make_indirect(Dictionary(Type=Name.Font, Subtype=Name.Type1, BaseFont=Name.Helvetica))
    for text in texts:
        pdf.add_blank_page(page_size=(300, 300))
        page = pdf.pages[-1]
        page.obj.Resources = Dictionary(Font=Dictionary(F1=font))
        page.obj.Contents = pdf.make_indirect(_page_text(pdf, text) if text else pikepdf.Stream(pdf, b""))
    pdf.save(path)
    return path
