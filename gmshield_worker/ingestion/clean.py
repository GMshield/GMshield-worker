"""The clean stage's child process (Story 1.15): make an uploaded PDF safe and split it into page batches.

Run only through the sandbox (`sandbox.py`), as a script in isolated mode, inside the step folder:

    python -I -B clean.py SOURCE OUT_DIR

It imports pikepdf and the standard library only (never `gmshield_worker`: the isolated child has no path to it), holds
no credential, reaches no network, and writes only under OUT_DIR:

- `original.pdf`: the whole file, cleaned and saved without encryption;
- `batch-NNNNN.pdf`: every `BATCH_PAGES` pages of the cleaned file as a PDF of its own;
- `manifest.json`: `{"pageTotal": N, "original": "original.pdf", "batches": [{"file", "firstPage", "lastPage"}]}`.

Cleaning removes JavaScript (document, page, form and annotation actions, and the name tree), embedded and attached
files (and associated files), `OpenAction` and every additional action (`AA`), every action other than GoTo and URI
(a `javascript:` URI included), XFA forms, and RichMedia, Movie, Sound, Screen, 3D and file attachment annotations.
qpdf writes only what is still reachable, so whatever was cut off is not kept. The exit status says how it went:
0 done, `EXIT_PASSWORD` (a user password is needed), `EXIT_UNREADABLE` (qpdf cannot read it, or it has no page);
anything else is a crash. Nothing is printed: the parent logs the status only.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path

import pikepdf
from pikepdf import Array, Dictionary, Name, Pdf, Stream

EXIT_PASSWORD = 10
EXIT_UNREADABLE = 11
BATCH_PAGES = 20
MANIFEST = "manifest.json"
ORIGINAL = "original.pdf"

ALLOWED_ACTIONS = frozenset({"/GoTo", "/URI"})
# Every action type of PDF 2.0: a dictionary whose /S is one of them is an action.
ACTION_TYPES = frozenset(
    {
        "/GoTo", "/GoToR", "/GoToE", "/GoToDp", "/Launch", "/Thread", "/URI", "/Sound", "/Movie", "/Hide", "/Named",
        "/SubmitForm", "/ResetForm", "/ImportData", "/JavaScript", "/SetOCGState", "/Rendition", "/Trans",
        "/GoTo3DView", "/RichMediaExecute",
    }
)  # fmt: skip
REMOVED_ANNOTATIONS = frozenset({"/RichMedia", "/Movie", "/Sound", "/Screen", "/3D", "/FileAttachment"})
# Keys removed wherever they appear: actions that run by themselves, scripts, embedded and associated files, XFA.
REMOVED_KEYS = (
    "/AA", "/OpenAction", "/JS", "/JavaScript", "/EmbeddedFiles", "/EF", "/RF", "/AF", "/XFA", "/NeedsRendering",
    "/Collection", "/RichMediaContent", "/RichMediaSettings",
)  # fmt: skip
# What a batch drops from its catalog and its pages: everything that points across the whole document.
BATCH_ROOT_KEYS = ("/Outlines", "/Names", "/Dests", "/StructTreeRoot", "/AcroForm", "/Threads", "/PageLabels")


def _containers(item: object) -> Iterator[Dictionary | Array | Stream]:
    """The direct dictionaries, arrays and streams in `item`, `item` first; indirect ones are visited on their own."""
    stack: list[object] = [item]
    while stack:
        current = stack.pop()
        if isinstance(current, (Dictionary, Stream)):
            yield current
            stack.extend(value for _key, value in current.items() if _direct_container(value))
        elif isinstance(current, Array):
            yield current
            stack.extend(value for value in current if _direct_container(value))


def _direct_container(value: object) -> bool:
    return isinstance(value, (Dictionary, Array, Stream)) and not value.is_indirect


def _is_action(item: Dictionary | Stream) -> bool:
    kind = item.get("/S")
    return item.get("/Type") == Name.Action or (isinstance(kind, Name) and str(kind) in ACTION_TYPES)


def _allowed(action: object) -> bool:
    if not isinstance(action, Dictionary) or str(action.get("/S")) not in ALLOWED_ACTIONS:
        return False
    uri = action.get("/URI")
    return uri is None or not str(uri).strip().lower().startswith("javascript:")


def _clean(item: Dictionary | Stream) -> None:
    for key in REMOVED_KEYS:
        if key in item:
            del item[key]
    if "/A" in item and not _allowed(item.get("/A")):
        del item["/A"]
    if _is_action(item):
        if _allowed(item):
            if "/Next" in item:
                del item["/Next"]
        else:
            for key in list(item.keys()):
                if key != "/Type":
                    del item[key]
    annotations = item.get("/Annots")
    if isinstance(annotations, Array):
        kept = [annotation for annotation in annotations if str(annotation.get("/Subtype")) not in REMOVED_ANNOTATIONS]
        if len(kept) != len(annotations):
            item["/Annots"] = Array(kept)


def clean(pdf: Pdf) -> None:
    """Cleans every object of `pdf` in place."""
    for key in REMOVED_KEYS:
        if key in pdf.Root:
            del pdf.Root[key]
    for item in list(pdf.objects):
        for container in _containers(item):
            if isinstance(container, (Dictionary, Stream)):
                _clean(container)


def _points_outside(target: object, kept: set[tuple[int, int]]) -> bool:
    """Whether a destination is not an explicit one to a page of this batch (a named one never resolves here)."""
    if isinstance(target, Dictionary):
        target = target.get("/D")
    if not isinstance(target, Array) or len(target) == 0:
        return True
    page = target[0]
    return not (isinstance(page, Dictionary) and page.is_indirect and page.objgen in kept)


def write_batch(original: Path, first: int, last: int, target: Path) -> None:
    """Pages `first` to `last` (from 1) of the cleaned original, as a PDF that holds nothing of the other pages."""
    with Pdf.open(original) as part:
        del part.pages[last:]
        del part.pages[: first - 1]
        for key in BATCH_ROOT_KEYS:
            if key in part.Root:
                del part.Root[key]
        kept = {page.obj.objgen for page in part.pages}
        for page in part.pages:
            if "/B" in page.obj:
                del page.obj["/B"]
            annotations = page.obj.get("/Annots")
            if not isinstance(annotations, Array):
                continue
            for annotation in annotations:
                if not isinstance(annotation, Dictionary):
                    continue
                for key in ("/Parent", "/IRT"):
                    if key in annotation:
                        del annotation[key]
                if "/Dest" in annotation and _points_outside(annotation.get("/Dest"), kept):
                    del annotation["/Dest"]
                action = annotation.get("/A")
                if isinstance(action, Dictionary) and action.get("/S") == Name.GoTo and _points_outside(action, kept):
                    del annotation["/A"]
        part.save(target, encryption=False)


def run(source: Path, out: Path) -> int:
    """Cleans and splits `source` into `out`; the exit status."""
    try:
        pdf = Pdf.open(source)
    except pikepdf.PasswordError:
        return EXIT_PASSWORD
    except pikepdf.PdfError:
        return EXIT_UNREADABLE
    try:
        with pdf:
            total = len(pdf.pages)
            if total == 0:
                return EXIT_UNREADABLE
            clean(pdf)
            out.mkdir(mode=0o700, exist_ok=True)
            pdf.save(out / ORIGINAL, encryption=False)
    except pikepdf.PdfError:
        return EXIT_UNREADABLE
    batches = []
    for first in range(1, total + 1, BATCH_PAGES):
        last = min(total, first + BATCH_PAGES - 1)
        name = f"batch-{first:05d}.pdf"
        try:
            write_batch(out / ORIGINAL, first, last, out / name)
        except pikepdf.PdfError:
            return EXIT_UNREADABLE
        batches.append({"file": name, "firstPage": first, "lastPage": last})
    manifest = {"pageTotal": total, "original": ORIGINAL, "batches": batches}
    (out / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        return 2
    return run(Path(argv[1]), Path(argv[2]))


if __name__ == "__main__":
    sys.exit(main(sys.argv))
