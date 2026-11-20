"""The clean stage's child process (Story 1.15): make an uploaded PDF safe and split it into page batches.

Run only through the sandbox (`sandbox.py`), as a script in isolated mode, inside the step folder:

    python -I -B clean.py SOURCE OUT_DIR
    python -I -B clean.py bookmarks ORIGINAL OUT_FILE

It imports pikepdf and the standard library only (never `gmshield_worker`: the isolated child has no path to it), holds
no credential, reaches no network, and writes only under OUT_DIR:

- `original.pdf`: the whole file, cleaned and saved without encryption;
- `batch-NNNNN.pdf`: every `BATCH_PAGES` pages of the cleaned file as a PDF of its own;
- `manifest.json`: `{"pageTotal": N, "original": "original.pdf", "batches": [{"file", "firstPage", "lastPage"}]}`.

Cleaning removes JavaScript (document, page, form and annotation actions, and the name tree), embedded and attached
files (and associated files), `OpenAction` and every additional action (`AA`), every action other than GoTo and URI,
a URI action whose scheme (read after dropping control characters and whitespace) is not http, https or mailto, XFA
forms, and RichMedia, Movie, Sound, Screen, 3D and file attachment annotations.
qpdf writes only what is still reachable, so whatever was cut off is not kept. The exit status says how it went:
0 done, `EXIT_PASSWORD` (a user password is needed), `EXIT_UNREADABLE` (qpdf cannot read it, or it has no page);
anything else is a crash. Nothing is printed: the parent logs the status only.

The `bookmarks` command (pipeline "7") reads a cleaned original's bookmarks (its outline) into OUT_FILE, a JSON list of
`{"title", "depth", "page"}` in the outline's order: depth from 0, page from 1. A destination is resolved whether it is
explicit, named (`/Dests`, or the `/Names` → `/Dests` name tree) or a GoTo action's. An entry whose page does not
resolve, or whose title is empty, is dropped and its children take its place under its parent; the document-title root
is dropped when it is the only entry at depth 0 and has children. A file without bookmarks, or one whose outline cannot
be read, gives an empty list. At most `MAX_BOOKMARKS` entries are kept, `MAX_BOOKMARK_DEPTH` deep. Batches never hold
the outline.
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

# A URI action is kept only for these schemes (javascript:, file:, smb:, data:, vbscript: and the rest are removed).
SAFE_URI_SCHEMES = frozenset({"http", "https", "mailto"})
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
BOOKMARKS = "bookmarks"
MAX_BOOKMARKS = 10_000
MAX_BOOKMARK_DEPTH = 16
MAX_BOOKMARK_TITLE = 1000


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


def _uri_scheme(uri: object) -> str | None:
    """The URI's scheme in lower case, read after dropping C0 controls and whitespace; None without one."""
    if not isinstance(uri, pikepdf.String):
        return None
    text = "".join(character for character in str(uri) if ord(character) > 0x20 and not character.isspace())
    scheme, colon, _rest = text.partition(":")
    if not colon or not scheme or not scheme[0].isascii() or not scheme[0].isalpha():
        return None
    if not all(character.isascii() and (character.isalnum() or character in "+.-") for character in scheme):
        return None
    return scheme.lower()


def _allowed(action: object) -> bool:
    """A GoTo action, or a URI action to an http, https or mailto address: nothing else is kept."""
    if not isinstance(action, Dictionary):
        return False
    kind = str(action.get("/S"))
    if kind == "/GoTo":
        return True
    return kind == "/URI" and _uri_scheme(action.get("/URI")) in SAFE_URI_SCHEMES


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
        # An entry that is not a dictionary (a null, a dangling reference) is left as it is: never a crash.
        kept = [
            annotation
            for annotation in annotations
            if not (isinstance(annotation, Dictionary) and str(annotation.get("/Subtype")) in REMOVED_ANNOTATIONS)
        ]
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


def _named(pdf: Pdf, key: str) -> object:
    """A named destination: looked up in `/Dests` (PDF 1.1), then in the `/Names` → `/Dests` name tree, whichever kind
    of name it was given as."""
    named = pdf.Root.get("/Dests")
    found = named.get("/" + key) if isinstance(named, Dictionary) else None
    if found is None:
        tree = pdf.Root.get("/Names")
        tree = tree.get("/Dests") if isinstance(tree, Dictionary) else None
        found = pikepdf.NameTree(tree).get(key) if isinstance(tree, Dictionary) else None
    return found


def _page_of(pdf: Pdf, target: object, pages: dict[tuple[int, int], int]) -> int | None:
    """The page (from 1) a destination leads to: explicit, named, or inside a dictionary's `/D`; None when it does not
    resolve."""
    for _hop in range(4):
        if isinstance(target, Dictionary):
            target = target.get("/D")
        elif isinstance(target, Name):
            target = _named(pdf, str(target)[1:])
        elif isinstance(target, pikepdf.String):
            target = _named(pdf, str(target))
        else:
            break
    if not isinstance(target, Array) or len(target) == 0:
        return None
    page: object = target[0]
    if isinstance(page, Dictionary) and page.is_indirect:
        return pages.get(page.objgen)
    if type(page) is int and 0 <= page < len(pages):
        # A page index where a page should be (as for a remote destination): read as from 0.
        return page + 1
    return None


def _resolved(pdf: Pdf, item: Dictionary, pages: dict[tuple[int, int], int]) -> int | None:
    """An entry's page, or None when its destination does not resolve (a broken name tree included)."""
    try:
        return _page_of(pdf, _destination(item), pages)
    except pikepdf.PdfError, ValueError, TypeError, KeyError, AttributeError:
        return None


def _destination(item: Dictionary) -> object:
    if "/Dest" in item:
        return item.get("/Dest")
    action = item.get("/A")
    if isinstance(action, Dictionary) and action.get("/S") == Name.GoTo:
        return action.get("/D")
    return None


def _title(item: Dictionary) -> str:
    value = item.get("/Title")
    if not isinstance(value, pikepdf.String):
        return ""
    return " ".join(str(value).split())[:MAX_BOOKMARK_TITLE]


def bookmarks(pdf: Pdf) -> list[dict[str, object]]:
    """The outline's entries as `{title, depth, page}`, in the outline's order (see the module's docstring)."""
    outlines = pdf.Root.get("/Outlines")
    first = outlines.get("/First") if isinstance(outlines, Dictionary) else None
    if not isinstance(first, Dictionary):
        return []
    pages = {page.obj.objgen: number for number, page in enumerate(pdf.pages, start=1)}
    # The document-title root: the only entry at depth 0, over children (they are the outline's real top). A lone
    # entry with no children is kept.
    lone = not isinstance(first.get("/Next"), Dictionary)
    root = first if lone and isinstance(first.get("/First"), Dictionary) else None
    found: list[dict[str, object]] = []
    seen: set[tuple[int, int]] = set()
    # (item, its raw depth, the depth it takes: its kept ancestors), walked depth first in the outline's order.
    stack: list[tuple[object, int, int]] = [(first, 0, 0)]
    while stack and len(found) < MAX_BOOKMARKS:
        item, raw, depth = stack.pop()
        if not isinstance(item, Dictionary):
            continue
        if item.is_indirect:
            if item.objgen in seen:
                continue
            seen.add(item.objgen)
        stack.append((item.get("/Next"), raw, depth))
        kept = False
        if raw > 0 or root is None:
            name = _title(item)
            page = _resolved(pdf, item, pages) if name else None
            kept = page is not None
            if kept:
                found.append({"title": name, "depth": depth, "page": page})
        if raw + 1 < MAX_BOOKMARK_DEPTH:
            stack.append((item.get("/First"), raw + 1, depth + 1 if kept else depth))
    return found


def read_bookmarks(source: Path, target: Path) -> int:
    """A cleaned original's bookmarks into `target`; the exit status."""
    try:
        with Pdf.open(source) as pdf:
            try:
                found = bookmarks(pdf)
            except pikepdf.PdfError, ValueError, TypeError, KeyError, RecursionError:
                # An outline that cannot be read is no outline: the passages keep their headings' paths.
                found = []
    except pikepdf.PasswordError:
        return EXIT_PASSWORD
    except pikepdf.PdfError:
        return EXIT_UNREADABLE
    target.write_text(json.dumps(found, ensure_ascii=False), encoding="utf-8")
    return 0


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
    if len(argv) == 4 and argv[1] == BOOKMARKS:
        return read_bookmarks(Path(argv[2]), Path(argv[3]))
    if len(argv) != 3:
        return 2
    return run(Path(argv[1]), Path(argv[2]))


if __name__ == "__main__":
    sys.exit(main(sys.argv))
