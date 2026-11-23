"""Ingestion steps (Story 1.15 onward): the worker defines the stages and the page batches; the backend records them.

`step.py` runs one claimed step of the `ingestion` job kind and dispatches on its stage. Each stage's child process
is a script of its own, started through `sandbox.py`: `clean.py` (pikepdf), `triage.py` (pypdfium2), `layout.py`
(Docling), `ocr.py` (OpenCV, Docling and Tesseract; Story 1.17) and, for a text upload, `text.py` (Story 1.16).
`errors.py` holds the codes a step reports. The `passages` stage (Story 1.18) cuts in-process, since it reads only our
own JSON: `passages.py` cuts every text file of a version into passages and an outline (from pipeline "7", a PDF's
bookmarks are read first from its cleaned original by `clean.py bookmarks`, in the sandbox). From Story
1.19 the reading stages also report a page view per page they read: `layout.py` renders each page with pypdfium2,
`ocr.py` keeps its straightened picture, both as WebP; `text` reports each page's text. From Story 1.21, the first stage
of an image set, `images.py` (Pillow and pi-heif, then pikepdf), assembles its images into one image-only PDF that
the later stages read as any scan.

`PIPELINE_VERSION` names this worker's stages and batches: every step reports it, and the backend keeps the first
report's, so a chain never mixes the output of two pipelines. A step of a chain started under another version fails
with `ingestion.pipeline_changed`, and the backend restarts the chain. "2" (Story 1.16): clean, then triage per batch,
then layout per run of text pages; or one text step. "3" (Story 1.17): triage also sends a page with fewer than 200
characters under a half-page image to OCR, and asks an `ocr` step for each run of at most 10 image pages. "4"
(Story 1.18): clean and text also ask the `passages` step over every text file, which the backend holds until every
page has an outcome. "5" (Story 1.19): layout, OCR and text also report `pageViews` (layout and OCR send each page's
WebP picture back). "6" (Story 1.21): the `images` stage assembles an image set into one PDF, one page per image
(a blank page for one that does not decode, which OCR reports unreadable), and asks triage over every batch. "7": a
PDF's passages step reads the cleaned original's bookmarks, and a section path is the last bookmark's path plus the last
Docling heading under it (Docling's heading levels no longer nest); passages are joined to about `PASSAGE_TARGET`
(1,200) code points, the first of each section up to `PASSAGE_LIMIT`. "8" (Story 4.2): the passages step also asks
a `records` step per run of 10 pages that holds a passage, over the passages of those pages; `records.py` reads their
rules, spells and items through the claim's model, in-process (it reads only the backend's passages). "9" (Story 4.4):
the same call also proposes the record types and terms its passages name or define, each with up to three quotes and a
term with its quoted definition, and a records report carries them as `proposals` (internal contract 1.20.0).
"""

PIPELINE_VERSION = "9"
