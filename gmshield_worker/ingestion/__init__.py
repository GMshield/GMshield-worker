"""Ingestion steps (Story 1.15 onward): the worker defines the stages and the page batches; the backend records them.

`step.py` runs one claimed step of the `ingestion` job kind and dispatches on its stage. Each stage's child process
is a script of its own, started through `sandbox.py`: `clean.py` (pikepdf), `triage.py` (pypdfium2), `layout.py`
(Docling) and, for a text upload, `text.py` (Story 1.16). `errors.py` holds the codes a step reports. Later stages
(OCR, chunking, page pictures) arrive with Stories 1.17 to 1.19 and 1.21.

`PIPELINE_VERSION` names this worker's stages and batches: every step reports it, and the backend keeps the first
report's, so a chain never mixes the output of two pipelines. A step of a chain started under another version fails
with `ingestion.pipeline_changed`, and the backend restarts the chain. "2" (Story 1.16): clean, then triage per batch,
then layout per run of text pages; or one text step.
"""

PIPELINE_VERSION = "2"
