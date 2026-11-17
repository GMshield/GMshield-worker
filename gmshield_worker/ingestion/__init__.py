"""Ingestion steps (Story 1.15 onward): the worker defines the stages and the page batches; the backend records them.

`step.py` runs one claimed step of the `ingestion` job kind and dispatches on its stage; `clean.py` is the clean
stage's child process (pikepdf), started through `sandbox.py`; `errors.py` holds the codes a step reports. Later
stages (triage, layout or OCR, chunking, page pictures) arrive with Stories 1.16 to 1.19 and 1.21.

`PIPELINE_VERSION` names this worker's stages and batches: every step reports it, and the backend keeps the first
report's, so a chain never mixes the output of two pipelines.
"""

PIPELINE_VERSION = "1"
