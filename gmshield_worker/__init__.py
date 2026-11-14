"""GMshield worker: the process that does the AI work (ingestion, OCR, retrieval, model calls).

It talks only to the backend's internal API, opens no listening port and empties its work
folder at start. `python -m gmshield_worker.main` runs the supervisor.
"""
