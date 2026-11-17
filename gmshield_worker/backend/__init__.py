"""The backend's internal API: the only service the worker calls.

`generated/` is the client and its models, generated from `contracts/internal.json` by
`tools/generate_client.py` (git-ignored, never edited). `client.py` builds one client per
request; `files.py` streams a claimed job's input and output files (Story 1.15); `contract.py` holds
the minimum contract version.
"""
