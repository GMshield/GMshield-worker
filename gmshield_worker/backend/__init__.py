"""The backend's internal API: the only service the worker calls.

`generated/` is the client and its models, generated from `contracts/internal.json` by
`tools/generate_client.py` (git-ignored, never edited). `client.py` builds one client per
request; `contract.py` holds the minimum contract version.
"""
