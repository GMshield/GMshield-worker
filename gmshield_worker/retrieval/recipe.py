"""The index recipe (Story 2.1, AD-10): how this worker turns a passage into a Qdrant point.

The worker states it on every claim, every `index.change` job reports it, and each source records the one it was
indexed with; when the worker states another one, the backend creates index jobs for every source recorded under
another one. Versioning rule, as for `PIPELINE_VERSION`: any change to the embedding model, the embedding input's
format, the splitting of a passage too long for the model, the keyword tokens or their numbering, or the collection's
vectors, bumps it. A changed payload field that search does not depend on does not.

"1" (Story 2.1): embeddinggemma:300m, `title: {title or "none"} | text: {text}`, `truncate: false`, a passage the
model refuses split in two by `ingestion.passages.split` up to 4 times, the normalized mean of the parts; keyword
tokens by NFKC, casefold, diacritics stripped and Unicode words, numbered by the first 4 bytes of HMAC-SHA256 under the
owner's token key, valued by their count, with Qdrant's `idf` modifier.

A question (Story 2.3) is embedded under embeddinggemma's query prompt, `task: search result | query: {question}`, with
the same model and `truncate: false`, and its keyword vector is built as a passage's. The query prompt is part of the
search, not of the index: changing it needs no re-index and does not bump the recipe (re-record the test vectors with
`tools/record_embeddings.py` instead).
"""

from __future__ import annotations

INDEX_RECIPE = "1"
