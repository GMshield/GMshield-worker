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

"2": as "1", but each keyword token is valued by BM25's term weight, `tf·(k1+1) / (tf + k1·(1 - b + b·L/AVG))`, with
k1 = 1.2, b = 0.75, L the passage's token count and AVG = 250 (`keywords.KEYWORD_AVG_TOKENS`, a fixed,
corpus-independent average), instead of its count; Qdrant's `idf` modifier stays, and the token numbers are unchanged.
While "2" re-indexes after a deploy, the points still under "1" carry raw counts. The passages themselves change only
for new uploads (pipeline "7"): an accepted version keeps its pipeline-"6" passages until it is uploaded again.

Records under "2" (Story 4.2 part B): a record's point is embedded as `title: {name} | text: {lines}`, `{lines}` its
fields as `name: value` lines joined by line feeds (its name alone when it has no field), name and lines cut together
to 16,384 code points, name first; its keyword vector is computed as a passage's, over the name and the lines (the
name alone without fields); its payload adds `record_type`. This adds a format without changing how a passage is
embedded, and no version accepted before it has records (records are extracted from pipeline "8", first released with
it), so no point is stale: "2" stays. Any later change to the record format bumps the recipe like any other.

A question (Story 2.3) is embedded under embeddinggemma's query prompt, `task: search result | query: {question}`, with
the same model and `truncate: false`, and its keyword vector has the same token numbers, each valued 1
(`keywords.query_vector`). The query prompt is part of the
search, not of the index: changing it needs no re-index and does not bump the recipe (re-record the test vectors with
`tools/record_embeddings.py` instead).
"""

from __future__ import annotations

INDEX_RECIPE = "2"
