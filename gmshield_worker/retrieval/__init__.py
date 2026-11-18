"""Retrieval: the index of accepted materials in Qdrant (Story 2.1); searching it comes with Story 2.3.

- `recipe.py`: `INDEX_RECIPE`, which names how this worker builds a point. The worker states it at every claim, every
  index change reports it, and the backend re-indexes a source recorded under another one.
- `embedding.py`: the dense vector, embeddinggemma through Ollama's `/api/embed` over plain httpx, with no silent cut:
  a passage too long for the model is split, and its vector is the normalized mean of its parts.
- `keywords.py`: the sparse keyword vector, its token numbers keyed by the owner's token key from the backend.
- `store.py`: the collection `passages` (created when missing), its points and their removal by owner and source.
- `indexing.py`: the `index.change` and `index.removal` job handlers.

The worker is the only reader and writer of Qdrant. No text, title, name or token string ever reaches Qdrant, Redis or
a log line: a point carries ids and numbers only. No LangChain vector store, loader or splitter, and no `ollama`
package (AD-32).
"""
