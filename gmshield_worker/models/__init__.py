"""The one way to a model (Story 2.10, AD-18). Provider SDKs and LangChain's provider packages are imported here and
nowhere else in the worker (a boundary test fails the build otherwise).

- `providers.py`: the providers (`azure-openai`, `anthropic`, `openai`), the claim's `ModelChoice`, and each
  provider's explicit base URL and key. A provider without a key or unknown fails the attempt, never retried.
- `chat.py`: `chat_model(choice, keys)` builds `ChatOpenAI` (Azure OpenAI, OpenAI) or `ChatAnthropic`, or the fake
  model in test mode. No fallback provider, no retry inside a call.
- `output.py`: the uniform output of every provider: text deltas without citation markers, citations by passage id
  and code-point offset, usage once per call under a worker-minted call id, and the typed failures `model.refused`,
  `model.unavailable` and `model.truncated`.
- `fake.py`: the fake chat model for any provider, with its directives (`fake:refusal`, `fake:outage`,
  `fake:truncated`, `fake:tool`).

The backend chooses the provider, model and tier per attempt; the worker states the providers it holds a key for at
every claim (`ModelKeys.providers`). Agents (`gmshield_worker.agents`) call models only through `chat_model` and
`output.stream_call`.
"""
