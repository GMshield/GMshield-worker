# GMshield-worker

AI worker for GMshield: ingestion, OCR, retrieval and calls to the cloud model. Stack: Python 3.14
with Docling and Tesseract, managed with uv. The worker calls only the backend's internal API, opens
no listening port, and runs with nothing downloaded at run time: the OCR data and the Docling models
are inside the image.

## Versions

Checked on 2026-11-13: each is the newest release at least two weeks old that day (released by
2026-10-30). `pyproject.toml` pins every direct dependency exactly, `uv.lock` (committed) holds
every transitive one, and `[tool.uv] exclude-newer` keeps them all on or before 2026-10-30.

| What | Version | Where it is set |
|---|---|---|
| Python | 3.14.10 (`python:3.14.10-slim-trixie`, pinned by digest) | `pyproject.toml`, `Dockerfile` |
| uv | 0.13.6 (`ghcr.io/astral-sh/uv:0.13.6`, pinned by digest) | `Dockerfile` |
| Docling | 2.145.0 | `pyproject.toml` |
| torch, torchvision (CPU builds, from the PyTorch CPU index) | 2.15.0, 0.29.2 | `pyproject.toml` |
| anthropic | 2.3.0 | `pyproject.toml` |
| httpx | 0.28.1 | `pyproject.toml` |
| attrs (used by the generated client) | 26.1.0 | `pyproject.toml` |
| openapi-python-client (client generator) | 0.29.1 | `pyproject.toml` (development) |
| pytest, pytest-cov | 9.2.0, 7.1.0 | `pyproject.toml` (development) |
| ruff | 0.17.3 | `pyproject.toml` (development) |
| mypy | 2.4.1 | `pyproject.toml` (development) |
| Tesseract | 5.5.0 (Debian trixie `tesseract-ocr` and `libtesseract5` 5.5.0-1+b1), with `tesseract-ocr-por`, `tesseract-ocr-eng` and `tesseract-ocr-osd` (orientation detection, used to straighten pages) 1:4.1.0-2 | `Dockerfile` |

**Tesseract differs from the architecture spine**, which names 5.5.4: only Debian unstable has
that version. The image uses trixie's 5.5.0, pinned by package version, so that every package comes
from the base image's own release. A Debian point release can replace these versions in the
archive, and the build then fails at `apt-get install`: look up the new versions with
`apt-cache policy tesseract-ocr libtesseract5 tesseract-ocr-por tesseract-ocr-eng tesseract-ocr-osd`
in the pinned Python image, update the pins in the `Dockerfile` and this table, and rerun the
image check.

### Security check of the pinned versions (2026-11-13)

`uvx pip-audit` over `uv export --no-hashes` and Trivy 0.76.1 (the pipeline's image) over the
lockfile and the production image:

- **soupsieve 3.0.4** (transitive: Docling, then beautifulsoup4) has CVE-2026-126190 (HIGH, a
  denial of service in the descendant combinator), fixed in 4.0. No fixed release meets the
  two-week rule yet: 4.0 came out on 2026-11-03 and 4.0.1 on 2026-11-04, so the lock keeps 3.0.4.
  Trivy counts it as fixable, so the pipeline's SCA and image gates block until it is raised;
  4.0.1 meets the rule from 2026-11-18.
- **zlib1g**: the image installs trixie's current security update, unpinned because the security
  archive keeps only the newest version (on 2026-11-13, `1:1.3.dfsg+really1.3.1-1+deb13u1`, which
  fixes CVE-2026-76844 and CVE-2026-85091).
- Every other high or critical finding in the image is a Debian package with no fix released
  (Trivy status `affected`, `will_not_fix` or `fix_deferred`), including Tesseract 5.5.0 and
  libxml2; the pipeline's image gate counts only fixable findings.
- pip-audit cannot check torch and torchvision (`+cpu` builds come from the PyTorch index, not
  PyPI); Trivy reported nothing for them.

## Structure

| Path | Holds |
|---|---|
| `gmshield_worker/main.py` | the supervisor (`python -m gmshield_worker.main`): start-up order, backend checks, shutdown |
| `gmshield_worker/settings.py` | the settings check |
| `gmshield_worker/log.py` | JSON log lines and W3C trace context |
| `gmshield_worker/health.py` | the health file and `python -m gmshield_worker.health` (the image's `HEALTHCHECK`) |
| `gmshield_worker/workdir.py` | emptying the work folder |
| `gmshield_worker/backend/` | the internal API: `client.py` (one client per request), `contract.py` (minimum contract version), `generated/` (generated, not committed) |
| `gmshield_worker/models/` | the model-client interface, the fake model and the Claude client |
| `gmshield_worker/lanes/`, `ingestion/`, `retrieval/`, `agents/` | empty until their stories (each `__init__.py` names it) |
| `contracts/internal.json` | the worker's copy of the backend's internal OpenAPI document |
| `tools/generate_client.py` | generates the client from `contracts/internal.json` |
| `tests/` | unit tests; `tests/image/` runs only inside the production image |

## Start-up and behaviour

1. **Settings.** Outside test mode a missing or invalid setting stops the start with exit status
   78 and one `startup.refused` line that names the settings, never a value.
2. **Work folder.** Everything inside `GMSHIELD_WORK_DIR` is removed; the folder itself stays, and
   a symbolic link is removed as a link, never followed. A removal that fails is logged by error
   type and the start continues. A folder that is a link or a file, or that cannot be created or
   read, stops the start with exit status 78 (`startup.refused`, reason `workdir.unusable`).
3. **Backend.** `GET /internal/v1/version` is retried with exponential backoff (1 s doubling up to
   30 s, with jitter of plus or minus 20%, 2 s per attempt) until the backend answers. Each failure
   is logged by attempt and error type only.

- **Contract version.** `MINIMUM_BACKEND_CONTRACT_VERSION` in `gmshield_worker/backend/contract.py`
  is the oldest backend contract the worker accepts. After a compatible answer the worker logs
  `worker.ready`, writes the health file and re-checks every 30 s. Against an older backend it logs
  one `backend.incompatible` line with both versions, stays up, unhealthy, and keeps checking.
- **Health.** The file `/tmp/gmshield-worker.healthy`, refreshed on every compatible re-check and
  removed as soon as a check fails. `python -m gmshield_worker.health` exits 0 when it exists and
  is at most 90 s old. The worker opens no port, so health cannot be an HTTP endpoint.
- **Backend calls.** Only through the generated client, a new one per request: a W3C
  `traceparent` with the start-up trace id and a new span id, a 2 s timeout, no redirects, and no
  proxy taken from the environment.
- **Model.** `models/` has one interface, `ModelClient`, with the fake model (deterministic, no
  network) and the Claude client (Anthropic SDK, default model `claude-sonnet-5-5`, the full tier;
  `claude-haiku-5-5` is the lightweight tier, and the backend will choose per job under AD-18; the
  SDK client is created on first use, so building it calls nothing). Test mode always uses the fake
  model; outside it a missing key is an error, never a silent fake.
  Nothing calls a model, Qdrant or Ollama in this story.
- **Logs.** One JSON object per line, ERROR lines on standard error and the rest on standard
  output: `timestamp`, `level`, `traceId`, `unit`, `event`, then ids,
  numbers, versions, status codes, setting names or error types. Never content, never a secret,
  never a URL.
- **Shutdown.** SIGTERM or SIGINT ends the process with exit status 0, whether it is retrying or
  ready.

## Settings

| Setting | Meaning |
|---|---|
| `GMSHIELD_BACKEND_URL` | the backend's address, `http` or `https`, without credentials (required outside test mode) |
| `GMSHIELD_ANTHROPIC_API_KEY` | the model provider's key (required outside test mode; secret) |
| `GMSHIELD_WORK_DIR` | optional, absolute; default `/tmp/gmshield-work` (Story 1.5 moves it onto the encrypted volume) |
| `GMSHIELD_MODE` | unset, or `test` |

The worker's service credential and its Redis credentials become required in Story 1.12, which
first uses them. `GMSHIELD_MODE=test` is set only by the pipeline: it needs no secret, uses the
fake model, and uses `http://stand-in-backend:8080` unless `GMSHIELD_BACKEND_URL` is set; the
worker then connects to that address only.

## Internal API client

`openapi-python-client` generates the client and its models from `contracts/internal.json` into
`gmshield_worker/backend/generated/`, with `--meta none` and the settings in
`tools/openapi-python-client.yaml`:

```sh
uv run python tools/generate_client.py
```

The folder is git-ignored and rebuilt on every image build and before every local test run. It is
never edited, and no hand-written class describes an internal request or response (a test checks
this). The generator also writes `generated/_contract.py` with the contract's version and SHA-256,
and a test fails when the client was generated from another copy of the contract.

### Refresh `contracts/internal.json`

The file is a copy of `GMshield-backend/openapi/internal.json`. When the backend's internal
contract changes:

```sh
cp ../GMshield-backend/openapi/internal.json contracts/internal.json
uv run python tools/generate_client.py && uv run pytest
```

When the worker needs something the new version added, raise `MINIMUM_BACKEND_CONTRACT_VERSION`
in `gmshield_worker/backend/contract.py` in the same change.

## Build and test

With uv 0.13.6 or later (uv uses Python 3.14 from the system or downloads it):

```sh
uv sync --locked
uv run python tools/generate_client.py
uv run ruff check && uv run ruff format --check
uv run mypy
uv run pytest                      # tests/image is left out: it runs in the image
```

The images:

```sh
docker build --target test .
docker build --target test-reports --output type=local,dest=out .   # out/junit/, out/coverage.xml
docker build -t gmshield-worker .
docker run --rm --network none --read-only --tmpfs /tmp gmshield-worker python /app/tests/image/test_image.py
```

The image check asserts that Tesseract lists `por` and `eng` and that Docling converts a small
generated PDF, with Tesseract as its OCR engine, with no network. The production image runs as UID
10001, writes nothing outside `/tmp` (so its root filesystem can be read-only), has no `EXPOSE`, no
pip and no compiler, and sets `HF_HUB_OFFLINE=1` and `DOCLING_ARTIFACTS_PATH=/opt/docling-models`.
Docling's `layout` and `tableformer` models are downloaded into `/opt/docling-models` when the image
is built.

To run the worker with the rest of GMshield, use the shared development setup in
`GMshield-backend/dev` (`dev/start`), which starts the worker after the backend.

## Branches and delivery

| Branch | What happens on push |
|---|---|
| `develop` | roberto-server runs unit tests, secret scan, SAST, SCA/SBOM/IaC, image scan and an active DAST scan; the quality gate decides whether this code may go to `main`. |
| `main` | Only code that already passed on `develop` is accepted. roberto-server builds and publishes the production image, deploys it, runs a passive DAST scan, and rolls back on failure. |

Work on feature branches, merge into `develop`, then open a pull request from `develop` to `main`.
Reports for every run are kept by the pipeline; see [GMshield-pipeline](https://github.com/GMshield/GMshield-pipeline).

## Pipeline contract

The pipeline reads [`.gmshield/pipeline.toml`](.gmshield/pipeline.toml) and builds the multi-stage
`Dockerfile`: `test` generates the client and runs ruff, mypy and pytest; `test-reports` holds the
JUnit results and Cobertura `coverage.xml`; the production image is the last stage. The worker has
no port (`port = 0`), so the pipeline skips staging and DAST, and a deploy is healthy when the
container is still running after 20 s: the worker keeps retrying the backend rather than exiting.
