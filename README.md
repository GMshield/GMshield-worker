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
| langchain-core, langchain-openai, langchain-anthropic (the chat models, Story 2.10; MIT) | 1.7.3, 1.7.3, 1.7.7 | `pyproject.toml` |
| langgraph (the agents' graphs, Story 2.10; MIT) | 1.2.19 | `pyproject.toml` |
| openai, anthropic (the provider SDKs, whose errors `models/output.py` maps; the versions the LangChain packages above lock; both bring `httpx2`, and `opentelemetry-api` without an SDK comes with anthropic) | 3.41.0, 2.3.0 | `pyproject.toml` |
| httpx2 (the provider SDKs' HTTP client; the tests build its requests and responses) | 2.16.0 | `pyproject.toml` (development) |
| httpx | 0.28.1 | `pyproject.toml` |
| redis (redis-py, released 2026-07-30) | 8.1.0 | `pyproject.toml` |
| attrs (used by the generated client) | 26.1.0 | `pyproject.toml` |
| pikepdf (the clean stage, Story 1.15; MPL-2.0, released 2026-10-11, qpdf 12.4.2 inside) | 10.17.0 | `pyproject.toml` |
| pypdfium2 (the triage stage, Story 1.16; already installed by Docling, released 2026-10-04) | 5.14.0 | `pyproject.toml` |
| pi-heif (the images stage's HEIC decoder, Story 1.21; released 2026-06-10). The binding is BSD-3-Clause; its wheel bundles libheif and libde265 (LGPL-3.0, decode only, no x265 encoder). Never pillow-heif, whose wheel bundles x265 (GPL) | 1.4.0 | `pyproject.toml` |
| qdrant-client (the index, Story 2.1; Apache-2.0, released 2026-10-22; REST only, its gRPC and protobuf dependencies come along unused) | 1.20.0 | `pyproject.toml` |
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

- **soupsieve 4.0.1** (transitive: Docling, then beautifulsoup4) is an exception to the two-week
  rule, approved by Dalton on 2026-11-13: 3.0.4 has CVE-2026-126190 (HIGH, a denial of service in
  the descendant combinator), fixed in 4.0, and 4.0.1 (released 2026-11-04) is the fixed release.
  `[tool.uv] exclude-newer-package` allows soupsieve up to 2026-11-05 only; remove that line once
  the global `exclude-newer` date passes 2026-11-04.
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
| `gmshield_worker/workdir.py` | emptying the work folder, each step's own folder and the sweep thread |
| `gmshield_worker/backend/` | the internal API: `client.py` (one client per request), `files.py` (a claimed job's input and output files, streamed), `contract.py` (minimum contract version), `generated/` (generated, not committed) |
| `gmshield_worker/models/` | the one way to a model (Story 2.10): `providers.py` (the providers, the claim's `ModelChoice`, base URLs and keys), `chat.py` (`chat_model`: `ChatOpenAI`, `ChatAnthropic` or the fake), `output.py` (the uniform events, the citation markers, the typed failures), `fake.py` (the fake chat model); the only place a provider SDK is imported |
| `gmshield_worker/lanes/` | the work lanes (Story 1.12): `consumer.py` (the loop both lanes share), `interactive.py`, `ingestion.py`, `renewal.py`, `streams.py` (Redis), `handlers.py` (handlers by kind), `answer.py` (Story 2.4's `question.answer`) and `batches.py` (its event batches), `probe.py`, `question.py` and `model_probe.py` (test mode only; Story 2.3's `test.question`, Story 2.10's `test.model`) |
| `gmshield_worker/ingestion/` | ingestion steps (Stories 1.15 to 1.21): `step.py` (the `ingestion` handler, dispatch on the stage), the stages' children `clean.py` (pikepdf), `images.py` (Pillow, pi-heif and pikepdf; Story 1.21), `triage.py` (pypdfium2), `layout.py` (Docling; pypdfium2 and Pillow for the page pictures), `ocr.py` (OpenCV, Docling and Tesseract) and `text.py` (standard library), `passages.py` (the in-process passages cutter), `sandbox.py` (the child's sandbox: environment, seccomp filter, deadline), `errors.py` (failure codes); `PIPELINE_VERSION` in `__init__.py` |
| `gmshield_worker/retrieval/` | the index (Story 2.1): `recipe.py` (`INDEX_RECIPE`), `embedding.py` (embeddinggemma through Ollama), `keywords.py` (the keyed keyword vector), `store.py` (the Qdrant collection `passages`), `indexing.py` (the `index.change` and `index.removal` handlers); the search (Story 2.3): `search.py` (`find_passages`) |
| `gmshield_worker/agents/` | the agents (Story 2.10): `graphs.py` (`compile_graph`, the only place a LangGraph graph is compiled; `AnswerWithTools`, the model-and-tools node), `tools.py` (`read_passages`, read-only under the job's grant) |
| `contracts/internal.json` | the worker's copy of the backend's internal OpenAPI document |
| `tools/generate_client.py` | generates the client from `contracts/internal.json` |
| `tools/record_embeddings.py` | records (or `--check`s) the real embeddinggemma vectors of the retrieval fixtures (Story 2.3) |
| `tools/record_model_responses.py` | records (or `--check`s) the real Azure OpenAI streams of the model fixtures, on roberto-server (Story 2.10) |
| `tests/` | unit tests; `tests/image/` runs only inside the production image, `tests/integration/` only beside a Redis (the `integration` stage); `tests/fixtures/retrieval/` the retrieval fixtures and their recorded vectors (SRD 5.2.1 passages, CC-BY-4.0, attribution in `ATTRIBUTION.txt`); `tests/fixtures/models/` each provider's HTTP streams, replayed by `tests/model_fixtures.py` |

## Start-up and behaviour

0. **Non-dumpable** (Story 1.15). `PR_SET_DUMPABLE 0`, before anything else; refused, the start
   stops with exit status 78 (`startup.refused`, reason `process.dumpable`).
1. **Tracing, then settings.** Any trace export stops the start, in every mode, with exit status 78
   (`startup.refused`, reason `tracing.on`, naming the variables; Story 2.10, below). Then, outside
   test mode, a missing or invalid setting stops the start with exit status 78 and one
   `startup.refused` line that names the settings, never a value.
2. **Work folder.** Everything inside `GMSHIELD_WORK_DIR` is removed; the folder itself stays, and
   a symbolic link is removed as a link, never followed. A removal that fails is logged by error
   type and the start continues. A folder that is a link or a file, or that cannot be created or
   read, stops the start with exit status 78 (`startup.refused`, reason `workdir.unusable`).
3. **Backend.** `GET /internal/v1/version` is retried with exponential backoff (1 s doubling up to
   30 s, with jitter of plus or minus 20%, 2 s per attempt) until the backend answers. Each failure
   is logged by attempt and error type only.
4. **Lanes.** After the first compatible answer the supervisor starts the interactive lane, then
   the ingestion lane; SIGTERM stops both.
5. **Index recipe.** After every compatible answer of a backend at internal contract 1.12.0 or later
   (at the start, then every 30 s), the worker states its `INDEX_RECIPE` with
   `PUT /internal/v1/index-recipe` (`backend.client.state_index_recipe`, the worker's credential, no
   grant). The backend reacts as to a recipe stated at a claim: what was indexed under another
   recipe, or not at all, gets new index changes at once, so a deploy starts indexing within one
   check, before any question or job. The backend keeps the last recipe in memory only, so stating it
   at every check also covers a backend redeployed while the worker runs; the same recipe again
   changes nothing. A failed statement is logged (`index.recipe_unstated`, by error type, or
   `unexpected_answer` with the status) and changes neither the health nor the lanes; the next check
   states it again. The first success is logged once (`index.recipe_stated`). Against a backend
   before 1.12.0 the route is never called and the worker works as before
   (`MINIMUM_BACKEND_CONTRACT_VERSION` stays 1.11.0; the gate is `INDEX_RECIPE_CONTRACT_VERSION`). Deploy order:
   the backend (internal contract 1.12.0, public 1.16.0), then the frontend, then this worker; the worker and the
   frontend do not depend on each other here. A rollback goes the other way.

- **Contract version.** `MINIMUM_BACKEND_CONTRACT_VERSION` in `gmshield_worker/backend/contract.py`
  is the oldest backend contract the worker accepts. After a compatible answer the worker logs
  `worker.ready`, writes the health file and re-checks every 30 s. Against an older backend it logs
  one `backend.incompatible` line with both versions, stays up, unhealthy, and keeps checking.
- **Health.** The file `/run/gmshield-worker/healthy` (a folder owned by the worker, mode 0700), refreshed on every compatible re-check and
  removed as soon as a check fails. `python -m gmshield_worker.health` exits 0 when it exists and
  is at most 90 s old. The worker opens no port, so health cannot be an HTTP endpoint.
- **Backend calls.** Only through the generated client, a new one per request: the worker's
  credential (`Authorization: Bearer`), a W3C `traceparent` with the start-up trace id (a job's own
  for its renewals and outcome) and a new span id, a 2 s timeout, no redirects, and no proxy taken
  from the environment.
- **Model.** Every model call goes through `models/` (Story 2.10, below). The start calls no model,
  Qdrant or Ollama; `startup.checked` logs `model` (`fake` in test mode, else `provider`) and the
  `providers` the worker states at each claim.
- **Logs.** One JSON object per line, ERROR lines on standard error and the rest on standard
  output: `timestamp`, `level`, `traceId`, `unit`, `event`, then ids,
  numbers, versions, status codes, setting names or error types. Never content, never a secret,
  never a URL.
- **Lanes (Story 1.12).** Each lane reads only its own stream (`gmshield:work:interactive` or
  `gmshield:work:ingestion`) with `XREADGROUP`, group `workers`, under one consumer name per process,
  one message at a time and only while it has a free slot: the interactive lane has
  `GMSHIELD_INTERACTIVE_CAPACITY` (default 2), the ingestion lane 1, so an interactive job is claimed
  at once even while ingestion is busy. At start a lane takes over the messages pending for more than
  2 minutes (`XAUTOCLAIM`): a worker that died between reading and claiming.
  - For each message it claims the job (`POST /internal/v1/jobs/{id}/claims`). After a definite
    answer (2xx or 409) the message is acknowledged (`XACK`); after no answer or a 5xx it stays
    pending, and the backend offers the job again if nobody claims it. The backend alone decides
    retries.
  - A claimed job runs its kind's handler while a timer renews the claim at its `renewAt`
    (`POST …/claims/{claimId}/renewals`, with `X-Job-Grant`). A refused renewal, or none answered
    before `lapsesAt`, stops the work and nothing is reported: the backend takes the job back. The
    outcome (`POST …/claims/{claimId}/outcome`: `done` with a result, or `failed` with a code) is
    retried while the claim lives. A handler that raises is a failed attempt (`worker.handler_failed`),
    a kind without a handler `worker.unknown_kind`.
  - Every line about a job, and every renewal and outcome, carries the job's trace id (the claim's
    `traceparent`): the backend logs the same id for the same job.
  - Handlers: `ingestion` in every mode (Story 1.15, below), `index.change` and `index.removal` (Story 2.1) and
    `question.answer` (Story 2.4, below) in every mode, the test-mode `probe` (it works
    `workMs`, fails its first `failTimes` attempts with `probe.failed`), the test-mode `test.question` (Story 2.3,
    below) and the test-mode `test.model` (Story 2.10, below).
  - Redis down, restarted or not yet holding the group (`NOGROUP`): the lane logs
    `stream.unavailable` by error type and retries with backoff (1 s doubling up to 30 s). The worker
    never runs `XADD`, `XTRIM` or `XGROUP`; its Redis account allows only `AUTH`, `HELLO`, `PING`,
    `XREADGROUP`, `XACK`, `XAUTOCLAIM`, `XCLAIM` and `XPENDING` on the two streams. The client speaks
    RESP2 and sends no `CLIENT SETINFO`.
- **Ingestion steps (Story 1.15).** The `ingestion` handler runs one step of a version's chain; this
  story has one stage, `clean`, for a PDF. In `<work>/<jobId>/` (erased however the step ends) it
  streams the step's input (`GET …/claims/{claimId}/inputs/source`, 8 MiB chunks, its own timeouts:
  2 s to connect, 60 s per read), runs `ingestion/clean.py` in the sandbox, streams the cleaned
  original and every batch of 20 pages back (`POST …/outputs`, each checked against the answer's
  size and SHA-256) and reports `{pipelineVersion, pageTotal, original, batches, next: []}`. A PDF
  that needs a user password fails with `ingestion.password_required` and one qpdf cannot read with
  `ingestion.file_unreadable`, both with `retry: false`; a crash, the deadline or a failed transfer
  is `ingestion.step_failed`, which the backend retries while attempts remain.
  - The clean stage removes JavaScript, embedded and attached files, `OpenAction` and every `AA`,
    every action other than GoTo and URI, XFA, and RichMedia, Movie, Sound, Screen, 3D and file
    attachment annotations, and saves without encryption (an owner password is dropped). A batch
    holds nothing of the other pages.
  - **Sandbox.** The child runs isolated (`python -I -B`) with only `PATH`, `LANG` and `HOME` (its
    folder) in its environment, its folder as working directory, every other descriptor closed and
    standard input and output on `/dev/null`, in a session of its own that is killed at 9 minutes
    30 seconds, on cancel, and when it ends. A launcher installs a seccomp filter before `exec`
    (`PR_SET_NO_NEW_PRIVS`, then classic BPF through ctypes for x86-64 and AArch64; any other
    machine refuses to start the child) that answers `EACCES` to `socket()` for every family but
    `AF_UNIX`, to `io_uring_setup`, and to any call of another ABI. A network namespace is not
    possible: the container drops every capability and runs with no-new-privileges. The worker
    makes itself non-dumpable (`PR_SET_DUMPABLE 0`) before anything else, so no child can read its
    memory or `/proc/<pid>/environ`.
  - **Work folder.** Besides the emptying at start, a sweep thread removes every 2 minutes what no
    running step holds and was last changed more than 10 minutes ago.
  - Logs carry the job's trace id, the sizes, counts and times, and the child's and the worker's
    peak RSS (`childMaxRssKib`, `workerMaxRssKib`). A generated 1 GiB PDF is cleaned with a child
    peak under 512 MiB (`tests/test_clean.py`).
- **Reading pages (Stories 1.16 and 1.17; `PIPELINE_VERSION` "3", now "6").** The clean report asks one `triage` step per
  batch. Triage (`triage.py`, pypdfium2) gives each page a layer: `text` when its text layer has at
  least 16 characters other than whitespace, unless it has fewer than 200 and one image covers at least
  half of the page, else `image`; its report gives the `layers`, one `layout` step per run of text pages,
  then one `ocr` step per run of image pages, cut to at most `OCR_PAGES` (10). Layout (`layout.py`)
  reads the run with Docling 2.145.0 (`do_ocr=False`, table structure on; the models' folder and 2
  threads on the command line, since the child's environment is empty) and sends one JSON file of
  its pages back as `texts`, each page `read_directly`: the page's lines (text and box) and its
  blocks in reading order (label, text, box, heading level, table rows), boxes as fractions from the
  top-left corner. A TXT or Markdown upload gets one `text` step (`text.py`): decoded as it streams
  (UTF-8 without its BOM, else cp1252) and cut into pages of at most 3,000 code points at Markdown
  headings, then blank lines, line ends and whitespace, each with its heading path and code-point
  range; the file goes back unchanged as the `original`, with one `texts` file per 500 pages.
  - OCR (`ocr.py`, Story 1.17) renders each image page at 216 DPI (smaller when its longer side would
    pass 10,000 pixels), turns it upright with Tesseract's orientation detection (`--psm 0 -l osd`),
    checked by reading (Story 1.22: the detection was half a turn wrong on every page of an 1892 scan): Tesseract
    reads a band across the middle of the page as detected and half a turn further, and half a turn further wins
    only when it reads at least 10 more words at 60 or more confidence (about 1 s more per page on roberto-server;
    Docling's own detection is switched off, since the page reaches it upright),
    straightens a skew within ±20° with OpenCV (from 0.5°, white fill), and reads it with Docling and
    Tesseract (`por`+`eng`, full page, tables and heading levels) into layout's page shape, its lines
    being Tesseract's words. Each page is `read_by_ocr`, or `unreadable` when OCR finds fewer than 16
    characters other than whitespace (a blank scan); the pages read go back as `texts`.
  - A step whose claim input names another `pipelineVersion` fails before any download with
    `ingestion.pipeline_changed` and `retry: false`; the backend restarts the chain.
  - From the claim's second attempt on, triage, layout and OCR run one child per page, each with 540 s
    divided by the step's pages; a page whose child fails, crashes or runs out of time is reported
    `unreadable`.
- **Passages (Story 1.18, `PIPELINE_VERSION` "4", backend contract 1.4.0 or later).** The clean report and the text
  report end their `next` with `{"stage": "passages", "firstPage": 1, "lastPage": <pageTotal>, "input": "texts"}`:
  a step over every text file of the version, which the backend holds with no job until every page has an outcome.
  Its claim input names those files in page order (`texts: [{name: "text-<fileId>", firstPage, lastPage}]`). The
  stage runs in-process, with no child (it reads only our own JSON): it downloads one text file at a time
  (`GET …/inputs/text-<fileId>`) and feeds its page records to `passages.py`, which carries the heading stack across
  files.
  - PDF records: blocks in reading order; a heading block (`headingLevel` set) closes the open sections at its level
    or deeper and opens its own; the section path is the open headings, outermost first, and every heading with text
    is an outline entry `{level, title, page}`. Text records: `headingPath` is the section path, and each heading a
    page newly enters is an outline entry whose level is its depth. `page_header` and `page_footer` blocks, and
    blocks with no text, never enter a passage.
  - Cutting at 3,000 code points (the text stage's page limit): a new passage starts at a new section, at a new page
    (a passage lies on one page) or when the next block would pass the limit. A passage that does not open its
    section starts with its leaf heading, so each part of a long stat block repeats it. A table (rows one per line,
    cells joined by ` | `) stays whole while it fits; a longer one is cut between rows, each part starting with the
    heading and the first row. Any other block over the limit is cut as `text.py` cuts; a text page at its blank
    lines.
  - The report gives `passages: [{page, sectionPath, text, boxes | range}]` in reading order (`boxes`: the blocks'
    boxes on the page; `range`: `[start, end)` in code points of the page's text) and `outline`. The backend mints
    the passage ids and stores both encrypted; a version with no text file (every page unreadable) reports none.
- **Page views (Story 1.19, `PIPELINE_VERSION` "5", backend contract 1.5.0 or later).** Every page a reading child
  reads gets a page view in its step's report, `pageViews: [{page, picture: {fileId, width, height}, lines: [{text,
  box}]}]`, or for a text page `{page, text}`. A page whose child fails, crashes or times out gets none.
  - `layout.py` renders each page it read with pypdfium2, after Docling, at the scale whose longer side is 2,000
    pixels (never above the OCR stage's 3.0), and writes it as WebP of quality 80 (`page-NNNNNNN.webp`, listed with its
    size in the manifest's `pictures`). `ocr.py` keeps the straightened picture it hands to Docling, shrunk the same
    way, so its boxes (fractions of that picture) still lie over its lines; a blank scan reported `unreadable` keeps
    its picture with no lines.
  - The parent checks each listed picture (pages asked for, each once, a plain file of its own, 1 to 10,000 pixels a
    side), sends it back as an output, and takes the page's lines from its record: those with text and a box of four
    fractions, each cut to 4,096 code points, at most 10,000. The text stage gives each page's text from its own
    `texts` files (at most 16,384 code points).
  - The backend stores one encrypted page view per page and adopts each picture as a `picture` file of its page; a
    report replaces the views of the pages it carries. Deploy the backend before this worker.
- **Photos as the pages of one material (Story 1.21, `PIPELINE_VERSION` "6", backend contract 1.6.0 or later).** An
  image set's version has the format `images` (an image sent alone is a set of one). Its first step, `images`, has the
  claim input `input: "images"` and `inputs: [{name: "image-<uploadId>", format}]`, the set's images in their
  confirmed order, each downloaded by its name (`GET …/inputs/image-<uploadId>`).
  - `images.py` decodes each with Pillow (HEIC through pi-heif, no system libheif), turns it upright by its EXIF
    orientation, lays a transparent one on white, keeps it grey or makes it RGB, shrinks it (never enlarges) to at most
    24 million pixels and 10,000 a side, and writes it as a JPEG (quality 90, which also drops the photo's other
    metadata) on a page of its pixels at 216 DPI, the OCR stage's resolution. A multi-frame TIFF gives one page per
    frame, in place. A picture past 200 million pixels is refused before it is decoded. An image or a frame that does
    not decode is a blank A4 page at its place; it never fails the step.
  - The child writes what `clean.py` writes (`original.pdf`, batches of 20 pages, the manifest) plus `unreadable`,
    the blank pages (logged as `undecodedPages`). The report is clean's: `original`, `batches`, one `triage` step per
    batch and the held `passages` step. Triage, layout, OCR and passages then read the pages as a PDF's (format
    `images`); OCR reports a blank page `unreadable`, as any blank scan, so Reprocessar finds the step that read it.
    The backend removes every image's raw upload once the original is recorded.
  - From the claim's second attempt on, one child decodes each image (`--decode`, a share of the 540 s each) and a
    last one assembles (`--assemble`): an image whose child crashes or runs out of time is one unreadable page.
  - A restart (`ingestion.pipeline_changed`) or a retry from the stored original gives the `images` step
    `input: "source"`, the assembled PDF, which is cleaned and split again as `clean` does.
  - The in-image check decodes `tests/image/two-colours.heic` (provenance in `tests/image/test_image.py`) in the
    sandbox and reads two photographed pages, one of them lying on its side with its EXIF orientation, by OCR.
- **Index an accepted material (Story 2.1, AD-10, backend contract 1.7.0 or later).** Every claim states the
  worker's `INDEX_RECIPE` (`retrieval/recipe.py`; bump it on any change to the model, the embedding input, the
  splitting or the keyword scheme): the backend re-indexes each source recorded under another one. From backend
  contract 1.12.0 the supervisor also states it at every check (start-up step 5), so indexing starts right after a
  deploy, not at the first job claimed. A run that fails stays not indexed: the backend keeps its passages in the
  keyword fallback's `pending` and retries the failed run alone every 10 minutes, at most 3 times. The worker is the
  only reader and writer of Qdrant.
  - `index.change`: the claim input is `{source, owner, kind, language, tokenKey, passages}`; the run (at most 200
    passages) is read whole from the input `passages`, JSON Lines `{id, ordinal, page, title, text}`. Four at a
    time (`BATCH`), each passage is embedded by `embeddinggemma:300m` through `POST {GMSHIELD_OLLAMA_URL}/api/embed`
    (plain httpx; input `title: {title or "none"} | text: {text}`, `truncate: false`). Questions go first: within the
    worker, no index call (a batch, or one passage or one part of a split passage) starts while a question's embedding
    waits or runs (`QUESTIONS_FIRST`, one gate shared by every `Embedder`), waiting at most 10 s
    (`INDEX_WAIT_SECONDS`) before it goes anyway. Ollama embeds one request at a time (about 0.6 s a passage), so a
    question waits behind at most one batch of 4. The gate holds within one worker process: that bound assumes one
    worker process per Ollama, as deployed. Neither the batch size nor the gate is part of the index recipe. When Ollama
    refuses an input as longer than the context, the passage is split in two by `ingestion.passages.split`, each half
    embedded with the same title and split again if refused, at most 4 times deep, and its vector is the L2-normalized
    mean of its parts: nothing is cut silently. The keyword vector folds the text (NFKC, casefold, diacritics stripped),
    takes its Unicode word tokens and numbers each by the first 4 bytes, big-endian, of HMAC-SHA256 under the owner's
    `tokenKey`, valued by its count; Qdrant applies the `idf` modifier. Each passage is upserted as one point (id: the
    passage id; vectors `dense` and `sparse`; payload exactly `{owner, source, kind, page, language}`). The result is
    `{indexRecipe, points}`.
  - The collection `passages` is created when missing: `dense` (768, cosine), `sparse` (`idf`), a tenant keyword index
    on `owner` and a keyword index on `source`. Qdrant is reached over REST at `GMSHIELD_QDRANT_URL` with its API key,
    an https URL verified against `/run/gmshield/certs/ca.crt`, one client per job.
  - `index.removal`: the claim input is `{source, owner}`; the points go by a filter on both, never by ids. Safe to
    repeat.
  - Failures: `index.input_invalid` and `index.passage_too_long` with `retry: false`; `index.embedding_unavailable`
    (Ollama), `index.store_unavailable` (Qdrant) and `index.input_unavailable` (the backend's input route) for the
    backend to retry; a re-run upserts the same ids. Lines carry ids, counts and codes only; the token key is held with
    `repr=False` and no text, title or token reaches Qdrant, Redis or a log.
  - Deploy the backend first: an older backend fails the claim body's contract check (`backend.incompatible`).
- **Passages by id (backend contract 1.8.0, Story 2.2).** `contracts/internal.json` carries
  `POST /internal/v1/jobs/{jobId}/claims/{claimId}/passages` (`readJobPassages`): with the claim's `X-Job-Grant` and
  `{ids}` (1 to 100 passage ids), the backend answers `{passages: [{id, tier, text, sectionPath, page}]}` in the order
  asked, each id once, `tier` being `campaign` or `system`. It checks every id against the job's resolved scope before
  decrypting anything. The refusals: 400 `request.invalid` for a malformed body or an id that is not a UUID (before
  the grant); 403 `job.grant_refused` for a wrong, ended or lapsed grant; 400 `job.passages_invalid` for no id, more
  than 100 or a null one; 403 `job.passages_refused` for a kind that reads no passage; 404 `request.not_found` when the
  job's campaign is gone; 403 `scope.passage_refused`, naming no id, when any id is out of the scope. The worker calls
  it through `JobsApi.passages` since Story 2.3, so `MINIMUM_BACKEND_CONTRACT_VERSION` is 1.8.0.
- **Find the passages that answer a question (Story 2.3, AD-10, backend contract 1.8.0).**
  `retrieval/search.py` `find_passages` runs for an interactive job whose claim input is the backend's question input
  (`Scopes.questionInput`): `{question, owner, tiers: [{tier, sources, pending}], tokenKey}`, `question` 1 to 4,000
  characters (code points), the tiers in order (campaign, then system), `sources` the allowed version ids, `pending`
  the ids of the tier's passages that are not indexed (their index change has not ended or failed; the backend lists
  at most the first 5,000, 100 before backend contract 1.12.0),
  `tokenKey` Base64 of the owner's 32-byte token key. Provider, model and tier come with the claim's `model` (Story
  2.10), never with the input; nothing here calls a model.
  - When no tier has a source the answer is an empty list, with no embedding, no Qdrant query and no passages call.
  - The question is embedded as at indexing (`embeddinggemma:300m`, `truncate: false`, normalized) under the query
    prompt `task: search result | query: {question}`; its keyword vector is `keyword_vector(question, tokenKey)`. The
    index recipe does not change.
  - Per tier with sources, one `query_points` call: a dense and a sparse prefetch (40 each) fused by Qdrant's RRF
    (`FusionQuery(Fusion.RRF)`), 8 points, the filter `owner == the job's owner AND source IN the tier's sources` on
    the query and on each prefetch. A question with no keyword token has no sparse prefetch. No payload or vector is
    read back.
  - Pending: once every tier is searched, the first 5,000 (`MAX_PENDING`) `pending` ids of each tier, in input order,
    are read by id in pages of 100 (`PAGE`, the backend's most per passages call), at most 4 at a time (`PARALLEL`),
    the tiers' pages in turns (campaign, system, campaign, ..., then the rest of the longer one), under one deadline
    1.5 s (`PENDING_BUDGET_SECONDS`) after the question's first pending read, both tiers together (`read_pending`), so
    the first words stay within NFR-14's 5 s: no page starts after it, and a page not back by it is dropped, never
    waited for. A page that fails before the deadline fails the search as one read did. Each page is tokenized as it
    comes back, in its read's thread and so inside the deadline: each passage keeps only the question's distinct folded
    tokens of two characters or more (`question_terms`, from `keywords.tokens`, never an HMAC) that it contains. After
    the deadline only the counting runs: a passage scores the sum, over those tokens, of BM25's weight
    ln(1 + (N - df + 0.5) / (df + 0.5)), N the pending passages read for the question (both tiers) and df how many
    contain the token (`rarity`), so a rare name ("tarrasque") outweighs common words ("armor", "class", "the"); a
    passage scoring 0 is dropped, and ties keep input order (`rank_pending`). That ranking is fused with the tier's
    Qdrant results as a third RRF list (a pending id at rank r scores 1/(r + 2), Qdrant's constant), 8 ids; an id
    found both ways is one entry whose scores add up, as RRF does. The ids not read (past the cap, in pages the
    deadline left out, or in pages it dropped) are skipped and logged as a count per tier
    (`retrieval.pending_skipped`; `pendingFetched` and `pendingSkipped` in `retrieval.search_done`). A question with no
    token of two characters or more reads no pending passage, since none could match.
  - The text of every found id, in rank order, campaign tier first, comes from one passages call: the handed-over list
    is `[{id, tier, text, sectionPath, page}]`. Library text reaches the worker by no other path.
  - Failures: `retrieval.embedding_unavailable` (Ollama), `retrieval.store_unavailable` (Qdrant) and
    `retrieval.passages_unavailable` (the passages route down, or an answer out of the contract) for the backend to
    retry; `retrieval.question_too_long` (refused by the model), `retrieval.input_invalid` and a refusal of the
    passages route with its own code (`scope.passage_refused`, ...), or `retrieval.passages_refused` when it names none,
    with `retry: false`. Lines carry ids, counts and timings (`retrieval.search_done`: `embedSeconds`,
    `searchSeconds`, `pendingSeconds`, `rankSeconds` for the counting after the pending deadline when pending passages
    were read, and `readSeconds`), never the question, a text or the key.
  - No production kind carries a question until Story 2.4. The test-mode kind `test.question` (`lanes/question.py`)
    runs `find_passages` and reports the handed-over ids, `{ids: [...]}`.
  - Tests (`tests/test_search.py`) use Qdrant's local mode and the real embeddinggemma vectors of a fixed fixture set
    (`tests/fixtures/retrieval/`: SRD 5.2.1 passages in English, Portuguese passages written for the tests, and their
    questions), recorded once through the Ollama tunnel (rounded to 7 decimals) and committed; an input that was not
    recorded fails the test. Re-record after changing a fixture, the prompts or the model, and check the recording
    against the live model:

    ```sh
    uv run python tools/record_embeddings.py           # rewrites tests/fixtures/retrieval/vectors.json
    uv run python tools/record_embeddings.py --check   # expected: the largest difference is within 1e-4
    ```

    Ollama is `GMSHIELD_OLLAMA_URL`, else `http://localhost:11434` (the `roberto-server` SSH alias forwards it).
- **Models through LangGraph, with a switchable provider (Story 2.10, AD-18, backend contract 1.9.0).** The backend
  chooses the provider, model and tier of each attempt; the worker never chooses one and never falls back to another.
  - **Claim.** Every claim states `providers`: those the worker holds a key for (test mode: `azure-openai`,
    `anthropic` and `openai`). A claim of a kind that calls a model answers `model: {provider, model, tier}`
    (`ClaimedJob.model`; `None` for every other kind). The backend refuses such a job to a worker that did not state
    its provider: the job ends `failed` with `job.provider_unavailable`, no attempt used, and the claim is answered
    409 `job.not_offered`. A claim that names a provider the worker has no key for fails the attempt with
    `model.provider_key_missing`, one it does not know with `model.provider_unknown`, both with `retry: false`.
  - **Chat models** (`models/chat.py`, `chat_model`). Azure OpenAI and OpenAI: `ChatOpenAI` on Chat Completions,
    chosen explicitly (`use_responses_api=False`), at an explicit base URL (Azure: `GMSHIELD_AZURE_OPENAI_BASE_URL`,
    the resource's `https://<host>/openai/v1/`, with the deployment name as the model; OpenAI:
    `https://api.openai.com/v1`), `store=False`, `stream_usage=True`, no provider-hosted tool, no `api-version`, never
    `AzureChatOpenAI`. Anthropic: `ChatAnthropic` at `https://api.anthropic.com`. Every model has `max_retries=0` (a
    job retry is the only retry), a 60 s timeout and at most 4,096 output tokens. Test mode builds the fake model of
    the claim's provider. Provider SDK and LangChain provider imports appear only in `models/` (a test fails the build
    otherwise).
  - **Uniform output** (`models/output.py`, `stream_call`). Each call has a worker-minted call id (a UUIDv7). Its
    events, in order: `TextDelta` (display text, the citation markers removed), `Citation(passage_id, offset)` (the
    model writes `[[<passage id>]]` after a statement; the offset is in code points of the display text; only ids
    given to the model are kept, a marker with another id is removed; a marker split across chunks is held back until
    the next chunk decides it), one `Usage(call_id, response_id, input, cache_read, cache_write, output, reasoning)`
    from `usage_metadata` (input includes the cache reads and writes; the provider's response id when it sent one),
    and `ModelFailed(code)` when the call ended in `model.refused` (a refusal or content filter), `model.unavailable`
    (connection, timeout, 5xx, 429, an error inside the stream) or `model.truncated` (the length or max-tokens stop,
    or a prompt past the context window). Deltas may already have gone out before a trailing `ModelFailed`: it ends
    the answer, and they are not withdrawn. Story 2.6: `stream_call(..., should_stop=)` asks `should_stop` before it
    takes each chunk; once it is true, the provider's stream is closed, the held-back marker text is flushed as text,
    the usage the provider sent is emitted, or else an estimate marked `estimated=True` (input: the code points of the
    messages' text ÷ 4; output: the code points of the text emitted ÷ 4; both rounded up), and the call ends
    `CallResult(stopped=True)` with no failure. `AnswerWithTools` passes its `should_stop` into every call.
    Any other exception propagates (the attempt fails as `worker.handler_failed`).
  - **Agents** (`agents/`). Every agent is a LangGraph graph compiled by `graphs.compile_graph`: in memory for one
    job, with no persistence, shared store or pause (a test fails the build otherwise), run with no callback handler of ours and
    within 10 steps. `AnswerWithTools` calls the model, runs the tools it asks for and gives their results back, at
    most 4 calls (then `model.truncated`). Tools are read-only internal-API calls under the job's grant:
    `read_passages` reads passages by id (`JobsApi.passages`) and gives their ids to the citation markers. Library
    text enters only as tool results or user-turn data, never in the system prompt.
  - **Fake model** (`models/fake.py`, test mode). A LangChain chat model shaped as its provider's integration
    (Chat Completions chunks for `azure-openai` and `openai`, Anthropic chunks for `anthropic`), deterministic,
    streamed in small chunks so markers split, citing the passage ids it was given. Directives in the user turn:
    `fake:refusal`, `fake:outage`, `fake:truncated` (one `ModelFailed` each) and `fake:tool` (it calls
    `read_passages` with the user turn's ids, then cites the results).
  - **`test.model`** (`lanes/model_probe.py`, test mode only). Input `{prompt}` (1 to 4,000 characters); a one-node
    graph around `AnswerWithTools` with `read_passages`; the result is `{model, events, usage}`. A typed failure fails
    the attempt with its code (`model.unavailable` retried by the backend, the others with `retry: false`); a claim
    without a model or a prompt fails with `model.input_invalid`. Lines carry ids, the provider, model and tier,
    call counts and token totals, never a prompt or a text.
  - **Tracing is refused.** The worker does not start (`tracing.on`, naming the variables) when
    `LANGSMITH_` or `LANGCHAIN_` followed by `TRACING`, `TRACING_V2`, `OTEL_ENABLED` or `OTEL_ONLY` (any of the eight)
    is truthy, `LANGSMITH_GATEWAY` is set,
    `OTEL_TRACES_EXPORTER`, `OTEL_METRICS_EXPORTER` or `OTEL_LOGS_EXPORTER` is anything but empty or `none`, any
    `OTEL_EXPORTER_OTLP_*` variable is set, or `ANTHROPIC_OPEN_TELEMETRY_TRACES_CONTENT_MODE` is `content`. The image
    holds no OpenTelemetry SDK or exporter (a test checks the installed packages and the lock).
  - **Recorded streams** (`tests/fixtures/models/`, replayed over HTTP by `tests/model_fixtures.py` in
    `tests/test_model_recordings.py`). Each real chat model gets its provider's bytes: an answer (display text, its
    citation, the usage), a refusal (for Azure also a Chat Completions refusal delta), a 503 and a length stop (and
    for Anthropic an error inside the stream), mapped
    to the same codes as the fake's. The requests are checked too: the explicit path, no `api-version`,
    `store: false`, usage streamed, no tool, one request. The prompt is synthetic, with an invented passage. The
    Anthropic cases are hand-built from its documented format (no Anthropic key exists). Azure's `answer` and
    `length` are recorded on roberto-server, inside the worker image, with the server's `worker.env`, so the key
    never leaves the server; the tool drops every header, replaces the response ids and refuses a body that holds the
    key or the host, and only the cleaned file is copied back. Its `refusal`, `refusal_delta` and `outage` stay
    hand-built (a synthetic prompt causes none of them). Each case says `recorded` or `hand-built`.

    ```sh
    # on roberto-server, in a folder holding a copy of tools/ and tests/ (~/gm2-10-record):
    docker run --rm --env-file ~/gmshield-ci/secrets/worker.env --user "$(id -u):$(id -g)" \
        -v ~/gm2-10-record:/record -w /record <worker image> python tools/record_model_responses.py
    # the same with --check: today's live streams still have the recorded shape
    ```

  - Deploy the backend first: an older backend fails the contract check (`backend.incompatible`), so
    `MINIMUM_BACKEND_CONTRACT_VERSION` is 1.9.0 (1.10.0 from Story 2.4, 1.11.0 from Story 2.6).
- **The answer to a question (Story 2.4, `question.answer`, backend contract 1.10.0).** Registered in every mode
  (`lanes/answer.py`), on the interactive lane.
  - The claim's input is the backend's question input (`retrieval.search.question_input`; else
    `retrieval.input_invalid`, no retry) and its `model` the attempt's provider, model and tier (none:
    `model.input_invalid`, no retry; a provider this worker cannot call: its code, no retry).
  - `find_passages` finds the passages (Story 2.3, its failures as there). Then one model call through a one-node
    LangGraph graph (`agents.graphs.answer_graph` around `AnswerWithTools` with no tools, compiled by `compile_graph`)
    with `models.chat.chat_model(job.model)`. The `SystemMessage` holds instructions only: answer only from the
    passages, write `[[<passage id>]]` after each statement, answer in the question's language, and say so in one
    sentence when the passages do not answer. The one `HumanMessage` holds the passages as data blocks
    (`<passage id="…" section="Magias > Fire Bolt">text</passage>`, XML-escaped), then `<question>…</question>`.
    The handed ids go to `CitationMarkers.give`, so only they can be cited. Nothing of the account (owner, token key,
    grant) is sent.
  - Every event goes to `lanes/batches.EventBatcher`, whose sender thread posts numbered batches to
    `POST …/claims/{claimId}/events` (`JobsApi.events`, from 1 per claim): the first text delta at once and alone,
    then at most every 100 ms or 32 events; a call's usage rides in the next batch. A 5xx or a transport error is
    retried with the same number while the claim lives; any other refusal stops the answer (`failed` with its code, no
    retry). The final batch is flushed before the outcome.
  - A model failure is reported with its code (`model.refused`, `model.truncated`, `model.unavailable`). Only an
    unavailable store, embedder, passages route or model before any event was accepted may be retried: once the GM
    has seen words, the attempt fails with `retry: false` (the backend refuses a splice too).
  - **The GM's stop (Story 2.6, backend contract 1.11.0).** A batch's answer `{stop: true}` sets the batcher's `stop`.
    From then on the batcher posts no event (those waiting and any emitted later are dropped), only the usage still to
    report, sent with `estimated` (`JobsApi.events`). The model stream is closed before its next chunk (`should_stop`
    is `cancel` or the batcher halted), and the outcome is `done`: a stop is the GM's choice, not a failure.
    `answer.ended` says `stopped=true`. Deploy order for this story: backend, then worker, then frontend.
  - Lines (`answer.started`, `answer.ended`, `events.posted`) carry ids, counts and timings only.
- **Shutdown.** SIGTERM or SIGINT stops the lanes (a running job is abandoned unreported and the
  backend takes it back when its claim lapses) and ends the process with exit status 0, whether it
  is retrying or ready.

## Settings

[`.env.example`](.env.example) lists them with their defaults and empty secrets, for a start by hand
(`docker run --env-file .env`).

| Setting | Meaning |
|---|---|
| `GMSHIELD_BACKEND_URL` | the backend's address, `http` or `https`, without credentials (required outside test mode) |
| `GMSHIELD_AZURE_OPENAI_API_KEY`, `GMSHIELD_AZURE_OPENAI_BASE_URL` | Azure OpenAI (Story 2.10; the key is secret): the key, and the resource's v1 URL `https://<resource>.openai.azure.com/openai/v1/` (https, that path, no query); a key without the URL is refused (`setting.missing`), a URL that is not one too (`setting.invalid`) |
| `GMSHIELD_ANTHROPIC_API_KEY`, `GMSHIELD_OPENAI_API_KEY` | Anthropic's and OpenAI's keys (Story 2.10; secret) |
| `GMSHIELD_WORKER_CREDENTIAL` | the worker's credential for `/internal/v1` (required outside test mode; secret): 32 to 512 characters of `A-Z a-z 0-9 . _ ~ + / = -`, the same value as the backend's |
| `GMSHIELD_REDIS_HOST`, `GMSHIELD_REDIS_PORT` | Redis (the host required outside test mode; port default 6379) |
| `GMSHIELD_REDIS_USERNAME`, `GMSHIELD_REDIS_PASSWORD` | the worker's Redis account (required outside test mode; `worker` in production; the password is secret) |
| `GMSHIELD_REDIS_TLS` | `true` (default): TLS verified, chain and host name, against `/run/gmshield/certs/ca.crt`; `false` only in the development setup |
| `GMSHIELD_INTERACTIVE_CAPACITY` | interactive jobs at once, 1 to 16 (default 2); the ingestion lane runs one |
| `GMSHIELD_OLLAMA_URL` | Ollama, for the embeddings (Story 2.1), `http` or `https` without credentials (required outside test mode; production: `http://172.30.5.1:11434`, the `gmshield-model` gateway, set in `.gmshield/pipeline.toml`) |
| `GMSHIELD_QDRANT_URL`, `GMSHIELD_QDRANT_API_KEY` | Qdrant (Story 2.1; both required outside test mode, the key is secret; production: `https://qdrant:6333`, written with the key by GMshield-pipeline `deploy/data-credentials.sh`); an https URL is verified against `/run/gmshield/certs/ca.crt` |
| `GMSHIELD_WORK_DIR` | optional, absolute; default `/var/lib/gmshield-worker/work`, where production mounts the named volume `gmshield-worker-work` (on the encrypted volume) |
| `GMSHIELD_HEALTH_FILE` | for tests that run the worker outside the image only; default `/run/gmshield-worker/healthy` |
| `GMSHIELD_MODE` | unset, or `test` |

Each model provider's key is optional, and outside test mode at least one is required: none refuses the start
(`setting.missing` naming the three). A key must be visible ASCII without spaces (it goes into a header). The worker
states at each claim the providers it holds a key for; the backend's `GMSHIELD_MODEL_PROVIDER` chooses the provider.
A placeholder key the worker does not really hold makes it claim jobs it cannot run (they fail with the provider's
error): leave out every key that is not real.

`GMSHIELD_MODE=test` is set only by the pipeline: it needs no secret, reads no model key, uses the fake model for every
provider (and states all three), uses
`http://stand-in-backend:8080` unless `GMSHIELD_BACKEND_URL` is set, Redis at `redis:6379` (or
`GMSHIELD_REDIS_HOST`/`GMSHIELD_REDIS_PORT`) without an account, a password or TLS, and the fixed,
public worker credential `gmshield-test-mode-worker-credential-not-a-secret` that the backend's test
mode knows, and the stand-ins `http://stand-in-ollama:11434` and `http://stand-in-qdrant:6333` (no key)
unless `GMSHIELD_OLLAMA_URL` or `GMSHIELD_QDRANT_URL` is set. A worker credential that is the test-mode
value is refused outside test mode.

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
docker build --target integration -t gmshield-worker-it .              # the integration tests, beside a Redis:
docker network create wk-it
docker run -d --name wk-it-redis --network wk-it --network-alias redis redis:8.10.3-alpine
docker run --rm --network wk-it -e GMSHIELD_MODE=test gmshield-worker-it   # exit 0: passed
```

The integration tests (`tests/integration`) run the lanes against a real Redis as an account with
exactly the production worker ACL, with the internal API stood in: claim, run, report and
acknowledge; a claim without an answer leaves the message pending; the take-over at start; a lane
that waits for the group and resumes after Redis is flushed; and the commands the account refuses.
Locally: `GMSHIELD_REDIS_HOST=127.0.0.1 GMSHIELD_REDIS_PORT=<port> uv run pytest -o addopts= tests/integration`.

The image check asserts that the work and health folders are owned by the worker, mode 0700 and
writable under `--read-only`, that Tesseract lists `por` and `eng` and that Docling converts a small
generated PDF, with Tesseract as its OCR engine, with no network. The production image runs as UID
10001, writes only to `/tmp`, `/var/lib/gmshield-worker/work` and `/run/gmshield-worker`, all three volumes and the last two
owned by the worker with mode 0700 (so its root filesystem can be read-only), has no `EXPOSE`, no
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

In production (Story 1.6) the worker joins `gmshield-work` (the backend at
`http://gmshield-backend:8080`, Redis, later Qdrant), `gmshield-model` (Ollama on the network's
gateway) and `gmshield-outbound` (the model provider), runs as 10001 with a read-only root
filesystem and every capability dropped, and finds the private authority's certificate at
`/run/gmshield/certs/ca.crt`. `~/gmshield-ci/secrets/worker.env` holds its Redis account
(`worker`, consumer-group commands on the two work streams only), the worker's credential (the same
value as in `backend.env`, written by `deploy/data-credentials.sh`; Dalton reruns it at the first
release that carries Story 1.12) and the Qdrant API key, which the worker does not use until Epic 2.

The pipeline reads [`.gmshield/pipeline.toml`](.gmshield/pipeline.toml) and builds the multi-stage
`Dockerfile`: `test` generates the client and runs ruff, mypy and pytest; `test-reports` holds the
JUnit results and Cobertura `coverage.xml`; `integration` runs the integration tests (`[test]`: the
pipeline's integration stage runs it on `develop` beside a throwaway Redis); the production image is
the last stage. The worker has
no port (`port = 0`), so the pipeline skips staging and DAST, and a deploy is healthy when the
container is still running after 20 s: the worker keeps retrying the backend rather than exiting.
