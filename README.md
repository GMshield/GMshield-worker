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
| redis (redis-py, released 2026-07-30) | 8.1.0 | `pyproject.toml` |
| attrs (used by the generated client) | 26.1.0 | `pyproject.toml` |
| pikepdf (the clean stage, Story 1.15; MPL-2.0, released 2026-10-11, qpdf 12.4.2 inside) | 10.17.0 | `pyproject.toml` |
| pypdfium2 (the triage stage, Story 1.16; already installed by Docling, released 2026-10-04) | 5.14.0 | `pyproject.toml` |
| pi-heif (the images stage's HEIC decoder, Story 1.21; released 2026-06-10). The binding is BSD-3-Clause; its wheel bundles libheif and libde265 (LGPL-3.0, decode only, no x265 encoder). Never pillow-heif, whose wheel bundles x265 (GPL) | 1.4.0 | `pyproject.toml` |
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
| `gmshield_worker/models/` | the model-client interface, the fake model and the Claude client |
| `gmshield_worker/lanes/` | the work lanes (Story 1.12): `consumer.py` (the loop both lanes share), `interactive.py`, `ingestion.py`, `renewal.py`, `streams.py` (Redis), `handlers.py` (handlers by kind) and `probe.py` (test mode only) |
| `gmshield_worker/ingestion/` | ingestion steps (Stories 1.15 to 1.21): `step.py` (the `ingestion` handler, dispatch on the stage), the stages' children `clean.py` (pikepdf), `images.py` (Pillow, pi-heif and pikepdf; Story 1.21), `triage.py` (pypdfium2), `layout.py` (Docling; pypdfium2 and Pillow for the page pictures), `ocr.py` (OpenCV, Docling and Tesseract) and `text.py` (standard library), `passages.py` (the in-process passages cutter), `sandbox.py` (the child's sandbox: environment, seccomp filter, deadline), `errors.py` (failure codes); `PIPELINE_VERSION` in `__init__.py` |
| `retrieval/`, `agents/` | empty until their stories (each `__init__.py` names it) |
| `contracts/internal.json` | the worker's copy of the backend's internal OpenAPI document |
| `tools/generate_client.py` | generates the client from `contracts/internal.json` |
| `tests/` | unit tests; `tests/image/` runs only inside the production image, `tests/integration/` only beside a Redis (the `integration` stage) |

## Start-up and behaviour

0. **Non-dumpable** (Story 1.15). `PR_SET_DUMPABLE 0`, before anything else; refused, the start
   stops with exit status 78 (`startup.refused`, reason `process.dumpable`).
1. **Settings.** Outside test mode a missing or invalid setting stops the start with exit status
   78 and one `startup.refused` line that names the settings, never a value.
2. **Work folder.** Everything inside `GMSHIELD_WORK_DIR` is removed; the folder itself stays, and
   a symbolic link is removed as a link, never followed. A removal that fails is logged by error
   type and the start continues. A folder that is a link or a file, or that cannot be created or
   read, stops the start with exit status 78 (`startup.refused`, reason `workdir.unusable`).
3. **Backend.** `GET /internal/v1/version` is retried with exponential backoff (1 s doubling up to
   30 s, with jitter of plus or minus 20%, 2 s per attempt) until the backend answers. Each failure
   is logged by attempt and error type only.
4. **Lanes.** After the first compatible answer the supervisor starts the interactive lane, then
   the ingestion lane; SIGTERM stops both.

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
  - Handlers: `ingestion` in every mode (Story 1.15, below) and the test-mode `probe` (it works
    `workMs`, fails its first `failTimes` attempts with `probe.failed`).
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
    reads a band across the middle of the page as detected and half a turn further, and the turn with more words
    read at 60 or more confidence wins (a tie keeps the detection; about 1 s more per page on roberto-server),
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
- **Shutdown.** SIGTERM or SIGINT stops the lanes (a running job is abandoned unreported and the
  backend takes it back when its claim lapses) and ends the process with exit status 0, whether it
  is retrying or ready.

## Settings

[`.env.example`](.env.example) lists them with their defaults and empty secrets, for a start by hand
(`docker run --env-file .env`).

| Setting | Meaning |
|---|---|
| `GMSHIELD_BACKEND_URL` | the backend's address, `http` or `https`, without credentials (required outside test mode) |
| `GMSHIELD_ANTHROPIC_API_KEY` | the model provider's key (required outside test mode; secret) |
| `GMSHIELD_WORKER_CREDENTIAL` | the worker's credential for `/internal/v1` (required outside test mode; secret): 32 to 512 characters of `A-Z a-z 0-9 . _ ~ + / = -`, the same value as the backend's |
| `GMSHIELD_REDIS_HOST`, `GMSHIELD_REDIS_PORT` | Redis (the host required outside test mode; port default 6379) |
| `GMSHIELD_REDIS_USERNAME`, `GMSHIELD_REDIS_PASSWORD` | the worker's Redis account (required outside test mode; `worker` in production; the password is secret) |
| `GMSHIELD_REDIS_TLS` | `true` (default): TLS verified, chain and host name, against `/run/gmshield/certs/ca.crt`; `false` only in the development setup |
| `GMSHIELD_INTERACTIVE_CAPACITY` | interactive jobs at once, 1 to 16 (default 2); the ingestion lane runs one |
| `GMSHIELD_WORK_DIR` | optional, absolute; default `/var/lib/gmshield-worker/work`, where production mounts the named volume `gmshield-worker-work` (on the encrypted volume) |
| `GMSHIELD_HEALTH_FILE` | for tests that run the worker outside the image only; default `/run/gmshield-worker/healthy` |
| `GMSHIELD_MODE` | unset, or `test` |

`GMSHIELD_MODE=test` is set only by the pipeline: it needs no secret, uses the fake model, uses
`http://stand-in-backend:8080` unless `GMSHIELD_BACKEND_URL` is set, Redis at `redis:6379` (or
`GMSHIELD_REDIS_HOST`/`GMSHIELD_REDIS_PORT`) without an account, a password or TLS, and the fixed,
public worker credential `gmshield-test-mode-worker-credential-not-a-secret` that the backend's test
mode knows. A worker credential that is the test-mode value is refused outside test mode.

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
