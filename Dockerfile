# GMshield worker. Stages the pipeline uses (GMshield-pipeline, "Repository contract"):
#   test          generates the internal API client, then ruff, mypy and pytest with coverage;
#                 a failure fails the build
#   test-reports  JUnit results and Cobertura coverage.xml, exported for the artifacts and SonarQube
#   integration   the integration tests (tests/integration), run as a container beside a throwaway
#                 Redis (.gmshield/pipeline.toml [test]) as an unprivileged user; built here, run offline
#   production    the image that runs (last stage)
# Images are pinned by version and digest: Python 3.14.10 on Debian trixie and uv 0.13.6, the
# newest releases at least two weeks old on 2026-11-13.

FROM ghcr.io/astral-sh/uv:0.13.6@sha256:af22a52f5eafef3d78e551599d405c02df61919d18028f0cea8eda93bc9986d0 AS uv

FROM python:3.14.10-slim-trixie@sha256:9aaa474097d094040649f2162a627c0049b408f52138d980916788d80f04802f AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    PATH=/opt/venv/bin:$PATH
WORKDIR /workspace
# The lockfile first: a source change reuses the dependency and model layers.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
# Docling's layout and table models, downloaded here once: nothing is downloaded at run time.
RUN docling-tools models download layout tableformer --output-dir /opt/docling-models \
    && chmod -R a+rX /opt/docling-models

FROM build AS test
RUN uv sync --locked --no-install-project
COPY . .
# The client is generated from contracts/internal.json on every build; it is never committed.
RUN python tools/generate_client.py \
    && ruff check \
    && ruff format --check \
    && mypy \
    && pytest -p no:cacheprovider \
        --junitxml=reports/junit/TEST-gmshield-worker.xml \
        --cov --cov-report=xml:reports/coverage.xml --cov-report=term

FROM scratch AS test-reports
COPY --from=test /workspace/reports/junit/ /junit/
COPY --from=test /workspace/reports/coverage.xml /coverage.xml

# The pipeline runs this image with GMSHIELD_MODE=test on an internal network (no internet) where Redis
# answers as redis:6379; the backend is the stand-in inside the tests. Everything is in the test stage.
FROM test AS integration
RUN groupadd --system --gid 10001 gmshield \
    && useradd --system --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin gmshield
ENV HOME=/tmp \
    PYTHONDONTWRITEBYTECODE=1
USER 10001:10001
CMD ["pytest", "-p", "no:cacheprovider", "-o", "addopts=--strict-markers", "tests/integration"]

FROM python:3.14.10-slim-trixie@sha256:9aaa474097d094040649f2162a627c0049b408f52138d980916788d80f04802f AS production
ARG GIT_SHA=unknown
LABEL org.opencontainers.image.title="gmshield-worker" \
      org.opencontainers.image.revision="${GIT_SHA}"
# Tesseract 5.5.0 from Debian trixie with its Portuguese and English data, pinned by package
# version, and zlib1g's current trixie security update (unpinned: the security archive keeps only
# the newest; on 2026-11-13 it fixed CVE-2026-76844 and CVE-2026-85091).
# pip is not needed to run, and is only more to patch.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        zlib1g \
        tesseract-ocr=5.5.0-1+b1 \
        libtesseract5=5.5.0-1+b1 \
        tesseract-ocr-por=1:4.1.0-2 \
        tesseract-ocr-eng=1:4.1.0-2 \
        tesseract-ocr-osd=1:4.1.0-2 \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* \
    && rm -rf /usr/local/lib/python3.14/site-packages/pip /usr/local/lib/python3.14/site-packages/pip-* \
        /usr/local/bin/pip /usr/local/bin/pip3 /usr/local/bin/pip3.14 /usr/local/bin/idle* \
    && groupadd --system --gid 10001 gmshield \
    && useradd --system --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin gmshield \
    && install -d -o 10001 -g 10001 -m 0700 /var/lib/gmshield-worker/work /run/gmshield-worker
COPY --from=build /opt/venv /opt/venv
COPY --from=build /opt/docling-models /opt/docling-models
# The application comes from the test stage, so the image exists only when the tests passed.
COPY --from=test /workspace/gmshield_worker /app/gmshield_worker
COPY --from=test /workspace/tests/image /app/tests/image
RUN python -m compileall -q /app/gmshield_worker
WORKDIR /app
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    HF_HUB_OFFLINE=1 \
    DOCLING_ARTIFACTS_PATH=/opt/docling-models
# Not root, and nothing written outside /tmp, the work folder and the health folder: the root
# filesystem can be read-only. All three are volumes so that `--read-only` works without any other
# option; Docker fills each new volume from the image's folder, owner and mode included.
# No EXPOSE: the worker opens no port. Health is a file that the supervisor keeps fresh.
USER 10001:10001
VOLUME ["/tmp", "/var/lib/gmshield-worker/work", "/run/gmshield-worker"]
# The check runs with an emptied environment (Story 1.15): it is a process of the worker's user, so with the worker's
# secrets in its environment a compromised parsing child could read them from /proc/<pid>/environ. It needs only the
# interpreter, the application and the default health file.
HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=3 \
    CMD ["/usr/bin/env", "-i", "PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin", "PYTHONPATH=/app", \
         "PYTHONDONTWRITEBYTECODE=1", "python", "-m", "gmshield_worker.health"]
CMD ["python", "-m", "gmshield_worker.main"]
