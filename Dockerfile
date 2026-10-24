# syntax=docker/dockerfile:1.7

FROM python:3.12-slim-bookworm AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libatspi2.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libfontconfig1 \
    libfreetype6 \
    libgbm1 \
    libglib2.0-0 \
    libgomp1 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libx11-6 \
    libx11-xcb1 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxkbcommon0 \
    libxrandr2 \
    libxrender1 \
    libxshmfence1 \
    libxss1 \
    libxtst6 \
    libxcb1 \
    zstd \
    xdg-utils \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
COPY render/src ./render/src
COPY embed/src ./embed/src
COPY index/src ./index/src
COPY serve/src ./serve/src

RUN uv sync --frozen --extra serve

# The pinned headless_shell is published for linux-x64 only — install_chrome()
# raises on anything else, which failed `docker compose up --build` outright on
# Apple Silicon. Skip it off amd64 instead: the image still serves search over a
# prebuilt index, and anything needing a renderer fails at run time with
# chrome.py's own message. Build with --platform linux/amd64 for a
# renderer-capable image. The mkdir keeps the runtime stage's COPY valid when
# the install is skipped.
ARG TARGETARCH
RUN mkdir -p /root/.cache/pixelrag/chrome && \
    arch="${TARGETARCH:-$(uname -m)}" && \
    case "$arch" in \
        amd64|x86_64) uv run pixelshot install-chrome ;; \
        *) echo "skipping install-chrome: no prebuilt headless_shell for $arch" ;; \
    esac


FROM python:3.12-slim-bookworm AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PIXELRAG_INDEX_DIR=/data/index \
    PIXELRAG_TILES_DIR=/data/tiles \
    PIXELRAG_ARTICLES_JSON=/data/index/articles.json

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libatspi2.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libfontconfig1 \
    libfreetype6 \
    libgbm1 \
    libglib2.0-0 \
    libgomp1 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libx11-6 \
    libx11-xcb1 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxkbcommon0 \
    libxrandr2 \
    libxrender1 \
    libxshmfence1 \
    libxss1 \
    libxtst6 \
    libxcb1 \
    xdg-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /root/.cache/pixelrag/chrome /root/.cache/pixelrag/chrome

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
COPY render/src ./render/src
COPY embed/src ./embed/src
COPY index/src ./index/src
COPY serve/src ./serve/src

EXPOSE 8000

# The image runs as root, so Chrome's setuid/namespace sandbox is unavailable and
# pixelshot disables it automatically (logging a warning). Chrome's own sandbox
# inside a container needs --cap-add=SYS_ADMIN or a seccomp profile; without those
# a non-root USER stops Chrome launching at all. Rely on container isolation here,
# and do not point this image at untrusted pages without it. See README.
CMD ["pixelrag", "serve", "--host", "0.0.0.0", "--port", "8000"]
