FROM ghcr.io/astral-sh/uv:bookworm-slim

WORKDIR /app

# Copy project metadata so uv can install dependencies
COPY pyproject.toml uv.lock README.md ./

# Use BuildKit caches for pip + uv cache and run uv sync to install pinned deps (including torch)
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    UV_CACHE_DIR=/root/.cache/uv \
    UV_INDEX_STRATEGY=unsafe-best-match \
    uv sync --frozen --no-dev

# Copy source and install editable package
COPY src/ ./src/
RUN uv run pip install -e .

# Download stanza models
RUN uv run python -c "import stanza; stanza.download('de', model_dir='/root/.stanza_models')"

# Copy notebooks (or mount at runtime)
COPY --link marimo /app/notebooks

# user and permission setup
RUN useradd -m app_user && \
    mkdir -p /app/.stanza_models && \
    cp -r /root/.stanza_models/* /app/.stanza_models/ && \
    chown -R app_user:app_user /app /app/.venv
USER app_user

ENV STANZA_RESOURCES_DIR=/app/.stanza_models

EXPOSE 8080
CMD ["uv","run","marimo","run","--host","0.0.0.0","-p","8080","--headless","--include-code","/app/notebooks/"]
