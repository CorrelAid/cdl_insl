# Builder stage
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy project metadata so uv can install dependencies
COPY pyproject.toml uv.lock README.md ./

# Use BuildKit caches for pip + uv cache and run uv sync to install pinned deps (torch and nvidia stuff already in base image)
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    UV_CACHE_DIR=/root/.cache/uv \
    UV_INDEX_STRATEGY=unsafe-best-match \
    uv venv --python /opt/conda/bin/python --system-site-packages && \
    uv sync --frozen --no-dev \
    --no-install-package torch \
    --no-install-package nvidia-nccl-cu12 \
    --no-install-package nvidia-cuda-runtime-cu12 \
    --no-install-package nvidia-cudnn-cu12 \
    --no-install-package nvidia-cublas-cu12 \
    --no-install-package nvidia-cufft-cu12 \
    --no-install-package nvidia-curand-cu12 \
    --no-install-package nvidia-cusolver-cu12 \
    --no-install-package nvidia-cusparselt-cu12 \
    --no-install-package nvidia-cusparse-cu12 \
    --no-install-package nvidia-nvtx-cu12 \
    --no-install-package nvidia-nvshmem-cu12

# Copy source
COPY src/ ./src/

# Install package in editable mode
RUN .venv/bin/pip install -e . --no-deps

# Download stanza models
RUN .venv/bin/python -c "import stanza; stanza.download('de', model_dir='/root/.stanza_models')"

# Copy notebooks
COPY --link marimo /app/notebooks

# Final stage
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# Copy only necessary files from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/notebooks /app/notebooks
COPY --from=builder /root/.stanza_models /app/.stanza_models

# user and permission setup
RUN useradd -m app_user && \
    chown -R app_user:app_user /app && \
    rm -rf /root/.cache /tmp/*
USER app_user

ENV STANZA_RESOURCES_DIR=/app/.stanza_models

EXPOSE 8080
CMD [".venv/bin/marimo","run","--host","0.0.0.0","-p","8080","--headless","--include-code","/app/notebooks/count_verbs_compare.py"]
