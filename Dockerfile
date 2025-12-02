FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy project metadata so uv can install dependencies
COPY pyproject.toml uv.lock README.md ./

# Use BuildKit caches for pip + uv cache and run uv sync to install pinned deps (torch already in base image)
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

# Copy source and install editable package
COPY src/ ./src/

# Download stanza models using the venv created by uv sync
RUN .venv/bin/python -c "import stanza; stanza.download('de', model_dir='/root/.stanza_models')"

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
CMD [".venv/bin/marimo","run","--host","0.0.0.0","-p","8080","--headless","--include-code","/app/notebooks/"]
