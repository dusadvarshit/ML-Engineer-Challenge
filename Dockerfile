FROM nvcr.io/nvidia/tensorrt:24.10-py3 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl ca-certificates python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip setuptools wheel uv \
    && /opt/venv/bin/uv sync \
        --frozen \
        --no-dev \
        --no-install-project

FROM nvcr.io/nvidia/tensorrt:24.10-py3 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8000 \
    MODELS_ARTIFACTS_DIR=/opt/models/artifacts \
    YOLO_MODEL_DIR=/opt/models/artifacts/object_detection/yolov8n/v1.0.0/pytorch

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system app \
    && useradd --system --gid app --create-home --home-dir /home/app app \
    && mkdir -p /opt/models/artifacts \
    && chown -R app:app /app /opt/models /home/app

COPY --from=builder /opt/venv /opt/venv
COPY alembic ./alembic
COPY alembic.ini ./alembic.ini
COPY api ./api

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:8000/health" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
