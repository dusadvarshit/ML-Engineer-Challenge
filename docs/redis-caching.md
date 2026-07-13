# Redis Caching Guide

## Best Use Case In This Repo

The best Redis use case in this repo is caching repeated inference results for identical images.

Why this fits the current codebase:

- The only implemented serving path is YOLO object detection.
- YOLO inference is the expensive step.
- Requests are image uploads, so the API can hash the raw image bytes and reuse results when the same image is submitted again.
- Redis settings already exist in `api/config.py`, so the repo was already intended to support this.

A secondary Redis use case would be rate limiting, since the challenge spec mentions it, but inference result caching gives the most immediate value.

## Current Repo State

Relevant files:

- `docker-compose.yml`
- `api/config.py`
- `api/routers/object_detection.py`
- `api/services/object_detection/yolo_service.py`

Right now:

- `docker-compose.yml` only runs `ml-api` and `nginx`
- `api/config.py` defines `REDIS_URL`
- Redis is not actually wired into the running application yet

## How To Add Redis With Docker Compose

Add a Redis service to `docker-compose.yml` and point the API container at it.

Example:

```yaml
services:
  ml-api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: ml-api
    restart: unless-stopped
    expose:
      - "8000"
    volumes:
      - ./models/artifacts:/opt/models/artifacts:ro
    environment:
      PORT: "8000"
      REDIS_URL: redis://redis:6379/0
      YOLO_MODEL_DIR: /opt/models/artifacts/object_detection/yolov8n/v1.0.0/pytorch
      NVIDIA_VISIBLE_DEVICES: all
      NVIDIA_DRIVER_CAPABILITIES: compute,utility
    depends_on:
      redis:
        condition: service_healthy
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-fsS", "http://127.0.0.1:8000/health"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 20s
    networks:
      - app-network

  redis:
    image: redis:7-alpine
    container_name: redis
    restart: unless-stopped
    expose:
      - "6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5
    networks:
      - app-network

  nginx:
    image: nginx:1.27-alpine
    container_name: nginx
    restart: unless-stopped
    depends_on:
      ml-api:
        condition: service_healthy
    ports:
      - "80:80"
    volumes:
      - ./nginx/default.conf:/etc/nginx/conf.d/default.conf:ro
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://127.0.0.1/health"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 10s
    networks:
      - app-network

networks:
  app-network:
    driver: bridge
```

## Important Docker Note

Inside Docker Compose, the API should not use `localhost` for Redis.

Use:

```text
redis://redis:6379/0
```

Do not use:

```text
redis://localhost:6379
```

`localhost` inside the `ml-api` container points back to the same container, not the Redis container.

## Python Dependency

Add Redis client support to `requirements.txt`:

```txt
redis[hiredis]>=5.0.0
```

## Suggested Cache Design

Cache detection results by hashing the uploaded image bytes.

Flow:

1. Read the uploaded image bytes.
2. Generate a SHA-256 hash from the bytes.
3. Use that hash as the Redis key.
4. If a cached result exists, return it immediately.
5. If not, run YOLO inference, store the response in Redis with a TTL, and return it.

This is a good fit because the same image can be submitted multiple times and object detection is the expensive operation.

## Example Cache Service

Create `api/services/cache_service.py`:

```python
import hashlib
import json

from redis import Redis

from api.config import settings

redis_client = Redis.from_url(settings.REDIS_URL, decode_responses=True)


def image_cache_key(prefix: str, image_bytes: bytes) -> str:
    digest = hashlib.sha256(image_bytes).hexdigest()
    return f"{prefix}:{digest}"


def get_cached_detection(image_bytes: bytes) -> list[dict] | None:
    key = image_cache_key("detect", image_bytes)
    payload = redis_client.get(key)
    return json.loads(payload) if payload else None


def set_cached_detection(image_bytes: bytes, detections: list[dict]) -> None:
    key = image_cache_key("detect", image_bytes)
    redis_client.setex(key, settings.MODEL_CACHE_TTL, json.dumps(detections))
```

## Where To Use It

Wire the cache check into `api/routers/object_detection.py` before calling the YOLO service.

Conceptually:

```python
cached = get_cached_detection(image_bytes)
if cached is not None:
    return ObjectDetectionResponse(
        detections=[ObjectDetection(**item) for item in cached]
    )

detections = yolo_prediction_service.predict(image_bytes)
set_cached_detection(
    image_bytes,
    [d.model_dump() for d in detections],
)
return ObjectDetectionResponse(detections=detections)
```

## Why This Is The Right First Cache

- It directly reduces repeated inference cost.
- It lowers response time for duplicate requests.
- It does not require changing the model service itself.
- It fits the existing FastAPI route structure.
- It uses the existing `MODEL_CACHE_TTL` setting already present in the repo.

## Secondary Redis Uses Later

After inference caching, Redis can also support:

- request rate limiting
- background job queueing
- model metadata caching
- temporary batch job status storage

## Run The Stack

After wiring Redis into Compose and the app:

```bash
docker compose up --build
```

## Summary

If you only add one Redis feature to this repo, make it object detection result caching keyed by image hash. That gives the clearest performance benefit with the smallest change to the current codebase.
