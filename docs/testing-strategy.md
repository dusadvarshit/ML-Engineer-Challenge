# Pytest Testing Strategy

## Goal

Define a practical pytest strategy for the current repository, not the full target architecture described in `README.md`.

Right now the highest-value test surface is:

- FastAPI app startup and health endpoints
- object detection routes in `api/routers/object_detection.py`
- YOLO inference behavior in `api/services/object_detection/yolo_service.py`
- request and response schema handling in `api/models/object_detection.py`

This document focuses on building a stable test pyramid around that code first, then extending coverage as classification, persistence, caching, and background jobs are added.

## Current Repo State

Relevant code:

- `api/main.py`
- `api/config.py`
- `api/routers/object_detection.py`
- `api/routers/classification.py`
- `api/services/object_detection/yolo_service.py`
- `api/models/object_detection.py`

Relevant test configuration:

- `pytest.ini` already defines `unit`, `integration`, `e2e`, and `slow` markers
- coverage is enforced with `--cov=api` and `--cov-fail-under=85`
- `tests/unit`, `tests/integration`, and `tests/e2e` exist but are empty

Important constraint:

- The current app loads the YOLO service during FastAPI lifespan startup, so tests must explicitly control or mock `yolo_prediction_service.load()` to avoid accidental model loading.

## Testing Principles

1. Default to fast, deterministic tests.
2. Mock model loading and inference in unit and most integration tests.
3. Use real image bytes where request validation matters.
4. Keep heavy model or Docker-dependent checks behind explicit markers such as `integration`, `e2e`, or `slow`.
5. Treat the API contract as stable: status codes, response schema, and error payloads should be asserted directly.

## Recommended Test Pyramid

### Unit Tests

Unit tests should be the majority of the suite and run on every local iteration and CI run.

Focus areas:

- `api/routers/object_detection.py`
  - accepts valid image uploads
  - rejects non-image uploads with `400`
  - rejects batches larger than `MAX_BATCH_SIZE`
  - returns `501` for `task=classify`
  - converts `FileNotFoundError` and `RuntimeError` from the service into `500`
  - preserves filenames in batch responses
- `api/services/object_detection/yolo_service.py`
  - `predict()` unwraps the first batch result
  - `predict_batch([])` returns an empty list
  - `_decode_image()` converts bytes into RGB PIL images
  - `_resolve_model_path()` selects the first `*.pt` or `*.pth` file
  - `_resolve_model_path()` raises `FileNotFoundError` when no checkpoint exists
  - `_normalize_result()` converts Ultralytics-like result objects into `ObjectDetection`
  - `_warm_up()` only runs once
  - `_load_model()` caches the loaded model
- `api/main.py`
  - root endpoint returns the configured title
  - health endpoint returns `{"status": "ok"}`
  - lifespan calls `yolo_prediction_service.load()`
- `api/models/object_detection.py`
  - enum values are accepted and serialized correctly
  - response models validate the expected payload shape

Unit tests should not:

- require a real YOLO checkpoint
- import Ultralytics as a hard dependency
- depend on Docker, Redis, Postgres, or network access

### Integration Tests

Integration tests should validate the FastAPI stack with the app mounted and HTTP requests flowing through real routes.

Focus areas:

- `POST /api/v1/detect`
  - success response shape with a mocked service
  - error translation from service exceptions
- `POST /api/v1/batch`
  - multiple file upload flow
  - batch response ordering and filenames
- FastAPI lifespan behavior
  - app startup succeeds when `load()` is mocked
  - startup failure behavior is visible if `load()` raises

Keep these tests lightweight by patching the singleton `yolo_prediction_service` methods rather than loading real model artifacts.

### End-to-End Tests

End-to-end tests should be minimal and reserved for environment-level validation.

Recommended scope:

- run the API in Docker Compose
- verify `/health`
- verify one happy-path `/api/v1/detect` request through the deployed container stack
- optionally verify Nginx routing if the request is meant to go through `nginx`

These should be marked `@pytest.mark.e2e` and excluded from default local runs unless explicitly requested.

## Proposed Test Layout

```text
tests/
├── conftest.py
├── fixtures/
│   ├── sample_images.py
│   └── yolo_stubs.py
├── unit/
│   ├── test_main.py
│   ├── test_object_detection_router.py
│   ├── test_yolo_service.py
│   └── test_object_detection_models.py
├── integration/
│   ├── test_detect_endpoint.py
│   ├── test_batch_endpoint.py
│   └── test_lifespan.py
└── e2e/
    └── test_api_smoke.py
```

If the repo stays small, `tests/fixtures/` can be skipped and fixture helpers can live in `tests/conftest.py`. Split them out only when the fixture file becomes noisy.

## Fixture Strategy

Create fixtures that remove model and filesystem coupling from the suite.

Recommended fixtures:

- `app`
  - returns the FastAPI app with startup-time model loading patched
- `client`
  - `TestClient(app)` for route tests
- `sample_image_bytes`
  - tiny in-memory JPEG or PNG payload generated with Pillow
- `sample_upload_file`
  - helper for multipart request construction
- `mock_yolo_service`
  - patches `predict`, `predict_batch_from_bytes`, and `load`
- `temp_model_dir`
  - temporary checkpoint directory for `_resolve_model_path()` tests
- `fake_yolo_result`
  - lightweight object exposing `.boxes.xyxy`, `.boxes.conf`, and `.boxes.cls`

Use `monkeypatch` or `pytest-mock` consistently. Do not mix styles randomly inside the same module.

## Mocking Strategy

### Router Tests

Patch the singleton imported by the router:

- `api.routers.object_detection.yolo_prediction_service.predict`
- `api.routers.object_detection.yolo_prediction_service.predict_batch_from_bytes`

This keeps the tests at the HTTP boundary and avoids leaking service internals into route assertions.

### Lifespan Tests

Patch:

- `api.main.yolo_prediction_service.load`

This prevents real startup model loading and makes it easy to assert that lifespan initialization happens exactly once.

### YOLO Service Unit Tests

Patch:

- `api.services.object_detection.yolo_service.settings.YOLO_MODEL_DIR`
- the imported `ultralytics.YOLO` constructor path indirectly by controlling `_load_model()`

Prefer testing service methods with fake model objects instead of importing the real ML stack. The point is to verify control flow, normalization, error handling, and path resolution.

## Coverage Strategy

The repo currently enforces `85%` coverage for `api/`.

Recommended target by phase:

1. Phase 1
   - keep global coverage at `85%`
   - get `api/routers/object_detection.py` and `api/services/object_detection/yolo_service.py` above `90%`
2. Phase 2
   - add classification tests as endpoints are implemented
   - raise global threshold to `90%` once empty or placeholder modules are replaced with real code

This is a better fit than forcing `90%` globally today, because the classification surface is still mostly a stub.

## Marker Usage

Use the existing markers intentionally:

- `@pytest.mark.unit`
  - pure Python logic, route validation, schema tests
- `@pytest.mark.integration`
  - FastAPI app wiring and request flow
- `@pytest.mark.e2e`
  - Docker or deployed-stack smoke checks
- `@pytest.mark.slow`
  - any test that loads a real model, uses large images, or exercises startup on production-like artifacts

Suggested commands:

```bash
pytest -m "unit"
pytest -m "integration"
pytest -m "not e2e and not slow"
pytest --cov=api --cov-report=term-missing
```

## Initial Test Backlog

Implement tests in this order:

1. `tests/unit/test_object_detection_router.py`
   - highest API risk and fastest coverage gain
2. `tests/unit/test_yolo_service.py`
   - highest logic density in the repo
3. `tests/unit/test_main.py`
   - startup and basic endpoint coverage
4. `tests/integration/test_detect_endpoint.py`
   - end-to-end request handling inside the app
5. `tests/integration/test_batch_endpoint.py`
   - multipart batch behavior and response ordering
6. `tests/e2e/test_api_smoke.py`
   - one Docker-level smoke test when container workflow is stable

## Example Assertions To Prioritize

Examples that matter more than implementation details:

- response status codes for each failure mode
- exact error messages where the API contract is user-facing
- batch size enforcement
- filename preservation in batch results
- stable serialization of `ObjectDetection`
- model path resolution precedence
- startup warmup idempotence

Avoid brittle assertions on:

- exact internal call ordering unless it is behaviorally important
- full FastAPI-generated OpenAPI documents
- object identity where value equality is enough

## CI Recommendation

When CI is added, keep the default pipeline narrow:

1. install dev dependencies
2. run `pytest -m "not e2e and not slow"`
3. publish coverage output

Add separate opt-in jobs later for:

- real-model smoke tests
- Docker Compose e2e tests
- performance or load tests from `scripts/testing/`

## Definition Of Done

The pytest strategy is in good shape when:

- every implemented route has happy-path and failure-path coverage
- startup behavior is tested without requiring real model artifacts
- YOLO service normalization and checkpoint resolution are covered by unit tests
- local default test runs stay fast
- CI can enforce the current `85%` threshold reliably

## Summary

For this repo, pytest should center on fast unit tests around the object detection router and YOLO service, with a thin layer of FastAPI integration tests and only a very small e2e surface. The main discipline is to mock model loading aggressively so tests validate API behavior without turning the suite into a heavyweight ML runtime check.
