# Testing strategy

## Fast feedback

The default CI-equivalent command is:

```bash
uv sync --frozen
uv run pytest -m 'not e2e and not slow'
```

Pytest enforces strict markers/configuration and reports coverage for `api/` with an 85% minimum. The CI workflow also checks that `uv.lock` matches the project metadata.

## Test layers

- **Unit tests** exercise API-key/rate-limit behavior, request logging, cache failures and reuse, Prometheus metric helpers, response models, and the YOLO/DETR/RetinaNet service paths using fakes rather than model weights.
- **Integration tests** use FastAPI's `TestClient`, mock startup dependencies, and verify route wiring, multipart detection/batch behavior, service error translation, model selection, and lifespan failures.
- **End-to-end tests** are marked `e2e` and `slow`. They can target an existing deployment with `E2E_BASE_URL`, or start Compose with `E2E_MANAGE_STACK=1`; they check Nginx, Prometheus, Grafana provisioning, and a real image request.

## Focused commands

```bash
uv run pytest tests/unit -m unit
uv run pytest tests/integration -m integration
uv run pytest tests/unit/test_metrics.py tests/unit/test_prometheus_metrics.py
uv run pytest -m e2e                         # existing stack via E2E_BASE_URL
E2E_MANAGE_STACK=1 uv run pytest -m e2e      # Compose-managed stack
```

The end-to-end path needs Docker, configured Compose secrets, and model artifacts. It is intentionally excluded from the fast CI suite. GPU/TensorRT validation remains environment-dependent and should be reported as blocked when unavailable.

## Coverage and failure-path expectations

Every implemented route should have success and failure coverage. Test doubles must model unavailable Redis/cache, malformed uploads, missing artifacts, model errors, and rejected credentials/rate limits. Real model loading belongs only in the explicitly marked smoke path; normal unit and integration tests must remain deterministic and not download weights.
