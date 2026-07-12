# Software Checklist

Scope: software components only.

Assumption: prototype the full system with only one model first, `YOLO`, and defer multi-model expansion until the software path works end to end.

## 1. Project Skeleton
- [x] Create the API app structure.
- [x] Create `api/main.py`.
- [x] Create basic routers.
- [x] Create basic service modules.
- [x] Create request/response schemas.
- [x] Create shared config handling for env-based settings.

## 2. Detection-Only Prototype
- [x] Wire one working `YOLO` detection model into the service.
- [x] Load the model from `models/artifacts/object_detection/...`.
- [x] Return a stable JSON response for detections.
- [ ] Expose final `POST /api/v1/detect` contract.
- [ ] Add model metadata to `GET /api/v1/models`.

## 3. Input Contract
- [x] Choose one primary inference input format for the prototype.
- [x] Use image upload as the prototype input path.
- [ ] Optionally support base64 JSON later if needed.
- [ ] Validate file type, size, decode success, and image dimensions.
- [ ] Reject malformed requests with clean error messages.

## 4. Batch Inference Flow
- [ ] Implement `POST /api/v1/batch`.
- [ ] Define batch request schema.
- [ ] Return a `job_id` immediately.
- [ ] Implement `GET /api/v1/batch/{job_id}/status`.
- [ ] Store batch job status and results.
- [ ] Start with YOLO batch detection only.

## 5. Async Worker Path
- [ ] Add a background worker process.
- [ ] Use Redis for queueing jobs.
- [ ] Define job payload schema.
- [ ] Add timeout and retry behavior.
- [ ] Handle worker failures without crashing API.

## 6. Caching
- [ ] Add Redis cache service.
- [ ] Cache model metadata.
- [ ] Optionally cache inference results using request hash.
- [ ] Define cache TTL rules.
- [ ] Add cache bypass controls for debugging.

## 7. Database and Logging
- [ ] Add PostgreSQL connection setup.
- [ ] Store inference logs.
- [ ] Store batch job metadata.
- [ ] Store latency, request id, model version, and status.
- [ ] Add structured logs with correlation ids.

## 8. Error Handling
- [ ] Add global exception handlers.
- [ ] Standardize error response format.
- [ ] Handle invalid image input.
- [x] Handle missing model artifacts.
- [x] Handle inference failures.
- [ ] Handle queue and database outages gracefully.

## 9. Model Routing
- [x] Keep routing simple for prototype.
- [x] Route detection requests to YOLO only.
- [ ] Add config-based default model selection.
- [ ] Design the code so later models can be added without changing the endpoint contract.
- [ ] Reserve fallback logic for later.

## 10. Health and Metadata
- [x] Implement a basic health endpoint.
- [ ] Implement dependency-aware health checks for API, Redis, DB, and model availability.
- [ ] Implement `GET /api/v1/models` for loaded model metadata and readiness.
- [ ] Separate liveness from readiness if possible.

## 11. Metrics and Monitoring
- [ ] Implement `GET /api/v1/metrics`.
- [ ] Expose request count, latency, error count, and queue depth.
- [ ] Expose model inference latency.
- [ ] Expose batch job counters.
- [ ] Add Prometheus integration.
- [ ] Leave Grafana optional until core flow works.

## 12. Security and Middleware
- [ ] Add request authentication stub or API key middleware.
- [ ] Add rate limiting.
- [ ] Add request id middleware.
- [ ] Add request/response logging middleware.
- [ ] Add upload size limits.

## 13. Testing
- [ ] Add unit tests for image validation and preprocessing.
- [ ] Add unit tests for model service and inference service.
- [ ] Add API tests for `/detect`, `/batch`, `/health`, and `/models`.
- [ ] Add integration tests for Redis and DB paths.
- [ ] Add one end-to-end test with YOLO loaded from artifacts.
- [ ] Add failure-path tests.

## 14. Containerization
- [ ] Create `Dockerfile` for API.
- [ ] Create worker container.
- [ ] Create `docker-compose.yml` with API, worker, Redis, and Postgres.
- [ ] Add health checks to containers.
- [ ] Add volume mounts for model artifacts.
- [ ] Make local startup one command.

## 15. Documentation
- [ ] Document the chosen request format for images.
- [ ] Document API endpoints and example requests.
- [ ] Document batch workflow and status polling.
- [ ] Document environment variables.
- [ ] Document how YOLO artifacts are loaded from disk.
- [ ] Document known limitations of the prototype.

## 16. Recommended Build Order
1. Basic FastAPI app.
2. YOLO local model loading.
3. Final `/detect` endpoint.
4. Input validation and error handling.
5. Redis and batch job flow.
6. Postgres logging.
7. Metrics and health checks.
8. Tests.
9. Docker Compose.
10. Documentation cleanup.
