# ML Engineer Challenge — Agentic Development Checklist

Use this file as the execution backlog for completing the challenge. Work from top to bottom unless a task explicitly says it can run in parallel. An agent must update the relevant checkbox only after running the stated validation and recording any meaningful result in the linked documentation.

## Operating Rules

- [ ] Before each work package, inspect the affected code, tests, and `README.md`; do not assume this checklist is more current than the code.
- [ ] Keep changes scoped to one work package. Do not refactor unrelated components.
- [ ] Add or update tests with every behavior change.
- [ ] Run the work-package validation command before marking an item complete.
- [ ] Record design decisions, hardware details, benchmark settings, and known limitations in `README.md` or `docs/`.
- [ ] Never commit model weights, datasets, secrets, or runtime database files. `models/artifacts/` is intentionally ignored by Git.
- [ ] If blocked by unavailable GPU, TensorRT, Redis, PostgreSQL, or Docker, complete the code and tests that can run locally, then add a concise blocker note under the work package.

## Definition of Done

A work package is done only when all applicable conditions are true:

- [ ] Implementation is type-hinted, documented where behavior is non-obvious, and follows existing project conventions.
- [ ] Unit and integration tests cover success and failure paths.
- [ ] Relevant tests pass.
- [ ] Configuration is environment-driven; no secrets or machine-specific paths are added.
- [ ] User-facing behavior and operational steps are documented.

## Current Baseline

### Completed ML work

- [x] Fine-tuned pretrained ResNet50, EfficientNet-B0, and ViT-B/16 on Tiny ImageNet.
- [x] Used custom augmentation: random resized crop, horizontal flip, and colour jitter.
- [x] Used mixed precision on CUDA, gradient clipping, and cosine learning-rate scheduling.
- [x] Recorded Tiny ImageNet validation metrics and native PyTorch latency for each classifier.
- [x] Stored classification checkpoints and per-run metadata in `models/artifacts/classification/<model>/v1.0.0/pytorch/`.
- [x] Prepared pretrained YOLOv8n, DETR ResNet-50, and RetinaNet ResNet-50 FPN detection artifacts.
- [x] Added reference-notebook code to evaluate detection models on COCO validation data.

### Known baseline issues

- [ ] Fix or deliberately preserve the naming inconsistency between `vit_b16` and `vit_b_16`; document the canonical artifact name and update all consumers.
- [ ] Regenerate classification summary JSON and CSV because they still reference the old timestamped checkpoint paths.
- [ ] Verify every checklist item below against the current code before starting it; older prototype entries may be stale.

---

# Work Queue

## WP-0 — Establish a Reproducible Baseline

**Goal:** Make the repository runnable and capture its actual starting state.

- [ ] Create or update `docs/baseline-status.md` with Python version, OS, GPU/CUDA availability, Docker availability, and known external-service constraints.
- [ ] Install locked dependencies with the repository-supported command.
- [ ] Run the fast test suite: `pytest -m "not e2e and not slow"`.
- [ ] Run formatting, linting, and type checks if configured; otherwise add the minimal tooling configuration.
- [ ] Fix only baseline failures that block all further work, or document each blocker with its command output summary.

**Acceptance evidence:** passing test output or a documented, reproducible blocker in `docs/baseline-status.md`.

## WP-1 — Make Model Artifacts Reproducible and Deployable

**Goal:** Ensure model files, metadata, and loading contracts are consistent.

- [ ] Define one artifact manifest schema: model name, task, version, framework, checkpoint path, input size, preprocessing, classes, metrics, source weights, and checksum.
- [ ] Generate manifests for ResNet50, EfficientNet-B0, ViT-B/16, YOLOv8n, DETR, and RetinaNet.
- [ ] Implement validation that reports a clear error for a missing or incompatible artifact.
- [ ] Update model loaders and registry code to read manifests rather than hard-coded paths where practical.
- [ ] Add model cards for the selected classification and detection deployment candidates in `docs/model_cards/`.
- [ ] In classification model cards, disclose ImageNet-pretraining overlap with Tiny ImageNet and state that reported validation metrics may be optimistic.
- [ ] Add tests for manifest parsing, model lookup, missing artifacts, and incompatible metadata.

**Validation:** `pytest tests/unit -k "model or registry or service"`.

## WP-2 — Complete Optimization, Export, and Benchmarking

**Goal:** Meet the README requirement for INT8, ONNX, TensorRT, and cross-format performance comparisons.

- [ ] Select the classification and detection models that will be exported for deployment; record the selection rationale.
- [ ] Export selected PyTorch models to ONNX with fixed, documented input shapes and dynamic-batch behavior only if supported and tested.
- [ ] Validate each ONNX artifact by comparing its outputs with PyTorch on representative inputs and defined tolerances.
- [ ] Produce INT8 artifacts. Use calibration data for static quantization when required and document the calibration split.
- [ ] Build TensorRT engines in a supported GPU environment; record TensorRT version, GPU model, precision, and input shapes.
- [ ] Implement a single benchmark runner that measures warm-up, latency percentiles, throughput, memory, batch size, input shape, and hardware metadata consistently.
- [ ] Benchmark PyTorch, INT8, ONNX, and TensorRT variants of every selected model.
- [ ] Write comparison reports to `benchmarks/` and summarize the result in `README.md`.
- [ ] Add a performance-regression test with realistic, environment-appropriate thresholds.

**Validation:** export smoke tests pass; benchmark reports exist; each artifact can load and infer one valid image.

## WP-3 — Finish Multi-Model Inference API

**Goal:** Serve classification and detection through stable, documented FastAPI contracts.

- [ ] Finalize `POST /api/v1/classify` request and response schemas, including model-version selection and confidence output.
- [ ] Finalize `POST /api/v1/detect` request and response schemas, including boxes, labels, scores, and model version.
- [ ] Implement `GET /api/v1/models` with loaded-model metadata, available versions, artifact status, and readiness.
- [ ] Implement model selection through configuration and request parameters; reject unavailable models with a clear error.
- [ ] Add a fallback policy for model-load or inference failures and document its behavior.
- [ ] Implement shared image validation: content type, maximum upload size, decode success, dimensions, and decompression-bomb protection.
- [ ] Standardize error responses and global exception handling.
- [ ] Ensure no endpoint requires a local absolute path from the client.
- [ ] Add OpenAPI examples for every endpoint and error contract.

**Validation:** `pytest tests/unit tests/integration -k "classification or detection or models or api"`.

## WP-4 — Implement Batch and Asynchronous Processing

**Goal:** Support large batch inference without blocking API workers.

- [ ] Define `POST /api/v1/batch` schema and a persistent job model.
- [ ] Implement `GET /api/v1/batch/{job_id}/status` with queued, running, succeeded, failed, and expired states.
- [ ] Add a worker service and Redis-backed queue.
- [ ] Implement idempotency for batch submission and safe retry behavior.
- [ ] Store result references and per-item failures without losing successful items.
- [ ] Add timeouts, retry limits, and clear error behavior for queue or worker outages.
- [ ] Add integration tests using disposable Redis and a mocked model.

**Validation:** submit a multi-image batch, poll it to completion, and verify both success and failure cases.

## WP-5 — Persistence, Caching, and Observability

**Goal:** Make the service diagnosable and operationally useful.

- [ ] Add PostgreSQL migrations for inference logs and batch-job metadata.
- [ ] Persist request ID, timestamp, model/version, status, latency, input metadata, and error category. Do not store image bytes by default.
- [ ] Implement structured JSON logging and correlation/request IDs across API and worker paths.
- [ ] Implement Redis caching for model metadata and optionally request-hash inference results with explicit TTL and bypass rules.
- [ ] Implement `/api/v1/metrics` with request counts, error counts, API latency, model latency, cache metrics, and queue depth.
- [ ] Add dependency-aware readiness checks for models, Redis, and PostgreSQL; keep liveness separate.
- [ ] Add Prometheus scrape configuration and a minimal Grafana dashboard.
- [ ] Add tests for database errors, cache failures, metrics, and readiness degradation.

**Validation:** run the local stack, make an inference request, and confirm logs, database row, Prometheus metric, and health/readiness behavior.

## WP-6 — Security and Resilience

**Goal:** Protect the API and make failures predictable.

- [ ] Add API-key or equivalent authentication middleware using environment-provided credentials.
- [ ] Add tier-aware rate limiting with Redis-backed state where applicable.
- [ ] Enforce upload-size limits before full processing.
- [ ] Add security headers and restrict CORS to documented origins.
- [ ] Validate external URLs are not accepted as image inputs unless SSRF protections are implemented.
- [ ] Add graceful-degradation behavior for unavailable model, cache, queue, database, and monitoring dependencies.
- [ ] Add tests for unauthorized, rate-limited, malformed, oversized, and dependency-failure requests.

**Validation:** security and failure-path tests pass with no secret values in logs or responses.

## WP-7 — Test Coverage and CI

**Goal:** Provide confidence for production changes.

- [ ] Inventory test coverage by API route, service, worker, export code, and error path.
- [ ] Add unit tests for preprocessing, model services, cache logic, configuration, and request logging.
- [ ] Add integration tests for API, database, Redis, model loading, and batch lifecycle.
- [ ] Add one end-to-end Docker Compose smoke test using a lightweight model path.
- [ ] Add performance and memory profiling scripts under `scripts/testing/`.
- [ ] Configure CI to run lockfile validation, fast tests, linting, and type checks.
- [ ] Ensure slow, GPU, and end-to-end tests are correctly marked and documented.
- [ ] Report coverage for critical paths and work toward the README target of at least 85 percent.

**Validation:** CI-equivalent command passes from a clean checkout.

## WP-8 — Containerization and Deployment

**Goal:** Make local and production deployment repeatable.

- [ ] Review Dockerfiles for multi-stage builds, non-root runtime users, pinned dependencies, small images, and layer caching.
- [ ] Provide `docker-compose.yml` for API gateway, ML API, worker, Redis, PostgreSQL, Prometheus, and optional Grafana.
- [ ] Provide `docker-compose.prod.yml` with production-safe configuration and resource limits.
- [ ] Add health checks, persistent volumes, service dependencies, and artifact mounts.
- [ ] Configure the API gateway for routing, upload limits, authentication forwarding, and rate-limit integration.
- [ ] Document one-command local startup, teardown, scaling the worker, and GPU/TensorRT prerequisites.
- [ ] Run the end-to-end stack smoke test.

**Validation:** `docker compose up` starts all required services, readiness becomes healthy, and one classify plus one detect request succeeds.

## WP-9 — Documentation and Submission Readiness

**Goal:** Produce a clear, evidence-backed submission.

- [ ] Update `README.md` with architecture diagram, setup, environment variables, API examples, benchmark table, monitoring URLs, and known limitations.
- [ ] Write `docs/technical-readme.md` with model rationale, optimization decisions, benchmark methodology/results, system architecture, security, and scaling decisions.
- [ ] Publish API usage examples or a Postman collection derived from the current OpenAPI schema.
- [ ] Confirm model cards, benchmark reports, and deployment instructions are linked from the README.
- [ ] Review every README deliverable against the original challenge requirements.
- [ ] Run the full supported test suite and deployment smoke test one final time.
- [ ] Record final limitations, including any unavailable GPU/TensorRT verification.

**Validation:** a new developer can follow the README from a clean checkout and reproduce the documented API and benchmark workflow.

---

## Copy-Ready Agent Briefs

Use exactly one brief per agent turn. Replace only the bracketed values. The agent must not start a later work package unless its dependencies are complete or explicitly marked blocked.

### Global Completion Contract

Every agent brief inherits these requirements:

- Inspect the listed files before editing and preserve unrelated user changes.
- Make the smallest implementation that satisfies the work package.
- Use existing patterns and dependencies before adding new ones.
- Add or update focused tests for changed behavior.
- Run the stated validation. If it cannot run, state the exact command, failure, and blocker.
- Update this checklist with completed checkboxes and append a handoff record under the active work package.
- Do not commit, push, download large model files, or change deployment infrastructure unless the assigned brief explicitly requires it.

### Agent Brief: WP-0 Baseline

```text
Work package: WP-0 — Establish a Reproducible Baseline
Objective: establish the actual runnable state of the repository without changing product behavior.
Inspect first: README.md, pyproject.toml, uv.lock, pytest.ini, .github/workflows/, existing tests, Docker configuration.
Allowed scope: baseline documentation, minimal test or tooling fixes only when they block all subsequent work.
Do not: implement new API features, refactor services, change model artifacts, or suppress failing tests.
Tasks: record environment capabilities in docs/baseline-status.md; install dependencies using the documented locked workflow; run the fast test suite; identify configured lint and type-check commands; document every reproducible blocker.
Validation: pytest -m "not e2e and not slow" and every configured lightweight quality command.
Done when: baseline-status.md contains commands, results, platform capabilities, and unresolved blockers; no unverified completion claim remains.
Handoff: list changed files, commands and outcomes, blockers, and the single best next action.
```

### Agent Brief: WP-1 Artifacts and Model Cards

```text
Work package: WP-1 — Make Model Artifacts Reproducible and Deployable
Dependency: WP-0 baseline complete or its blockers understood.
Objective: make local model artifacts discoverable, validated, and documented without committing weights.
Inspect first: models/export_utils.py, models/export_*.py, api/services/, api/models/, current artifact directories, tests/unit/test_*model*.py, tests/unit/test_*registry*.py.
Allowed scope: artifact manifest schema, loaders and registry integration, model-card documentation, focused tests.
Do not: move or delete existing checkpoint files; hard-code absolute user paths; require a GPU to run unit tests.
Tasks: choose one canonical ViT artifact name; define a versioned manifest schema; create manifests or generation logic for all six models; validate paths, checksums when files exist, task compatibility, preprocessing, and class metadata; update loaders to use manifests where practical; write model cards for deployment candidates; regenerate stale summary paths.
Validation: pytest tests/unit -k "model or registry or service" plus a manifest-load smoke test for each artifact available locally.
Done when: a missing or incompatible artifact yields an actionable error; canonical names are documented; model provenance and Tiny ImageNet overlap caveat are recorded.
Handoff: include canonical artifact naming decision and remaining missing-artifact constraints.
```

### Agent Brief: WP-2 Export and Benchmark

```text
Work package: WP-2 — Complete Optimization, Export, and Benchmarking
Dependency: WP-1 artifact contract complete. GPU or TensorRT may be a documented external blocker.
Objective: implement repeatable ONNX, INT8, TensorRT, and apples-to-apples benchmarking for the selected deployment models.
Inspect first: README.md model requirements, models/export_common.py, models/export_utils.py, models/export_*.py, artifact manifests, scripts/testing/.
Allowed scope: export scripts, calibration and benchmark code, benchmark reports, export tests, relevant docs.
Do not: claim an export or benchmark completed without its generated artifact/report; compare formats under different input or timing settings; silently fall back from TensorRT.
Tasks: document selected models and rationale; define a common benchmark protocol; export ONNX; compare ONNX and PyTorch outputs within declared tolerances; create INT8 artifacts with documented calibration; build TensorRT engines when the environment supports it; benchmark every available format; report unavailable formats as blocked.
Validation: each generated artifact loads and infers one representative image; report includes hardware, software versions, warm-up, batch size, input shape, latency percentiles, throughput, memory, and accuracy or output-equivalence results.
Done when: benchmarks/ contains reproducible reports and README.md links to their conclusions.
Handoff: list generated artifacts and clearly separate verified results from environment-blocked work.
```

### Agent Brief: WP-3 Synchronous Multi-Model API

```text
Work package: WP-3 — Finish Multi-Model Inference API
Dependency: WP-1 complete for model discovery; use mocks if a real artifact is unavailable.
Objective: provide stable, validated synchronous classify, detect, models, health, and error contracts.
Inspect first: api/main.py, api/routers/, api/models/, api/services/, api/config.py, api/middleware/, existing API tests and OpenAPI output.
Allowed scope: synchronous API endpoints, shared validation, model selection, exception handling, API tests, OpenAPI examples, endpoint docs.
Do not: implement queue workers, database migrations, authentication, or unrelated UI work; expose filesystem paths or raw internal exceptions in responses.
Tasks: finalize schemas; implement classify and detect routes; expose model metadata/readiness; add configuration-based model and version selection; validate uploads before inference; define consistent error envelopes; document fallback behavior and OpenAPI examples.
Validation: pytest tests/unit tests/integration -k "classification or detection or models or api"; exercise success, invalid image, unavailable model, and inference-failure paths.
Done when: endpoints have stable documented contracts and tests prove no client needs a local path.
Handoff: include endpoint examples and any artifact-dependent behavior that was mocked.
```

### Agent Brief: WP-4 Batch and Worker

```text
Work package: WP-4 — Implement Batch and Asynchronous Processing
Dependency: WP-3 synchronous inference contract complete.
Objective: accept batches quickly, process them asynchronously, and expose durable job status.
Inspect first: api/worker.py, api/database.py, api/services/cache_service.py, routers, schemas, docker-compose.yml, integration tests.
Allowed scope: batch schemas/routes, worker and queue integration, job state persistence, retries/timeouts, integration tests, docs.
Do not: change synchronous endpoint contracts; use in-memory state as the production source of truth; lose partial item failures.
Tasks: implement submit and status endpoints; define job state transitions and idempotency; queue model work through Redis; persist jobs/results; handle retry, timeout, worker crash, and expired result behavior.
Validation: with disposable Redis and mocked model inference, submit a mixed-validity multi-image batch, poll to terminal state, and verify idempotent resubmission and partial failure reporting.
Done when: API workers do not perform long batch inference inline and every terminal state has a documented response.
Handoff: state the queue backend, retry policy, and schema/migration status.
```

### Agent Brief: WP-5 Persistence, Caching, and Observability

```text
Work package: WP-5 — Persistence, Caching, and Observability
Dependency: WP-3 complete; WP-4 preferred for batch metrics.
Objective: make every inference and dependency state observable without retaining sensitive image content by default.
Inspect first: api/database.py, alembic/, api/metrics.py, api/services/, middleware, monitoring/, docker-compose.yml, related tests.
Allowed scope: migrations, logging, cache behavior, metrics, health/readiness, Prometheus/Grafana configuration, tests, operational docs.
Do not: log image bytes, credentials, authorization headers, or raw client data; make liveness fail due to an optional dependency.
Tasks: persist inference and job metadata; add structured correlation-aware logs; implement cache keys, TTLs, bypass, and failure behavior; publish metrics; separate liveness from readiness; configure Prometheus and minimal Grafana visibility.
Validation: local stack or isolated service tests demonstrate a request log, database row, metric changes, cache behavior, and degraded readiness under a failed dependency.
Done when: operators can identify request, model/version, latency, outcome, and dependency health from supported observability surfaces.
Handoff: include migration revision, metric names, cache TTL policy, and any stack prerequisites.
```

### Agent Brief: WP-6 Security and Resilience

```text
Work package: WP-6 — Security and Resilience
Dependency: WP-3 complete; WP-5 preferred for Redis-backed rate limiting.
Objective: protect inference endpoints and make dependency failures safe and predictable.
Inspect first: api/config.py, middleware, routers, validators, docker gateway configuration, security-related tests.
Allowed scope: authentication, rate limiting, upload restrictions, CORS/security headers, resilience behavior, tests, docs.
Do not: add hard-coded keys; log secrets; accept external image URLs without explicit SSRF controls; weaken validation to make tests pass.
Tasks: implement environment-sourced authentication; configure rate limits by tier; enforce upload limits; add CORS and security headers; define graceful errors for unavailable services; test unauthorized, oversized, malformed, rate-limited, and dependency-outage cases.
Validation: security/failure-path tests pass and response/log inspection confirms no secrets or internal stack traces escape.
Done when: all externally reachable failure modes have a stable safe response and documented operator behavior.
Handoff: state configuration variables without their secret values.
```

### Agent Brief: WP-7 Tests and CI

```text
Work package: WP-7 — Test Coverage and CI
Dependency: may begin after WP-0 and should be revisited after every later package.
Objective: make critical behavior continuously verifiable in a clean environment.
Inspect first: tests/, pytest.ini, pyproject.toml, uv.lock, .github/workflows/, scripts/testing/.
Allowed scope: tests, fixtures, test configuration, CI workflow, coverage/quality tooling, test documentation.
Do not: lower coverage gates or skip tests solely to get green CI; make GPU or network dependencies mandatory for fast CI.
Tasks: map coverage gaps; add focused unit/integration/e2e tests; mark slow/GPU/e2e tests correctly; configure CI for locked install, lock check, fast tests, lint/type checks; add lightweight performance/memory scripts.
Validation: run the CI-equivalent command from a clean environment or document the exact external limitation.
Done when: CI reliably exercises critical paths and the test strategy explains what runs where.
Handoff: include test counts, markers, CI commands, and remaining coverage gaps.
```

### Agent Brief: WP-8 Containerization and Deployment

```text
Work package: WP-8 — Containerization and Deployment
Dependency: WP-3 complete; WP-4 and WP-5 complete for the full stack.
Objective: provide secure, repeatable local and production-oriented service orchestration.
Inspect first: Dockerfile, docker-compose.yml, entrypoint.sh, nginx/, monitoring/, scripts/docker/, configuration docs.
Allowed scope: container definitions, compose files, gateway config, health checks, resource settings, deployment documentation, stack smoke tests.
Do not: bake secrets, datasets, or checkpoint weights into images; run containers as root without documented justification; make optional Grafana block core API startup.
Tasks: verify multi-stage/non-root builds; define API, worker, Redis, PostgreSQL, Prometheus, and optional Grafana services; add health checks, volumes, resource limits, networks, and artifact mounts; create production overrides; document startup, scaling, teardown, and GPU prerequisites.
Validation: docker compose up starts required services, readiness is healthy, and classify plus detect requests succeed through the intended gateway.
Done when: a fresh developer can start and verify the stack using documented commands.
Handoff: list images/services tested and any host-specific GPU/TensorRT limitations.
```

### Agent Brief: WP-9 Documentation and Submission Review

```text
Work package: WP-9 — Documentation and Submission Readiness
Dependency: all intended implementation packages complete or transparently marked blocked.
Objective: deliver a reproducible, honest, reviewer-friendly submission.
Inspect first: README.md, this checklist, docs/, benchmarks/, model cards, OpenAPI output, CI configuration, deployment files.
Allowed scope: documentation, examples, report collation, final requirement audit, small documentation-only corrections.
Do not: invent benchmark numbers, claim unrun tests passed, conceal environment limitations, or introduce broad code changes.
Tasks: map every README challenge requirement to evidence; update architecture/setup/API/benchmark/monitoring/security/scaling documentation; publish technical readme and endpoint examples; link reports and model cards; record final limitations; run all supported checks.
Validation: follow README instructions from a clean checkout or have an independent agent do so; complete the requirement-to-evidence audit.
Done when: a reviewer can locate every deliverable, reproduce supported workflows, and distinguish verified work from constraints.
Handoff: provide the final requirement audit and exact commands/results.
```

---

## Suggested Execution Order

1. WP-0 baseline
2. WP-1 artifacts and model cards
3. WP-3 synchronous API
4. WP-4 batch and worker
5. WP-5 persistence and observability
6. WP-6 security and resilience
7. WP-7 tests and CI
8. WP-8 containerization
9. WP-2 export and benchmarking can proceed after WP-1 and in parallel with WP-3 through WP-8 when compatible hardware is available
10. WP-9 final documentation and submission review

## Agent Handoff Template

Add this beneath the active work package after each meaningful handoff:

```text
Status: not started | in progress | blocked | complete
Changed: <files and a one-line summary>
Validated: <commands run and result>
Blockers: <none or concise blocker>
Next: <single next action>
```
