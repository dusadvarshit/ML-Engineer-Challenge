## Simple Plan

### Goal
Build a working ML project that can:
- classify images
- detect objects
- run as an API
- run with Docker
- include tests and documentation

### Step 1: Understand the repo
- Read `README.md` fully once.
- Look inside the `scripts/` folder.
- Identify which scripts are for:
  - dataset download
  - model training
  - Docker
  - testing
- Do not try to build everything at once.

### Step 2: Set up the project structure
- Create the main folders if missing:
  - `api/`
  - `models/`
  - `tests/`
  - `docs/`
  - `monitoring/`
- Keep the structure clean from the start.

### Step 3: Start with the smallest working version
- Build a very basic FastAPI app first.
- Add these endpoints first:
  - health check
  - classify
  - detect
- At first, it is okay if the endpoints use mock or simple model logic.
- The goal is to get the skeleton working early.

### Step 4: Prepare the ML models
- Download the dataset using the provided script.
- Train or fine-tune the classification model.
- Set up the detection model.
- Save model files in a clear location.
- Make sure each model can run locally before connecting it to the API.

### Step 5: Connect models to the API
- Replace mock logic with real model inference.
- Add input validation for images.
- Return clean JSON responses.
- Add error handling so bad inputs do not crash the app.

### Step 6: Add background processing and storage
- Add Redis for caching and queue/message support.
- Add Postgres for logs and metrics.
- Add a worker for batch jobs.
- Make batch inference run in the background.

### Step 7: Add tests
- Write unit tests for:
  - image utilities
  - model service
  - API routes
- Write integration tests for:
  - API + DB
  - model loading
  - batch flow
- Use helper files if provided, but still write your own real tests.

### Step 8: Optimize models
- Export models to ONNX.
- Add quantization.
- Add TensorRT version if possible.
- Benchmark the speed of each version.
- Save results in `benchmarks/` or `docs/`.

### Step 9: Add Docker support
- Create Dockerfiles for:
  - API
  - worker
  - gateway if needed
- Create `docker-compose.yml`.
- Make sure the whole stack can start with one command.
- Use helper scripts like `build_image.sh` only as support tools.

### Step 10: Add monitoring and logging
- Add structured logs.
- Add request logging.
- Add metrics endpoint.
- If possible, connect Prometheus and Grafana.

### Step 11: Finish documentation
- Update `README.md` with:
  - setup steps
  - how to run locally
  - how to run Docker
  - API examples
  - test instructions
  - benchmark summary
- Also mention limitations and future improvements.

### Recommended workflow order
1. Read README and inspect scripts
2. Create basic project structure
3. Build simple FastAPI skeleton
4. Get one model working
5. Connect model to API
6. Add second model
7. Add DB, Redis, worker
8. Add tests
9. Add Docker and Compose
10. Add optimization and benchmarking
11. Clean up docs and final polish

### Practical advice
- First make it work.
- Then make it clean.
- Then make it production-ready.
- Do not begin with optimization, Docker, or monitoring before the API and models work.
