# Technical Readme

## System design

```text
Client -> Nginx -> FastAPI -> model services
                       |         |
                       |         -> Redis inference-result cache
                       -> Celery -> Redis broker -> worker -> PostgreSQL request logs
                       -> /api/v1/metrics -> Prometheus -> Grafana
```

PostgreSQL is the durable store for request logs and job metadata. Redis is used for cache entries, rate-limit counters, and Celery transport; cache loss degrades to fresh inference. Model artifacts are mounted read-only at `/opt/models/artifacts` and are deliberately not baked into the image.

## API and security decisions

The API accepts multipart image uploads (`file`) rather than remote URLs or paths, keeping server-side filesystem access and SSRF out of the request contract. Inference routes require an environment-provisioned `X-API-Key`; keys are not persisted or exposed by the application. Redis applies a fixed-window request limit per client tier. If Redis is unavailable, the API remains available but skips rate limiting and result caching; operators should alert on the dependency gauge.

Access logs use request IDs and avoid recording uploaded image bytes or API-key values. The application keeps request metadata, timing, model/version, status, and error data for diagnosis.

## Model selection and optimisation

The initial serving path is YOLOv8n object detection: a practical low-latency baseline. The registry isolates the endpoint contract from the selected detector so alternatives can be added without changing client payloads. Classification and alternative detection artifacts remain separately versioned under `models/artifacts`.

ONNX, INT8, and TensorRT exports need compatible locally generated artifacts and, for TensorRT, NVIDIA hardware plus a matching runtime. Benchmark claims must include artifact version, input shape, batch size, warm-up, percentile latency, throughput, memory, and hardware/software versions. Unavailable GPU/TensorRT results are not presented as verified.

## Scaling and operations

Scale API and worker replicas independently after moving the gateway/load-balancer and Celery routing to the target platform. Keep artifact mounts consistent across replicas. Redis and PostgreSQL require managed high-availability deployments for production; Compose is for local development and single-host demonstration.

Prometheus scrapes `/api/v1/metrics` every five seconds. It exports HTTP request count/latency/errors, model-inference count/latency, batch counts and sizes, cache outcomes, and dependency availability. Queue depth depends on the asynchronous batch implementation and should be added once queued jobs are enabled.

## Known limitations

- The Compose GPU reservation and TensorRT base image require a compatible NVIDIA host.
- Model artifacts are external to source control; a deployment must supply them before a real inference smoke test succeeds.
- Compose does not replace managed secrets, TLS termination, multi-node stateful services, backups, or alert routing.
