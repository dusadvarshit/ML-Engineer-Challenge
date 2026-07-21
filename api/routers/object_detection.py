"""Object detection routes."""

import base64
from time import perf_counter
from typing import Annotated

from celery.exceptions import CeleryError  # type: ignore[import-untyped]
from kombu.exceptions import OperationalError  # type: ignore[import-untyped]
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)

from api.config import settings
from api.middleware.auth import AuthenticatedClient, require_api_key
from api.metrics import (
    observe_batch_request,
    observe_inference,
    set_batch_queue_depth,
)
from api.models.object_detection import (
    BatchJobAcceptedResponse,
    BatchJobItem,
    BatchJobStatus,
    BatchJobStatusResponse,
    BatchObjectDetectionItem,
    BatchObjectDetectionResponse,
    InferenceTask,
    ObjectDetection,
    ObjectDetectionModel,
    ObjectDetectionResponse,
)
from api.services.cache_service import redis_cache_service
from api.services.batch_job_service import (
    create_batch_job,
    delete_batch_job,
    get_batch_job,
)
from api.services.object_detection.registry import (
    ObjectDetectionPredictionService,
    get_object_detection_service,
)
from api.services.object_detection.yolo_service import (
    yolo_prediction_service,
)  # noqa: F401
from api.utils.image_validation import read_validated_image
from api.worker import run_batch_detection

__all__ = ["yolo_prediction_service"]

MAX_BATCH_SIZE = 5

router = APIRouter(
    tags=["object-detection"], dependencies=[Depends(require_api_key)]
)


@router.post("/detect", response_model=ObjectDetectionResponse)
async def detect_objects(
    file: UploadFile = File(...),
    model: Annotated[ObjectDetectionModel | None, Form()] = None,
) -> ObjectDetectionResponse:
    """Run object detection on an uploaded image."""

    image_bytes = await _read_image_bytes(file)
    model = model or ObjectDetectionModel(
        settings.get_default_object_detection_model()
    )
    prediction_service = get_object_detection_service(model)

    try:
        model_version = prediction_service.get_model_version()
        cached_detections = await redis_cache_service.get_detection(
            image_bytes,
            model_version,
        )
        if cached_detections is not None:
            return ObjectDetectionResponse(
                model=model,
                model_version=model_version,
                detections=cached_detections,
            )

        started_at = perf_counter()
        try:
            detections = prediction_service.predict(image_bytes)
        except (FileNotFoundError, RuntimeError):
            observe_inference(
                task=InferenceTask.DETECT.value,
                model=model_version,
                outcome="error",
                duration_seconds=perf_counter() - started_at,
            )
            raise

        observe_inference(
            task=InferenceTask.DETECT.value,
            model=model_version,
            outcome="success",
            duration_seconds=perf_counter() - started_at,
        )
        await redis_cache_service.set_detection(
            image_bytes,
            model_version,
            detections,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ObjectDetectionResponse(
        model=model,
        model_version=model_version,
        detections=detections,
    )


@router.post("/batch", response_model=BatchObjectDetectionResponse)
async def batch_inference(
    task: InferenceTask = Form(...),
    files: list[UploadFile] = File(...),
    model: Annotated[ObjectDetectionModel | None, Form()] = None,
) -> BatchObjectDetectionResponse:
    """Run batch inference for up to five uploaded images."""

    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch requests support up to {MAX_BATCH_SIZE} images.",
        )

    if task is InferenceTask.CLASSIFY:
        raise HTTPException(
            status_code=501,
            detail="Batch classification is not supported yet.",
        )

    outcome = "success"
    model = model or ObjectDetectionModel(
        settings.get_default_object_detection_model()
    )
    prediction_service = get_object_detection_service(model)

    try:
        image_payloads = [await _read_image_bytes(file) for file in files]
        model_version = prediction_service.get_model_version()
        batch_detections = await _get_batch_detections(
            image_payloads,
            model_version,
            prediction_service,
        )
    except FileNotFoundError as exc:
        outcome = "error"
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RuntimeError as exc:
        outcome = "error"
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        observe_batch_request(
            task=task.value,
            batch_size=len(files),
            outcome=outcome,
        )

    results = [
        BatchObjectDetectionItem(
            filename=file.filename or "uploaded-image",
            detections=detections,
        )
        for file, detections in zip(files, batch_detections, strict=True)
    ]

    return BatchObjectDetectionResponse(task=task, results=results)


@router.post(
    "/batch/jobs",
    response_model=BatchJobAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_batch_job(
    request: Request,
    task: InferenceTask = Form(InferenceTask.DETECT),
    files: list[UploadFile] = File(...),
    model: Annotated[ObjectDetectionModel | None, Form()] = None,
    client: AuthenticatedClient = Depends(require_api_key),
) -> BatchJobAcceptedResponse:
    """Persist and enqueue a durable asynchronous detection batch job."""

    if not files or len(files) > settings.MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch jobs support 1 to {settings.MAX_BATCH_SIZE} images.",
        )
    if task is not InferenceTask.DETECT:
        raise HTTPException(
            status_code=501,
            detail="Asynchronous batch classification is not supported yet.",
        )

    image_payloads = [await _read_image_bytes(file) for file in files]
    selected_model = model or ObjectDetectionModel(
        settings.get_default_object_detection_model()
    )
    item_metadata = [
        {
            "filename": file.filename or "uploaded-image",
            "content_type": file.content_type,
            "size_bytes": len(payload),
        }
        for file, payload in zip(files, image_payloads, strict=True)
    ]
    idempotency_key = request.headers.get("Idempotency-Key")
    if idempotency_key and len(idempotency_key) > 255:
        raise HTTPException(
            status_code=400,
            detail="Idempotency-Key must be 255 characters or fewer.",
        )
    job, replayed = create_batch_job(
        client_id=client.client_id,
        task=task,
        model=selected_model,
        item_metadata=item_metadata,
        idempotency_key=idempotency_key,
    )
    if not replayed:
        try:
            run_batch_detection.delay(
                job.id,
                [
                    base64.b64encode(payload).decode("ascii")
                    for payload in image_payloads
                ],
            )
            await _publish_queue_depth()
        except (CeleryError, OperationalError, OSError) as exc:
            delete_batch_job(job_id=job.id)
            raise HTTPException(
                status_code=503,
                detail="Batch queue is unavailable. Please retry.",
            ) from exc
    return BatchJobAcceptedResponse(
        job_id=job.id,
        status=BatchJobStatus(job.status),
        status_url=f"{settings.API_PREFIX}/batch/jobs/{job.id}",
        idempotent_replay=replayed,
    )


@router.get("/batch/jobs/{job_id}", response_model=BatchJobStatusResponse)
def get_batch_job_status(
    job_id: str,
    client: AuthenticatedClient = Depends(require_api_key),
) -> BatchJobStatusResponse:
    """Return the durable status and ordered per-image results for one client."""

    job = get_batch_job(job_id=job_id, client_id=client.client_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Batch job not found.")
    return BatchJobStatusResponse(
        job_id=job.id,
        task=InferenceTask(job.task),
        model=ObjectDetectionModel(job.model),
        model_version=job.model_version,
        status=BatchJobStatus(job.status),
        attempts=job.attempts,
        error=job.error,
        items=[
            BatchJobItem(
                filename=item.filename,
                content_type=item.content_type,
                size_bytes=item.size_bytes,
                detections=(
                    [
                        ObjectDetection.model_validate(detection)
                        for detection in item.detections
                    ]
                    if item.detections is not None
                    else None
                ),
                error=item.error,
            )
            for item in job.items
        ],
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        expires_at=job.expires_at,
    )


async def _publish_queue_depth() -> None:
    """Update the queue-depth gauge when Redis supports list operations."""

    redis_client = redis_cache_service.client
    llen = (
        getattr(redis_client, "llen", None)
        if redis_client is not None
        else None
    )
    if llen is None:
        return
    try:
        depth = await llen("celery")
    except Exception:
        return
    set_batch_queue_depth(queue="celery", depth=int(depth))


async def _get_batch_detections(
    image_payloads: list[bytes],
    model_version: str,
    prediction_service: ObjectDetectionPredictionService,
) -> list[list[ObjectDetection]]:
    """Return detections for a batch while reusing per-image cache entries."""

    detections_by_index: list[list[ObjectDetection] | None] = [None] * len(
        image_payloads
    )
    missed_indexes: list[int] = []
    missed_payloads: list[bytes] = []

    for index, image_bytes in enumerate(image_payloads):
        cached_detections = await redis_cache_service.get_detection(
            image_bytes,
            model_version,
        )
        if cached_detections is None:
            missed_indexes.append(index)
            missed_payloads.append(image_bytes)
            continue

        detections_by_index[index] = cached_detections

    if missed_payloads:
        started_at = perf_counter()
        try:
            fresh_detections = prediction_service.predict_batch_from_bytes(
                missed_payloads
            )
        except (FileNotFoundError, RuntimeError):
            observe_inference(
                task=InferenceTask.DETECT.value,
                model=model_version,
                outcome="error",
                duration_seconds=perf_counter() - started_at,
                image_count=len(missed_payloads),
            )
            raise

        observe_inference(
            task=InferenceTask.DETECT.value,
            model=model_version,
            outcome="success",
            duration_seconds=perf_counter() - started_at,
            image_count=len(missed_payloads),
        )
        for index, image_bytes, detections in zip(
            missed_indexes,
            missed_payloads,
            fresh_detections,
            strict=True,
        ):
            detections_by_index[index] = detections
            await redis_cache_service.set_detection(
                image_bytes,
                model_version,
                detections,
            )

    return [detections or [] for detections in detections_by_index]


async def _read_image_bytes(file: UploadFile) -> bytes:
    """Validate an uploaded image and return its raw bytes."""

    return await read_validated_image(file)
