"""Object detection routes."""

from time import perf_counter

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from api.metrics import observe_batch_request, observe_inference
from api.models.object_detection import (
    BatchObjectDetectionItem,
    BatchObjectDetectionResponse,
    InferenceTask,
    ObjectDetection,
    ObjectDetectionResponse,
)
from api.services.cache_service import redis_cache_service
from api.services.object_detection.yolo_service import yolo_prediction_service

MAX_BATCH_SIZE = 5

router = APIRouter(tags=['object-detection'])


@router.post('/detect', response_model=ObjectDetectionResponse)
async def detect_objects(file: UploadFile = File(...)) -> ObjectDetectionResponse:
    """Run object detection on an uploaded image."""

    image_bytes = await _read_image_bytes(file)

    try:
        model_version = yolo_prediction_service.get_model_version()
        cached_detections = await redis_cache_service.get_detection(
            image_bytes,
            model_version,
        )
        if cached_detections is not None:
            return ObjectDetectionResponse(detections=cached_detections)

        started_at = perf_counter()
        try:
            detections = yolo_prediction_service.predict(image_bytes)
        except (FileNotFoundError, RuntimeError):
            observe_inference(
                task=InferenceTask.DETECT.value,
                model=model_version,
                outcome='error',
                duration_seconds=perf_counter() - started_at,
            )
            raise

        observe_inference(
            task=InferenceTask.DETECT.value,
            model=model_version,
            outcome='success',
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

    return ObjectDetectionResponse(detections=detections)


@router.post('/batch', response_model=BatchObjectDetectionResponse)
async def batch_inference(
    task: InferenceTask = Form(...),
    files: list[UploadFile] = File(...),
) -> BatchObjectDetectionResponse:
    """Run batch inference for up to five uploaded images."""

    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f'Batch requests support up to {MAX_BATCH_SIZE} images.',
        )

    if task is InferenceTask.CLASSIFY:
        raise HTTPException(
            status_code=501,
            detail='Batch classification is not supported yet.',
        )

    outcome = 'success'

    try:
        image_payloads = [await _read_image_bytes(file) for file in files]
        model_version = yolo_prediction_service.get_model_version()
        batch_detections = await _get_batch_detections(image_payloads, model_version)
    except FileNotFoundError as exc:
        outcome = 'error'
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RuntimeError as exc:
        outcome = 'error'
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        observe_batch_request(
            task=task.value,
            batch_size=len(files),
            outcome=outcome,
        )

    results = [
        BatchObjectDetectionItem(
            filename=file.filename or 'uploaded-image',
            detections=detections,
        )
        for file, detections in zip(files, batch_detections, strict=True)
    ]

    return BatchObjectDetectionResponse(task=task, results=results)


async def _get_batch_detections(
    image_payloads: list[bytes],
    model_version: str,
) -> list[list[ObjectDetection]]:
    """Return detections for a batch while reusing per-image cache entries."""

    detections_by_index: list[list[ObjectDetection] | None] = [None] * len(image_payloads)
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
            fresh_detections = yolo_prediction_service.predict_batch_from_bytes(missed_payloads)
        except (FileNotFoundError, RuntimeError):
            observe_inference(
                task=InferenceTask.DETECT.value,
                model=model_version,
                outcome='error',
                duration_seconds=perf_counter() - started_at,
                image_count=len(missed_payloads),
            )
            raise

        observe_inference(
            task=InferenceTask.DETECT.value,
            model=model_version,
            outcome='success',
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

    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail='Uploaded file must be an image.',
        )

    return await file.read()
