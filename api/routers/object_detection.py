"""Object detection routes."""

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from api.models.object_detection import (
    BatchObjectDetectionItem,
    BatchObjectDetectionResponse,
    InferenceTask,
    ObjectDetectionResponse,
)
from api.services.object_detection.yolo_service import yolo_prediction_service

MAX_BATCH_SIZE = 5

router = APIRouter(tags=["object-detection"])


@router.post("/detect", response_model=ObjectDetectionResponse)
async def detect_objects(file: UploadFile = File(...)) -> ObjectDetectionResponse:
    """Run object detection on an uploaded image."""

    image_bytes = await _read_image_bytes(file)
    try:
        detections = yolo_prediction_service.predict(image_bytes)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ObjectDetectionResponse(detections=detections)


@router.post("/batch", response_model=BatchObjectDetectionResponse)
async def batch_inference(
    task: InferenceTask = Form(...),
    files: list[UploadFile] = File(...),
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

    try:
        image_payloads = [await _read_image_bytes(file) for file in files]
        batch_detections = yolo_prediction_service.predict_batch_from_bytes(image_payloads)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    results = [
        BatchObjectDetectionItem(
            filename=file.filename or "uploaded-image",
            detections=detections,
        )
        for file, detections in zip(files, batch_detections, strict=True)
    ]

    return BatchObjectDetectionResponse(task=task, results=results)


async def _read_image_bytes(file: UploadFile) -> bytes:
    """Validate an uploaded image and return its raw bytes."""

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image.",
        )

    return await file.read()
