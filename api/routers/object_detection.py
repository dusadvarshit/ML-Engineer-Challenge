"""Object detection routes."""

from fastapi import APIRouter, File, HTTPException, UploadFile

from api.models.object_detection import ObjectDetectionResponse
from api.services.object_detection.yolo_service import yolo_prediction_service

router = APIRouter(prefix="/object-detection", tags=["object-detection"])


@router.post("/yolo", response_model=ObjectDetectionResponse)
async def predict_yolo(file: UploadFile = File(...)) -> ObjectDetectionResponse:
    """Run YOLO object detection on an uploaded image."""

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image.",
        )

    image_bytes = await file.read()
    try:
        detections = yolo_prediction_service.predict(image_bytes)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ObjectDetectionResponse(detections=detections)
