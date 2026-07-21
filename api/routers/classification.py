"""Classification routes."""

from time import perf_counter
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from api.config import settings
from api.middleware.auth import require_api_key
from api.metrics import observe_inference
from api.models.classification import (
    ClassificationModel,
    ClassificationResponse,
)
from api.services.classification import (
    ClassificationUnavailableError,
    classification_prediction_service,
)
from api.utils.image_validation import read_validated_image

router = APIRouter(
    tags=["classification"], dependencies=[Depends(require_api_key)]
)


@router.post(
    "/classify",
    response_model=ClassificationResponse,
    responses={
        503: {"description": "No classification model is ready to serve."},
        415: {"description": "Unsupported image content type."},
    },
)
async def classify_image(
    file: UploadFile = File(...),
    model: Annotated[ClassificationModel | None, Form()] = None,
    top_k: Annotated[int, Form(ge=1, le=10)] = 1,
) -> ClassificationResponse:
    """Classify a validated image with the configured serving model."""

    selected_model = model or ClassificationModel(
        settings.CLASSIFICATION_MODEL_NAME
    )
    configured_model = ClassificationModel(settings.CLASSIFICATION_MODEL_NAME)
    if selected_model != configured_model:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "model_unavailable",
                "message": f"Classification model '{selected_model.value}' is not configured.",
            },
        )

    image_bytes = await read_validated_image(file)
    started_at = perf_counter()
    try:
        predictions = classification_prediction_service.predict(
            image_bytes, top_k
        )
    except ClassificationUnavailableError as exc:
        observe_inference(
            task="classify",
            model=classification_prediction_service.get_model_version(),
            outcome="unavailable",
            duration_seconds=perf_counter() - started_at,
        )
        raise HTTPException(
            status_code=503,
            detail={"code": "model_unavailable", "message": str(exc)},
        ) from exc
    except RuntimeError as exc:
        observe_inference(
            task="classify",
            model=classification_prediction_service.get_model_version(),
            outcome="error",
            duration_seconds=perf_counter() - started_at,
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    model_version = classification_prediction_service.get_model_version()
    observe_inference(
        task="classify",
        model=model_version,
        outcome="success",
        duration_seconds=perf_counter() - started_at,
    )
    return ClassificationResponse(
        model=selected_model,
        model_version=model_version,
        predictions=predictions,
    )
