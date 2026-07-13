"""Unit tests for object detection route caching behavior."""

from __future__ import annotations

import asyncio
from io import BytesIO
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from starlette.datastructures import Headers, UploadFile

from api.models.object_detection import InferenceTask, ObjectDetection
from api.routers.object_detection import _read_image_bytes, batch_inference, detect_objects
from api.services.cache_service import redis_cache_service
from api.services.object_detection.yolo_service import yolo_prediction_service

pytestmark = pytest.mark.unit


def _build_upload_file(
    image_bytes: bytes,
    filename: str = 'test.png',
    content_type: str | None = 'image/png',
) -> UploadFile:
    """Create an upload payload for route tests."""

    headers = Headers({'content-type': content_type} if content_type is not None else {})
    return UploadFile(
        file=BytesIO(image_bytes),
        filename=filename,
        headers=headers,
    )


def test_detect_objects_returns_cached_result_without_running_inference(mocker) -> None:
    """Cache hits should short-circuit the YOLO prediction call."""

    image_bytes = b'cached-image'
    cached_detections = [
        ObjectDetection(
            x1=1.0,
            y1=2.0,
            x2=3.0,
            y2=4.0,
            confidence=0.9,
            class_id=5,
        )
    ]

    mocker.patch.object(
        yolo_prediction_service,
        'get_model_version',
        return_value='yolov8n-v1.0.0-best.pt',
    )
    get_detection = mocker.patch.object(
        redis_cache_service,
        'get_detection',
        new=AsyncMock(return_value=cached_detections),
    )
    set_detection = mocker.patch.object(
        redis_cache_service,
        'set_detection',
        new=AsyncMock(return_value=False),
    )
    predict = mocker.patch.object(yolo_prediction_service, 'predict')

    response = asyncio.run(detect_objects(_build_upload_file(image_bytes)))

    assert [item.model_dump() for item in response.detections] == [
        item.model_dump() for item in cached_detections
    ]
    get_detection.assert_awaited_once_with(image_bytes, 'yolov8n-v1.0.0-best.pt')
    set_detection.assert_not_awaited()
    predict.assert_not_called()


def test_detect_objects_caches_fresh_inference_result(mocker) -> None:
    """Cache misses should run inference and store the new detections."""

    image_bytes = b'fresh-image'
    fresh_detections = [
        ObjectDetection(
            x1=10.0,
            y1=20.0,
            x2=30.0,
            y2=40.0,
            confidence=0.8,
            class_id=2,
        )
    ]

    mocker.patch.object(
        yolo_prediction_service,
        'get_model_version',
        return_value='yolov8n-v1.0.0-best.pt',
    )
    get_detection = mocker.patch.object(
        redis_cache_service,
        'get_detection',
        new=AsyncMock(return_value=None),
    )
    set_detection = mocker.patch.object(
        redis_cache_service,
        'set_detection',
        new=AsyncMock(return_value=True),
    )
    predict = mocker.patch.object(
        yolo_prediction_service,
        'predict',
        return_value=fresh_detections,
    )

    response = asyncio.run(detect_objects(_build_upload_file(image_bytes)))

    assert [item.model_dump() for item in response.detections] == [
        item.model_dump() for item in fresh_detections
    ]
    get_detection.assert_awaited_once_with(image_bytes, 'yolov8n-v1.0.0-best.pt')
    predict.assert_called_once_with(image_bytes)
    set_detection.assert_awaited_once_with(
        image_bytes,
        'yolov8n-v1.0.0-best.pt',
        fresh_detections,
    )


def test_detect_objects_returns_result_when_cache_store_fails(mocker) -> None:
    """Cache write failures should not prevent returning fresh detections."""

    image_bytes = b'fresh-image'
    fresh_detections = [
        ObjectDetection(
            x1=10.0,
            y1=20.0,
            x2=30.0,
            y2=40.0,
            confidence=0.8,
            class_id=2,
        )
    ]

    mocker.patch.object(
        yolo_prediction_service,
        'get_model_version',
        return_value='yolov8n-v1.0.0-best.pt',
    )
    mocker.patch.object(
        redis_cache_service,
        'get_detection',
        new=AsyncMock(return_value=None),
    )
    set_detection = mocker.patch.object(
        redis_cache_service,
        'set_detection',
        new=AsyncMock(return_value=False),
    )
    predict = mocker.patch.object(
        yolo_prediction_service,
        'predict',
        return_value=fresh_detections,
    )

    response = asyncio.run(detect_objects(_build_upload_file(image_bytes)))

    assert [item.model_dump() for item in response.detections] == [
        item.model_dump() for item in fresh_detections
    ]
    predict.assert_called_once_with(image_bytes)
    set_detection.assert_awaited_once_with(
        image_bytes,
        'yolov8n-v1.0.0-best.pt',
        fresh_detections,
    )


@pytest.mark.parametrize('error_type', [FileNotFoundError, RuntimeError])
def test_detect_objects_translates_service_errors(mocker, error_type: type[Exception]) -> None:
    """Inference failures should become HTTP 500 responses."""

    image_bytes = b'broken-image'

    mocker.patch.object(
        yolo_prediction_service,
        'get_model_version',
        return_value='yolov8n-v1.0.0-best.pt',
    )
    mocker.patch.object(
        redis_cache_service,
        'get_detection',
        new=AsyncMock(return_value=None),
    )
    mocker.patch.object(
        yolo_prediction_service,
        'predict',
        side_effect=error_type('service failed'),
    )

    with pytest.raises(HTTPException, match='service failed') as exc_info:
        asyncio.run(detect_objects(_build_upload_file(image_bytes)))

    assert exc_info.value.status_code == 500


def test_batch_inference_returns_cached_results_without_running_batch_inference(mocker) -> None:
    """Full cache hits should short-circuit batch YOLO inference."""

    payloads = [b'image-1', b'image-2']
    cached_batches = [
        [
            ObjectDetection(
                x1=1.0,
                y1=2.0,
                x2=3.0,
                y2=4.0,
                confidence=0.9,
                class_id=1,
            )
        ],
        [
            ObjectDetection(
                x1=5.0,
                y1=6.0,
                x2=7.0,
                y2=8.0,
                confidence=0.8,
                class_id=2,
            )
        ],
    ]
    files = [
        _build_upload_file(payloads[0], 'first.png'),
        _build_upload_file(payloads[1], 'second.png'),
    ]

    mocker.patch.object(
        yolo_prediction_service,
        'get_model_version',
        return_value='yolov8n-v1.0.0-best.pt',
    )
    get_detection = mocker.patch.object(
        redis_cache_service,
        'get_detection',
        new=AsyncMock(side_effect=cached_batches),
    )
    set_detection = mocker.patch.object(
        redis_cache_service,
        'set_detection',
        new=AsyncMock(return_value=True),
    )
    predict_batch = mocker.patch.object(yolo_prediction_service, 'predict_batch_from_bytes')

    response = asyncio.run(batch_inference(task=InferenceTask.DETECT, files=files))

    assert [item.filename for item in response.results] == ['first.png', 'second.png']
    assert [item.model_dump() for item in response.results[0].detections] == [
        item.model_dump() for item in cached_batches[0]
    ]
    assert [item.model_dump() for item in response.results[1].detections] == [
        item.model_dump() for item in cached_batches[1]
    ]
    assert get_detection.await_count == 2
    set_detection.assert_not_awaited()
    predict_batch.assert_not_called()


def test_batch_inference_only_runs_yolo_for_cache_misses_and_preserves_order(mocker) -> None:
    """Partial cache hits should infer only misses and merge results by original index."""

    payloads = [b'image-1', b'image-2', b'image-3']
    files = [
        _build_upload_file(payloads[0], 'first.png'),
        _build_upload_file(payloads[1], 'second.png'),
        _build_upload_file(payloads[2], 'third.png'),
    ]
    cached_first = [
        ObjectDetection(
            x1=1.0,
            y1=2.0,
            x2=3.0,
            y2=4.0,
            confidence=0.95,
            class_id=10,
        )
    ]
    fresh_second = [
        ObjectDetection(
            x1=11.0,
            y1=12.0,
            x2=13.0,
            y2=14.0,
            confidence=0.85,
            class_id=20,
        )
    ]
    fresh_third = [
        ObjectDetection(
            x1=21.0,
            y1=22.0,
            x2=23.0,
            y2=24.0,
            confidence=0.75,
            class_id=30,
        )
    ]

    mocker.patch.object(
        yolo_prediction_service,
        'get_model_version',
        return_value='yolov8n-v1.0.0-best.pt',
    )
    get_detection = mocker.patch.object(
        redis_cache_service,
        'get_detection',
        new=AsyncMock(side_effect=[cached_first, None, None]),
    )
    set_detection = mocker.patch.object(
        redis_cache_service,
        'set_detection',
        new=AsyncMock(return_value=True),
    )
    predict_batch = mocker.patch.object(
        yolo_prediction_service,
        'predict_batch_from_bytes',
        return_value=[fresh_second, fresh_third],
    )

    response = asyncio.run(batch_inference(task=InferenceTask.DETECT, files=files))

    assert [item.filename for item in response.results] == [
        'first.png',
        'second.png',
        'third.png',
    ]
    assert [item.model_dump() for item in response.results[0].detections] == [
        item.model_dump() for item in cached_first
    ]
    assert [item.model_dump() for item in response.results[1].detections] == [
        item.model_dump() for item in fresh_second
    ]
    assert [item.model_dump() for item in response.results[2].detections] == [
        item.model_dump() for item in fresh_third
    ]
    assert get_detection.await_count == 3
    predict_batch.assert_called_once_with([payloads[1], payloads[2]])
    assert set_detection.await_args_list[0].args == (
        payloads[1],
        'yolov8n-v1.0.0-best.pt',
        fresh_second,
    )
    assert set_detection.await_args_list[1].args == (
        payloads[2],
        'yolov8n-v1.0.0-best.pt',
        fresh_third,
    )


def test_batch_inference_returns_fresh_results_when_cache_operations_fail(mocker) -> None:
    """Batch responses should still succeed when cache reads and writes degrade."""

    payloads = [b'image-1', b'image-2']
    files = [
        _build_upload_file(payloads[0], 'first.png'),
        _build_upload_file(payloads[1], 'second.png'),
    ]
    fresh_first = [
        ObjectDetection(
            x1=11.0,
            y1=12.0,
            x2=13.0,
            y2=14.0,
            confidence=0.85,
            class_id=20,
        )
    ]
    fresh_second = [
        ObjectDetection(
            x1=21.0,
            y1=22.0,
            x2=23.0,
            y2=24.0,
            confidence=0.75,
            class_id=30,
        )
    ]

    mocker.patch.object(
        yolo_prediction_service,
        'get_model_version',
        return_value='yolov8n-v1.0.0-best.pt',
    )
    mocker.patch.object(
        redis_cache_service,
        'get_detection',
        new=AsyncMock(side_effect=[None, None]),
    )
    set_detection = mocker.patch.object(
        redis_cache_service,
        'set_detection',
        new=AsyncMock(return_value=False),
    )
    predict_batch = mocker.patch.object(
        yolo_prediction_service,
        'predict_batch_from_bytes',
        return_value=[fresh_first, fresh_second],
    )

    response = asyncio.run(batch_inference(task=InferenceTask.DETECT, files=files))

    assert [item.filename for item in response.results] == ['first.png', 'second.png']
    assert [item.model_dump() for item in response.results[0].detections] == [
        item.model_dump() for item in fresh_first
    ]
    assert [item.model_dump() for item in response.results[1].detections] == [
        item.model_dump() for item in fresh_second
    ]
    predict_batch.assert_called_once_with(payloads)
    assert set_detection.await_args_list[0].args == (
        payloads[0],
        'yolov8n-v1.0.0-best.pt',
        fresh_first,
    )
    assert set_detection.await_args_list[1].args == (
        payloads[1],
        'yolov8n-v1.0.0-best.pt',
        fresh_second,
    )


def test_batch_inference_rejects_batches_over_route_limit() -> None:
    """The route should enforce its tighter batch limit before inference."""

    files = [_build_upload_file(b'image', f'image-{index}.png') for index in range(6)]

    with pytest.raises(HTTPException, match='Batch requests support up to 5 images') as exc_info:
        asyncio.run(batch_inference(task=InferenceTask.DETECT, files=files))

    assert exc_info.value.status_code == 400


def test_batch_inference_rejects_unimplemented_classification_task() -> None:
    """Batch classification should fail explicitly until it is implemented."""

    files = [_build_upload_file(b'image', 'image-0.png')]

    with pytest.raises(HTTPException, match='Batch classification is not supported yet') as exc_info:
        asyncio.run(batch_inference(task=InferenceTask.CLASSIFY, files=files))

    assert exc_info.value.status_code == 501


@pytest.mark.parametrize('error_type', [FileNotFoundError, RuntimeError])
def test_batch_inference_translates_service_errors(mocker, error_type: type[Exception]) -> None:
    """Batch inference failures should become HTTP 500 responses."""

    files = [_build_upload_file(b'image', 'image-0.png')]

    mocker.patch.object(
        yolo_prediction_service,
        'get_model_version',
        return_value='yolov8n-v1.0.0-best.pt',
    )
    mocker.patch.object(
        redis_cache_service,
        'get_detection',
        new=AsyncMock(return_value=None),
    )
    mocker.patch.object(
        yolo_prediction_service,
        'predict_batch_from_bytes',
        side_effect=error_type('batch failed'),
    )

    with pytest.raises(HTTPException, match='batch failed') as exc_info:
        asyncio.run(batch_inference(task=InferenceTask.DETECT, files=files))

    assert exc_info.value.status_code == 500


def test_read_image_bytes_rejects_missing_content_type() -> None:
    """Uploads without an image content type should be rejected."""

    with pytest.raises(HTTPException, match='Uploaded file must be an image') as exc_info:
        asyncio.run(_read_image_bytes(_build_upload_file(b'image', content_type=None)))

    assert exc_info.value.status_code == 400
