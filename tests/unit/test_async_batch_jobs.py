"""Focused tests for durable asynchronous batch jobs."""

from __future__ import annotations

import asyncio

import pytest
import base64
from io import BytesIO
from datetime import datetime, timezone
from types import SimpleNamespace

from fastapi import Request, UploadFile
from starlette.datastructures import Headers

from api.middleware.auth import AuthenticatedClient
from api.models.object_detection import (
    BatchJobStatus,
    InferenceTask,
    ObjectDetection,
    ObjectDetectionModel,
)
from api.routers import object_detection as router_module
from api import worker as worker_module


def _request(headers: list[tuple[bytes, bytes]] | None = None) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "scheme": "http",
            "path": "/api/v1/batch/jobs",
            "headers": headers or [],
            "query_string": b"",
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
        }
    )


def _upload(name: str = "image.png") -> UploadFile:
    return UploadFile(
        BytesIO(b"image"),
        filename=name,
        headers=Headers({"content-type": "image/png"}),
    )


def test_submit_batch_job_persists_then_enqueues_payloads(mocker) -> None:
    job = SimpleNamespace(id="job-1", status="queued")
    mocker.patch.object(
        router_module, "_read_image_bytes", return_value=b"image"
    )
    create = mocker.patch.object(
        router_module, "create_batch_job", return_value=(job, False)
    )
    delay = mocker.patch.object(router_module.run_batch_detection, "delay")
    mocker.patch.object(router_module, "_publish_queue_depth")

    response = asyncio.run(
        router_module.submit_batch_job(
            request=_request([(b"idempotency-key", b"retry-1")]),
            files=[_upload()],
            task=InferenceTask.DETECT,
            model=ObjectDetectionModel.YOLOV8N,
            client=AuthenticatedClient(client_id="client-a", tier="standard"),
        )
    )

    assert response.job_id == "job-1"
    assert response.status is BatchJobStatus.QUEUED
    assert response.idempotent_replay is False
    assert create.call_args.kwargs["client_id"] == "client-a"
    assert create.call_args.kwargs["item_metadata"][0]["size_bytes"] == len(
        b"image"
    )
    assert delay.call_args.args == (
        "job-1",
        [base64.b64encode(b"image").decode("ascii")],
    )


def test_submit_batch_job_returns_503_when_broker_is_unavailable(
    mocker,
) -> None:
    job = SimpleNamespace(id="job-1", status="queued")
    mocker.patch.object(
        router_module, "_read_image_bytes", return_value=b"image"
    )
    mocker.patch.object(
        router_module, "create_batch_job", return_value=(job, False)
    )
    mocker.patch.object(
        router_module.run_batch_detection,
        "delay",
        side_effect=OSError("broker down"),
    )
    delete = mocker.patch.object(router_module, "delete_batch_job")

    with pytest.raises(Exception) as exc_info:
        asyncio.run(
            router_module.submit_batch_job(
                request=_request(),
                files=[_upload()],
                task=InferenceTask.DETECT,
                model=ObjectDetectionModel.YOLOV8N,
                client=AuthenticatedClient(
                    client_id="client-a", tier="standard"
                ),
            )
        )

    assert getattr(exc_info.value, "status_code", None) == 503
    delete.assert_called_once_with(job_id="job-1")


def test_submit_batch_job_does_not_reenqueue_idempotent_replay(mocker) -> None:
    job = SimpleNamespace(id="job-1", status="queued")
    mocker.patch.object(
        router_module, "_read_image_bytes", return_value=b"image"
    )
    mocker.patch.object(
        router_module, "create_batch_job", return_value=(job, True)
    )
    delay = mocker.patch.object(router_module.run_batch_detection, "delay")

    response = asyncio.run(
        router_module.submit_batch_job(
            request=_request(),
            files=[_upload()],
            task=InferenceTask.DETECT,
            model=ObjectDetectionModel.YOLOV8N,
            client=AuthenticatedClient(client_id="client-a", tier="standard"),
        )
    )

    assert response.idempotent_replay is True
    delay.assert_not_called()


def test_worker_persists_normalized_detection_results(mocker) -> None:
    job = SimpleNamespace(id="job-1", model="yolov8n", task="detect")
    detection = ObjectDetection(
        x1=1, y1=2, x2=3, y2=4, confidence=0.9, class_id=2
    )
    service = SimpleNamespace(
        get_model_version=lambda: "yolov8n-v1",
        predict_batch_from_bytes=lambda payloads: [
            [detection] for _ in payloads
        ],
    )
    mocker.patch.object(
        worker_module, "mark_batch_job_running", return_value=job
    )
    mocker.patch.object(
        worker_module, "get_object_detection_service", return_value=service
    )
    complete = mocker.patch.object(worker_module, "complete_batch_job")
    mocker.patch.object(worker_module, "observe_inference")
    mocker.patch.object(worker_module, "observe_batch_request")

    worker_module.run_batch_detection.run(
        "job-1", [base64.b64encode(b"image").decode("ascii")]
    )

    assert complete.call_args.kwargs["job_id"] == "job-1"
    assert complete.call_args.kwargs["model_version"] == "yolov8n-v1"
    assert (
        complete.call_args.kwargs["items"][0]["detections"][0]["class_id"] == 2
    )


def test_get_batch_job_status_returns_results_for_authenticated_client(
    mocker,
) -> None:
    now = datetime.now(timezone.utc)
    job = SimpleNamespace(
        id="job-1",
        task="detect",
        model="yolov8n",
        model_version="yolov8n-v1",
        status="succeeded",
        attempts=1,
        error=None,
        created_at=now,
        started_at=now,
        completed_at=now,
        expires_at=now,
        items=[
            SimpleNamespace(
                filename="image.png",
                content_type="image/png",
                size_bytes=5,
                detections=[
                    {
                        "x1": 1,
                        "y1": 2,
                        "x2": 3,
                        "y2": 4,
                        "confidence": 0.9,
                        "class_id": 1,
                    }
                ],
                error=None,
            )
        ],
    )
    get = mocker.patch.object(router_module, "get_batch_job", return_value=job)

    response = router_module.get_batch_job_status(
        "job-1", AuthenticatedClient(client_id="client-a", tier="standard")
    )

    assert get.call_args.kwargs == {"job_id": "job-1", "client_id": "client-a"}
    assert response.status is BatchJobStatus.SUCCEEDED
    assert response.items[0].detections[0].class_id == 1
