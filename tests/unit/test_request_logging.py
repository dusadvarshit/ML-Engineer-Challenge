"""Unit tests for request logging helpers and worker wiring."""

from __future__ import annotations

import io

import pytest
from starlette.datastructures import FormData, Headers, UploadFile

import api.middleware.request_logging as request_logging_module
import api.worker as worker_module

pytestmark = pytest.mark.unit



def test_serialize_form_data_preserves_repeated_fields_and_files() -> None:
    """Multipart serialization should keep all repeated keys."""

    form = FormData(
        [
            ('model', 'yolov8n'),
            ('tags', 'a'),
            ('tags', 'b'),
            (
                'files',
                UploadFile(
                    filename='one.png',
                    file=io.BytesIO(b'one'),
                    headers=Headers({'content-type': 'image/png'}),
                ),
            ),
            (
                'files',
                UploadFile(
                    filename='two.png',
                    file=io.BytesIO(b'two'),
                    headers=Headers({'content-type': 'image/png'}),
                ),
            ),
        ]
    )

    payload = request_logging_module._serialize_form_data(form)

    assert payload['form']['model'] == 'yolov8n'
    assert payload['form']['tags'] == ['a', 'b']
    assert payload['files']['files'] == [
        {'filename': 'one.png', 'content_type': 'image/png'},
        {'filename': 'two.png', 'content_type': 'image/png'},
    ]


@pytest.mark.parametrize(
    ('body', 'content_type', 'expected'),
    [
        (b'{"status":"ok"}', 'application/json', {'status': 'ok'}),
        (b'plain text body', 'text/plain', {'body_preview': 'plain text body'}),
        (b'\x00\x01', 'application/octet-stream', {'size_bytes': 2}),
    ],
)
def test_serialize_response_payload(
    body: bytes,
    content_type: str,
    expected: dict[str, object],
) -> None:
    """Response serialization should adapt to JSON, text, and binary payloads."""

    assert request_logging_module._serialize_response_payload(
        body=body,
        content_type=content_type,
    ) == expected



def test_extract_error_message_prefers_detail_field() -> None:
    """Error extraction should return the API detail when present."""

    assert (
        request_logging_module._extract_error_message(b'{"detail":"bad request"}')
        == 'bad request'
    )



def test_enqueue_request_log_swallows_broker_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Broker failures should not break the HTTP request path."""

    monkeypatch.setattr(
        request_logging_module.persist_request_log,
        'delay',
        lambda _payload: (_ for _ in ()).throw(RuntimeError('broker down')),
    )

    request_logging_module._enqueue_request_log({'request_id': 'req-1'})



def test_persist_request_log_saves_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """Celery task should delegate payload persistence."""

    captured: list[dict[str, object]] = []
    monkeypatch.setattr(worker_module, 'save_request_log', lambda payload: captured.append(payload))

    worker_module.persist_request_log.run({'request_id': 'req-1', 'status_code': 200})

    assert captured == [{'request_id': 'req-1', 'status_code': 200}]
