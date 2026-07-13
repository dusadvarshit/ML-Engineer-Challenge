"""End-to-end smoke tests for the containerized serving stack."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import httpx
import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_BASE_URL = "http://127.0.0.1"
_DEFAULT_PROMETHEUS_URL = "http://127.0.0.1:9090"
_REQUEST_TIMEOUT_SECONDS = 5.0
_STARTUP_TIMEOUT_SECONDS = 240.0


def _compose_env() -> dict[str, str]:
    """Return an environment suitable for docker compose commands."""

    env = os.environ.copy()
    env.setdefault("DOCKER_BUILDKIT", "1")
    return env


def _run_compose(*args: str) -> subprocess.CompletedProcess[str]:
    """Run one docker compose command for the repo stack."""

    return subprocess.run(
        ["docker", "compose", *args],
        cwd=_ROOT,
        env=_compose_env(),
        text=True,
        capture_output=True,
        check=True,
    )


def _wait_for_http_ready(url: str, *, timeout_seconds: float) -> httpx.Response:
    """Poll one HTTP endpoint until it returns a success response."""

    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None

    with httpx.Client(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
        while time.monotonic() < deadline:
            try:
                response = client.get(url)
                if response.status_code < 500:
                    return response
            except httpx.HTTPError as exc:
                last_error = exc

            time.sleep(2.0)

    if last_error is not None:
        raise AssertionError(f"Timed out waiting for {url}: {last_error}") from last_error
    raise AssertionError(f"Timed out waiting for {url}")


@pytest.fixture(scope="session")
def e2e_endpoints() -> dict[str, str]:
    """Provide the base URLs for the deployed stack."""

    base_url = os.getenv("E2E_BASE_URL")
    if base_url:
        return {
            "base_url": base_url.rstrip("/"),
            "prometheus_url": os.getenv("E2E_PROMETHEUS_URL", _DEFAULT_PROMETHEUS_URL).rstrip("/"),
        }

    if os.getenv("E2E_MANAGE_STACK") != "1":
        pytest.skip(
            "Set E2E_BASE_URL to target an existing stack or E2E_MANAGE_STACK=1 to run docker compose."
        )

    if shutil.which("docker") is None:
        pytest.skip("docker is not available in this environment.")

    try:
        _run_compose("up", "-d", "--build")
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip()
        stdout = exc.stdout.strip()
        raise AssertionError(
            "docker compose up failed.\n"
            f"stdout:\n{stdout}\n\nstderr:\n{stderr}"
        ) from exc

    base_url = _DEFAULT_BASE_URL
    prometheus_url = _DEFAULT_PROMETHEUS_URL

    try:
        _wait_for_http_ready(
            f"{base_url}/ml-api/health",
            timeout_seconds=_STARTUP_TIMEOUT_SECONDS,
        )
        _wait_for_http_ready(
            f"{prometheus_url}/-/healthy",
            timeout_seconds=_STARTUP_TIMEOUT_SECONDS,
        )
        yield {
            "base_url": base_url,
            "prometheus_url": prometheus_url,
        }
    finally:
        _run_compose("down", "-v", "--remove-orphans")


def test_nginx_health_endpoint_returns_ok(e2e_endpoints: dict[str, str]) -> None:
    """Nginx should expose the API health endpoint under the prefixed route."""

    response = httpx.get(
        f"{e2e_endpoints['base_url']}/ml-api/health",
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metrics_endpoint_exposes_prometheus_payload(e2e_endpoints: dict[str, str]) -> None:
    """The stack should expose the Prometheus scrape payload through nginx."""

    response = httpx.get(
        f"{e2e_endpoints['base_url']}/ml-api/api/v1/metrics",
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "ml_api_http_requests_total" in response.text
    assert "ml_dependency_up" in response.text


def test_prometheus_container_is_healthy(e2e_endpoints: dict[str, str]) -> None:
    """The Prometheus service should come up alongside the serving stack."""

    response = httpx.get(
        f"{e2e_endpoints['prometheus_url']}/-/healthy",
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )

    assert response.status_code == 200
    assert response.text.strip() == "Prometheus Server is Healthy."


def test_detect_endpoint_accepts_real_image_upload(
    e2e_endpoints: dict[str, str],
    sample_image_bytes: bytes,
) -> None:
    """The deployed stack should handle one real object detection request."""

    response = httpx.post(
        f"{e2e_endpoints['base_url']}/ml-api/api/v1/detect",
        files={"file": ("image.png", sample_image_bytes, "image/png")},
        timeout=30.0,
    )

    assert response.status_code == 200
    payload = response.json()
    assert "detections" in payload
    assert isinstance(payload["detections"], list)
