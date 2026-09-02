from __future__ import annotations

import asyncio
import ipaddress
import logging
import mimetypes
import os
from typing import Any
from urllib.parse import urlsplit

import httpx

from config.settings import get_settings
from layers.L3_visual.providers.base import VideoResult

log = logging.getLogger(__name__)

DEFAULT_MEDIA_BASE_URL = "https://api.muapi.ai/api/v1"
DEFAULT_VIDEO_MODEL = "seedance-2-image-to-video-fast"
STANDARD_VIDEO_MODEL = "seedance-2-image-to-video"
VIDEO_RATIOS = {"21:9", "16:9", "4:3", "1:1", "3:4", "9:16"}


def is_muapi_provider(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"muapi", "mu_api", "mu-api"}


def _api_key() -> str:
    cfg = get_settings()
    key = getattr(cfg, "muapi_api_key", "") or os.getenv("MUAPI_API_KEY") or os.getenv("MU_API_KEY")
    if not key:
        raise RuntimeError("MUAPI_API_KEY is required for MuAPI video generation")
    return key


def _media_base_url() -> str:
    cfg = get_settings()
    return (
        os.getenv("MUAPI_BASE_URL")
        or os.getenv("MUAPI_MEDIA_BASE_URL")
        or getattr(cfg, "muapi_media_base_url", "")
        or DEFAULT_MEDIA_BASE_URL
    ).rstrip("/")


def _headers() -> dict[str, str]:
    return {"x-api-key": _api_key(), "Content-Type": "application/json"}


def _prediction_id(payload: dict[str, Any]) -> str | None:
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    for key in ("request_id", "id", "prediction_id", "task_id"):
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _status_payload(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data")
    return data if isinstance(data, dict) else payload


def _validate_https_url(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"MuAPI {label} response did not include a URL")
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise RuntimeError(f"MuAPI {label} response included an invalid URL") from exc

    if parsed.scheme != "https" or not hostname or parsed.username or parsed.password or port:
        raise RuntimeError(f"MuAPI {label} response included an insecure URL")
    hostname = hostname.lower().rstrip(".")
    if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(".local"):
        raise RuntimeError(f"MuAPI {label} response targeted a local host")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if address and (address.is_private or address.is_loopback or address.is_link_local or address.is_reserved):
        raise RuntimeError(f"MuAPI {label} response targeted a private host")
    return value


def _collect_video_urls(value: Any) -> list[str]:
    urls: list[str] = []
    if isinstance(value, str):
        if value.startswith(("https://", "http://")):
            urls.append(value)
        return urls
    if isinstance(value, list):
        for item in value:
            urls.extend(_collect_video_urls(item))
        return urls
    if not isinstance(value, dict):
        return urls

    for key in ("video_url", "video", "url", "download_url", "output_url"):
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate.startswith(("https://", "http://")):
            urls.append(candidate)
    for key in ("output", "result", "data", "outputs", "results"):
        if key in value:
            urls.extend(_collect_video_urls(value[key]))
    return urls


def _duration(duration_sec: int) -> int:
    try:
        value = int(duration_sec)
    except (TypeError, ValueError):
        value = 5
    return max(4, min(15, value))


def _ratio(aspect_ratio: str) -> str:
    ratio = str(aspect_ratio or "").strip()
    return ratio if ratio in VIDEO_RATIOS else "9:16"


def _model(quality: str) -> str:
    cfg = get_settings()
    configured = getattr(cfg, "muapi_video_model", "") or DEFAULT_VIDEO_MODEL
    if str(quality or "").strip().lower() in {"pro", "quality", "high"}:
        if configured == DEFAULT_VIDEO_MODEL:
            return STANDARD_VIDEO_MODEL
    return configured


async def _upload_image(client: httpx.AsyncClient, image_path: str, headers: dict[str, str]) -> str:
    with open(image_path, "rb") as image_file:
        content = image_file.read()
    content_type = mimetypes.guess_type(image_path)[0] or "application/octet-stream"
    files = {"file": (os.path.basename(image_path), content, content_type)}
    response = await client.post(f"{_media_base_url()}/upload_file", headers={"x-api-key": headers["x-api-key"]}, files=files)
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    return _validate_https_url(data.get("url"), "upload")


async def _poll_prediction(
    client: httpx.AsyncClient,
    request_id: str,
    headers: dict[str, str],
    *,
    timeout_sec: int,
    poll_interval_sec: float,
) -> dict[str, Any]:
    deadline = asyncio.get_running_loop().time() + timeout_sec
    while True:
        response = await client.get(
            f"{_media_base_url()}/predictions/{request_id}/result",
            headers={"x-api-key": headers["x-api-key"]},
        )
        response.raise_for_status()
        data = _status_payload(response.json())
        status = str(data.get("status", "")).strip().lower()
        if status in {"completed", "succeeded", "success"}:
            return data
        if status in {"failed", "error", "cancelled", "canceled", "timeout"}:
            message = data.get("error") or data.get("message") or status
            raise RuntimeError(f"MuAPI video generation failed: {message}")
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError(f"MuAPI video generation timed out for request {request_id}")
        await asyncio.sleep(poll_interval_sec)


async def _download(client: httpx.AsyncClient, url: str, output_path: str) -> None:
    safe_url = _validate_https_url(url, "video output")
    response = await client.get(safe_url)
    response.raise_for_status()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as output_file:
        output_file.write(response.content)


async def image_to_video(
    image_path: str,
    prompt: str,
    output_path: str,
    duration_sec: int = 5,
    aspect_ratio: str = "9:16",
    quality: str = "standard",
    character_ref_path: str | None = None,
    camera_control: dict | None = None,
    timeout_sec: int = 600,
) -> VideoResult:
    """Generate a video from a local start frame through MuAPI Seedance 2."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"MuAPI start frame not found: {image_path}")
    if character_ref_path:
        log.info("[MuAPI] character_ref_path is ignored by the Seedance 2 schema")
    if camera_control:
        log.info("[MuAPI] camera_control is ignored by the Seedance 2 schema")

    duration = _duration(duration_sec)
    model_id = _model(quality)
    payload = {
        "prompt": prompt,
        "images_list": [],
        "aspect_ratio": _ratio(aspect_ratio),
        "duration": duration,
    }
    headers = _headers()

    async with httpx.AsyncClient(timeout=timeout_sec) as client:
        image_url = await _upload_image(client, image_path, headers)
        payload["images_list"] = [image_url]
        response = await client.post(
            f"{_media_base_url()}/{model_id}",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        request_id = _prediction_id(response.json())
        if not request_id:
            raise RuntimeError("MuAPI video generation returned no prediction id")

        result_payload = await _poll_prediction(
            client,
            request_id,
            headers,
            timeout_sec=timeout_sec,
            poll_interval_sec=5,
        )
        urls = [_validate_https_url(url, "video output") for url in _collect_video_urls(result_payload)]
        if not urls:
            raise RuntimeError("MuAPI video generation completed without an output URL")
        await _download(client, urls[0], output_path)

    return VideoResult(
        url=urls[0],
        local_path=output_path,
        duration_sec=duration,
        task_id=request_id,
        model=model_id,
    )
