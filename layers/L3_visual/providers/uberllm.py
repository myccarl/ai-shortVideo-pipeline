from __future__ import annotations

import asyncio
import logging
import mimetypes
import os
import uuid
from typing import Any

import httpx

from config.settings import get_settings
from layers.L3_visual.providers.base import ImageResult, VideoResult

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.uberllm.dev/v1"
DEFAULT_IMAGE_MODEL = "venice/seedream-v5-lite"
DEFAULT_VIDEO_MODEL = "bytedance/seedance-2-0-fast-image-to-video"
PRO_VIDEO_MODEL = "bytedance/seedance-2-0-image-to-video"
DEFAULT_VIDEO_RESOLUTION = "720p"


def is_uberllm_provider(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"uberllm", "uber_llm", "uber-llm"}


def _api_key() -> str:
    cfg = get_settings()
    key = getattr(cfg, "uberllm_api_key", "") or os.getenv("UBERLLM_API_KEY")
    if not key:
        raise RuntimeError("UBERLLM_API_KEY is required for UberLLM media generation")
    return key


def _base_url() -> str:
    cfg = get_settings()
    return (
        getattr(cfg, "uberllm_base_url", "")
        or os.getenv("UBERLLM_BASE_URL")
        or DEFAULT_BASE_URL
    ).rstrip("/")


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_api_key()}"}


async def _model_options(client: httpx.AsyncClient, model_id: str) -> dict[str, Any]:
    """Request fields the model accepts, from the public GET /models/<id>.

    Each model takes its own clip lengths, resolutions and shapes, and a value it
    does not take is refused, so the payload is fitted to this list.
    """
    try:
        resp = await client.get(f"{_base_url()}/models/{model_id}")
        resp.raise_for_status()
        props = (resp.json().get("parameters") or {}).get("properties")
    except (httpx.HTTPError, ValueError, AttributeError) as exc:
        log.warning("[UberLLM] could not read options for %s: %s", model_id, exc)
        return {}
    return props if isinstance(props, dict) else {}


def _allowed(options: dict[str, Any], field: str) -> list[Any] | None:
    spec = options.get(field)
    if not isinstance(spec, dict):
        return None
    values = spec.get("enum")
    return list(values) if isinstance(values, list) else []


def _fit_choice(options: dict[str, Any], field: str, value: Any) -> Any:
    """Return value if the model takes this field with this value, else None."""
    allowed = _allowed(options, field)
    if allowed is None or value in (None, ""):
        return None
    if allowed and value not in allowed:
        return None
    return value


def _fit_duration(options: dict[str, Any], duration_sec: int | None) -> int:
    try:
        duration = int(duration_sec or 5)
    except (TypeError, ValueError):
        duration = 5
    lengths = sorted(int(v) for v in (_allowed(options, "duration_seconds") or []) if isinstance(v, (int, float)))
    if not lengths:
        return duration
    # The clip is cut to the voiceover later, so round up to the next length the model makes.
    return next((v for v in lengths if v >= duration), lengths[-1])


def _closest_size(options: dict[str, Any], aspect_ratio: str) -> str | None:
    """For models that take a pixel size ("1024x1536") instead of a ratio."""
    try:
        w, h = (float(x) for x in str(aspect_ratio).split(":"))
        target = w / h
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    best = None
    for size in _allowed(options, "size") or []:
        try:
            sw, sh = (float(x) for x in str(size).lower().split("x"))
        except ValueError:
            continue
        diff = abs(sw / sh - target)
        if best is None or diff < best[0]:
            best = (diff, size)
    return best[1] if best else None


def _image_payload(
    options: dict[str, Any],
    *,
    model_id: str,
    prompt: str,
    negative_prompt: str,
    aspect_ratio: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"model": model_id, "prompt": prompt, "n": 1}
    ratio = _fit_choice(options, "aspect_ratio", aspect_ratio)
    size = None if ratio else _closest_size(options, aspect_ratio)
    if ratio:
        payload["aspect_ratio"] = ratio
    elif size:
        payload["size"] = size
    elif aspect_ratio:
        log.info("[UberLLM] %s does not take aspect_ratio=%s, using its default", model_id, aspect_ratio)
    if negative_prompt:
        if "negative_prompt" in options:
            payload["negative_prompt"] = negative_prompt
        else:
            log.info("[UberLLM] negative_prompt is ignored by %s schema", model_id)
    return payload


def _video_payload(
    options: dict[str, Any],
    *,
    model_id: str,
    prompt: str,
    image_ref: str,
    duration_sec: int | None,
    aspect_ratio: str,
    resolution: str,
    audio: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model_id,
        "prompt": prompt,
        "image_url": image_ref,
        "duration_seconds": _fit_duration(options, duration_sec),
    }
    for field, value in (("resolution", resolution), ("aspect_ratio", aspect_ratio)):
        fitted = _fit_choice(options, field, value)
        if fitted:
            payload[field] = fitted
    if "audio" in options:
        payload["audio"] = bool(audio)
    return payload


def _video_model(quality: str) -> str:
    cfg = get_settings()
    configured = getattr(cfg, "uberllm_video_model", "") or DEFAULT_VIDEO_MODEL
    if str(quality or "").strip().lower() in {"pro", "quality", "high"} and configured == DEFAULT_VIDEO_MODEL:
        return PRO_VIDEO_MODEL
    return configured


async def _upload_image(client: httpx.AsyncClient, image_path: str) -> str:
    """Upload the start frame to /files (kept 24h) and return its file id."""
    with open(image_path, "rb") as f:
        content = f.read()
    content_type = mimetypes.guess_type(image_path)[0] or "image/png"
    resp = await client.post(
        f"{_base_url()}/files",
        headers=_auth_headers(),
        files={"file": (os.path.basename(image_path), content, content_type)},
    )
    resp.raise_for_status()
    file_id = resp.json().get("id")
    if not isinstance(file_id, str) or not file_id:
        raise RuntimeError("UberLLM file upload returned no file id")
    return file_id


async def _poll_job(
    client: httpx.AsyncClient,
    job_id: str,
    *,
    timeout_sec: int,
    poll_interval_sec: float,
) -> dict[str, Any]:
    deadline = asyncio.get_running_loop().time() + timeout_sec
    while True:
        resp = await client.get(f"{_base_url()}/video/generations/{job_id}", headers=_auth_headers())
        resp.raise_for_status()
        job = resp.json()
        status = str(job.get("status", "")).lower()
        if status == "succeeded":
            return job
        if status in {"failed", "canceled", "expired"}:
            error = job.get("error") if isinstance(job.get("error"), dict) else {}
            raise RuntimeError(f"UberLLM video generation {status}: {error.get('message') or status}")
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError(f"UberLLM video generation timed out for job {job_id}")
        await asyncio.sleep(poll_interval_sec)


async def _download(client: httpx.AsyncClient, url: str, output_path: str) -> None:
    # Result links live under the API and need the same key.
    headers = _auth_headers() if url.startswith(_base_url()) else None
    resp = await client.get(url, headers=headers, follow_redirects=True)
    resp.raise_for_status()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(resp.content)


async def text_to_image(
    prompt: str,
    output_path: str,
    *,
    negative_prompt: str = "",
    aspect_ratio: str = "9:16",
    model: str = "",
    character_ref_path: str | None = None,
    positive_suffix: str = "",
    timeout_sec: int = 300,
) -> ImageResult:
    cfg = get_settings()
    model_id = model or getattr(cfg, "uberllm_image_model", "") or DEFAULT_IMAGE_MODEL
    full_prompt = f"{prompt}, {positive_suffix.strip()}" if positive_suffix else prompt
    if character_ref_path:
        log.info("[UberLLM] character_ref_path is ignored by %s schema", model_id)

    async with httpx.AsyncClient(timeout=timeout_sec) as client:
        options = await _model_options(client, model_id)
        payload = _image_payload(
            options,
            model_id=model_id,
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            aspect_ratio=aspect_ratio,
        )
        resp = await client.post(
            f"{_base_url()}/images/generations",
            headers={**_auth_headers(), "Content-Type": "application/json"},
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json().get("data") or []
        url = data[0].get("url") if data and isinstance(data[0], dict) else None
        if not url:
            raise RuntimeError("UberLLM image generation returned no image URL")
        await _download(client, url, output_path)

    return ImageResult(url=url, local_path=output_path, model=model_id)


async def image_to_video(
    image_path: str,
    prompt: str,
    output_path: str,
    duration_sec: int = 5,
    aspect_ratio: str = "9:16",
    quality: str = "standard",
    character_ref_path: str | None = None,
    camera_control: dict | None = None,
    timeout_sec: int = 900,
) -> VideoResult:
    """Generate a video from a local start frame through UberLLM /video/generations."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"UberLLM start frame not found: {image_path}")
    cfg = get_settings()
    model_id = _video_model(quality)
    if character_ref_path:
        log.info("[UberLLM] character_ref_path is ignored by %s schema", model_id)
    if camera_control:
        log.info("[UberLLM] camera_control is ignored by %s schema", model_id)

    async with httpx.AsyncClient(timeout=timeout_sec) as client:
        options = await _model_options(client, model_id)
        file_id = await _upload_image(client, image_path)
        payload = _video_payload(
            options,
            model_id=model_id,
            prompt=prompt,
            image_ref=file_id,
            duration_sec=duration_sec,
            aspect_ratio=aspect_ratio,
            resolution=getattr(cfg, "uberllm_video_resolution", "") or DEFAULT_VIDEO_RESOLUTION,
            audio=bool(getattr(cfg, "uberllm_video_audio", False)),
        )
        resp = await client.post(
            f"{_base_url()}/video/generations",
            headers={
                **_auth_headers(),
                "Content-Type": "application/json",
                "Idempotency-Key": str(uuid.uuid4()),
            },
            json=payload,
        )
        resp.raise_for_status()
        job_id = resp.json().get("id")
        if not job_id:
            raise RuntimeError("UberLLM video generation returned no job id")

        job = await _poll_job(client, job_id, timeout_sec=timeout_sec, poll_interval_sec=5)
        results = [r for r in job.get("results") or [] if isinstance(r, dict) and r.get("url")]
        if not results:
            raise RuntimeError("UberLLM video generation completed without an output URL")
        await _download(client, results[0]["url"], output_path)

    return VideoResult(
        url=results[0]["url"],
        local_path=output_path,
        duration_sec=results[0].get("duration_seconds") or payload["duration_seconds"],
        task_id=job_id,
        model=job.get("model") or model_id,
    )
