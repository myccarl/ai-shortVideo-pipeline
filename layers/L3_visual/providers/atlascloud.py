from __future__ import annotations

import asyncio
import base64
import logging
import os
from typing import Any

import httpx

from config.settings import get_settings
from layers.L3_visual.providers.base import ImageResult, VideoResult

log = logging.getLogger(__name__)

DEFAULT_MEDIA_BASE_URL = "https://api.atlascloud.ai/api/v1"
DEFAULT_IMAGE_MODEL = "bytedance/seedream-v5.0-lite"
DEFAULT_VIDEO_MODEL = "bytedance/seedance-2.0-fast/image-to-video"
DEFAULT_VIDEO_RESOLUTION = "720p"

IMAGE_SIZES_BY_RATIO = {
    "1:1": "2048*2048",
    "4:3": "2304*1728",
    "3:4": "1728*2304",
    "16:9": "2848*1600",
    "9:16": "1600*2848",
    "21:9": "3136*1344",
}
VIDEO_RATIOS = {"16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "adaptive"}


def is_atlascloud_provider(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"atlas", "atlascloud", "atlas_cloud"}


def _api_key() -> str:
    cfg = get_settings()
    key = (
        getattr(cfg, "atlascloud_api_key", "")
        or os.getenv("ATLASCLOUD_API_KEY")
        or os.getenv("ATLAS_CLOUD_API_KEY")
    )
    if not key:
        raise RuntimeError("ATLASCLOUD_API_KEY is required for Atlas Cloud media generation")
    return key


def _media_base_url() -> str:
    cfg = get_settings()
    return (
        getattr(cfg, "atlascloud_media_base_url", "")
        or os.getenv("ATLASCLOUD_MEDIA_BASE_URL")
        or DEFAULT_MEDIA_BASE_URL
    ).rstrip("/")


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
    }


def _prediction_id(payload: dict[str, Any]) -> str | None:
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    for key in ("id", "prediction_id", "request_id", "task_id"):
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _status_payload(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data")
    return data if isinstance(data, dict) else payload


def _collect_urls(value: Any) -> list[str]:
    urls: list[str] = []
    if isinstance(value, str):
        if value.startswith(("http://", "https://")):
            urls.append(value)
        return urls
    if isinstance(value, list):
        for item in value:
            urls.extend(_collect_urls(item))
        return urls
    if isinstance(value, dict):
        for key in (
            "url",
            "image",
            "image_url",
            "video",
            "video_url",
            "download_url",
            "output_url",
        ):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.startswith(("http://", "https://")):
                urls.append(candidate)
        for key in ("outputs", "output", "result", "results", "images", "videos", "data"):
            if key in value:
                urls.extend(_collect_urls(value[key]))
    return urls


def _size_for_aspect_ratio(aspect_ratio: str) -> str:
    return IMAGE_SIZES_BY_RATIO.get(str(aspect_ratio or "").strip(), IMAGE_SIZES_BY_RATIO["9:16"])


def _video_ratio(aspect_ratio: str) -> str:
    ratio = str(aspect_ratio or "").strip()
    return ratio if ratio in VIDEO_RATIOS else "adaptive"


def _video_duration(duration_sec: int | None) -> int:
    try:
        duration = int(duration_sec or 5)
    except (TypeError, ValueError):
        duration = 5
    return max(4, min(15, duration))


def _output_format(output_path: str) -> str:
    suffix = os.path.splitext(output_path)[1].lower()
    return "png" if suffix == ".png" else "jpeg"


async def _poll_prediction(
    client: httpx.AsyncClient,
    task_id: str,
    headers: dict[str, str],
    *,
    timeout_sec: int,
    poll_interval_sec: float,
) -> dict[str, Any]:
    deadline = asyncio.get_running_loop().time() + timeout_sec
    while True:
        resp = await client.get(f"{_media_base_url()}/model/prediction/{task_id}", headers=headers)
        resp.raise_for_status()
        data = _status_payload(resp.json())
        status = str(data.get("status", "")).lower()
        if status in {"completed", "succeeded", "success"}:
            return data
        if status in {"failed", "error", "canceled", "cancelled"}:
            message = data.get("error") or data.get("message") or status
            raise RuntimeError(f"Atlas Cloud generation failed: {message}")
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError(f"Atlas Cloud generation timed out for task {task_id}")
        await asyncio.sleep(poll_interval_sec)


async def _download(client: httpx.AsyncClient, url: str, output_path: str) -> None:
    resp = await client.get(url)
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
    model_id = model or getattr(cfg, "atlascloud_image_model", "") or DEFAULT_IMAGE_MODEL
    full_prompt = f"{prompt}, {positive_suffix.strip()}" if positive_suffix else prompt
    if negative_prompt:
        log.info("[AtlasCloud] negative_prompt is ignored by %s schema", model_id)
    if character_ref_path:
        log.info("[AtlasCloud] character_ref_path is ignored by %s schema", model_id)

    payload = {
        "model": model_id,
        "prompt": full_prompt,
        "size": _size_for_aspect_ratio(aspect_ratio),
        "output_format": _output_format(output_path),
    }
    headers = _headers()

    async with httpx.AsyncClient(timeout=timeout_sec) as client:
        resp = await client.post(f"{_media_base_url()}/model/generateImage", headers=headers, json=payload)
        resp.raise_for_status()
        task_id = _prediction_id(resp.json())
        if not task_id:
            raise RuntimeError("Atlas Cloud image generation returned no prediction id")

        result_payload = await _poll_prediction(
            client,
            task_id,
            headers,
            timeout_sec=timeout_sec,
            poll_interval_sec=3,
        )
        urls = _collect_urls(result_payload)
        if not urls:
            raise RuntimeError("Atlas Cloud image generation completed without an output URL")

        await _download(client, urls[0], output_path)

    return ImageResult(url=urls[0], local_path=output_path, model=model_id)


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
    cfg = get_settings()
    model_id = getattr(cfg, "atlascloud_video_model", "") or DEFAULT_VIDEO_MODEL
    resolution = getattr(cfg, "atlascloud_video_resolution", "") or DEFAULT_VIDEO_RESOLUTION
    generate_audio = bool(getattr(cfg, "atlascloud_video_generate_audio", True))
    if character_ref_path:
        log.info("[AtlasCloud] character_ref_path is ignored by %s schema", model_id)
    if camera_control:
        log.info("[AtlasCloud] camera_control is ignored by %s schema", model_id)

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    duration = _video_duration(duration_sec)
    payload = {
        "model": model_id,
        "image": image_b64,
        "prompt": prompt,
        "duration": duration,
        "resolution": resolution,
        "ratio": _video_ratio(aspect_ratio),
        "bitrate_mode": "high" if quality == "pro" else "standard",
        "generate_audio": generate_audio,
        "watermark": False,
    }
    headers = _headers()

    async with httpx.AsyncClient(timeout=timeout_sec) as client:
        resp = await client.post(f"{_media_base_url()}/model/generateVideo", headers=headers, json=payload)
        resp.raise_for_status()
        task_id = _prediction_id(resp.json())
        if not task_id:
            raise RuntimeError("Atlas Cloud video generation returned no prediction id")

        result_payload = await _poll_prediction(
            client,
            task_id,
            headers,
            timeout_sec=timeout_sec,
            poll_interval_sec=5,
        )
        urls = _collect_urls(result_payload)
        if not urls:
            raise RuntimeError("Atlas Cloud video generation completed without an output URL")

        await _download(client, urls[0], output_path)

    return VideoResult(
        url=urls[0],
        local_path=output_path,
        duration_sec=duration,
        task_id=task_id,
        model=model_id,
    )
