from __future__ import annotations

import asyncio
import base64
import logging
import os

import httpx

from config.settings import get_settings
from core.langfuse_client import observe
from layers.L3_visual.prompt_safety import fit_visual_prompt
from layers.L3_visual.providers.base import ImageResult
from layers.L3_visual.providers.atlascloud import (
    is_atlascloud_provider,
    text_to_image as atlascloud_text_to_image,
)

log = logging.getLogger(__name__)

IMAGE_REFERENCE_PRIMARY = "subject"
IMAGE_REFERENCE_FALLBACK = "face"
IMAGE_PROMPT_MAX_LEN = 2400


def _encode_image_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


@observe(name="text_to_image", as_type="generation")
async def generate_image(
    prompt: str,
    output_path: str,
    negative_prompt: str = "",
    aspect_ratio: str = "9:16",
    model: str = "",
    character_ref_path: str | None = None,
    positive_suffix: str = "",
    storyboard_id: str | None = None,
    clip_no: int | None = None,
) -> ImageResult:
    """
    调用可灵 Image 文生图 API。
    当传入 character_ref_path 时，启用参考图模式保持人物一致性。
    """
    from layers.L3_visual.providers.kling_v3 import kling_image_headers

    cfg = get_settings()
    requested_model = model
    if not model:
        model = cfg.kling_image_model

    full_prompt = f"{prompt}, {positive_suffix.strip()}" if positive_suffix else prompt
    if len(full_prompt) > IMAGE_PROMPT_MAX_LEN:
        log.warning(
            "[文生图] prompt too long (%d), truncating to %d chars before submit",
            len(full_prompt),
            IMAGE_PROMPT_MAX_LEN,
        )
        full_prompt = fit_visual_prompt(full_prompt, max_len=IMAGE_PROMPT_MAX_LEN)

    if is_atlascloud_provider(getattr(cfg, "visual_image_provider", "kling")):
        result = await atlascloud_text_to_image(
            prompt=prompt,
            output_path=output_path,
            negative_prompt=negative_prompt,
            aspect_ratio=aspect_ratio,
            model=requested_model or getattr(cfg, "atlascloud_image_model", ""),
            character_ref_path=character_ref_path,
            positive_suffix=positive_suffix,
        )
        return await _attach_clip_consistency(
            result,
            image_path=output_path,
            prompt=full_prompt,
            storyboard_id=storyboard_id,
            clip_no=clip_no,
        )

    log.info(
        "[文生图] prompt=%s... aspect=%s model=%s ref=%s realism_suffix=%s",
        prompt[:60],
        aspect_ratio,
        model,
        bool(character_ref_path),
        bool(positive_suffix),
    )

    payload = {
        "model_name": model,
        "prompt": full_prompt,
        "aspect_ratio": aspect_ratio,
    }
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt

    ref_modes: list[str | None] = [None]
    if character_ref_path and os.path.isfile(character_ref_path):
        ref_b64 = _encode_image_base64(character_ref_path)
        payload["image"] = ref_b64
        log.info("[文生图] 角色参考图已注入: %s", character_ref_path)
        ref_modes = [IMAGE_REFERENCE_PRIMARY, IMAGE_REFERENCE_FALLBACK]

    async with httpx.AsyncClient(timeout=300) as c:
        resp_data = None
        used_ref_mode = None

        for ref_mode in ref_modes:
            attempt_payload = dict(payload)
            if ref_mode:
                attempt_payload["image_reference"] = ref_mode
            else:
                attempt_payload.pop("image_reference", None)

            log.info(
                "[文生图] 提交 model=%s aspect=%s negative=%s ref_mode=%s has_image=%s",
                attempt_payload.get("model_name"),
                attempt_payload.get("aspect_ratio"),
                bool(attempt_payload.get("negative_prompt")),
                ref_mode or "none",
                bool(attempt_payload.get("image")),
            )

            resp = await c.post(
                f"{cfg.kling_base_url}/v1/images/generations",
                headers=kling_image_headers(),
                json=attempt_payload,
            )
            log.info("[文生图] 响应 status=%s body=%s", resp.status_code, resp.text[:800])
            resp_data = resp.json()

            if resp_data.get("code") == 0:
                used_ref_mode = ref_mode
                break

            msg = str(resp_data.get("message", ""))
            if ref_mode == IMAGE_REFERENCE_PRIMARY and "image_reference value" in msg:
                log.warning(
                    "[文生图] ref_mode=%s 不被当前接口接受，回退到 %s",
                    ref_mode,
                    IMAGE_REFERENCE_FALLBACK,
                )
                continue

            raise RuntimeError(f"文生图提交失败: {resp_data.get('message')} | {resp_data}")

        if not resp_data or resp_data.get("code") != 0:
            raise RuntimeError(f"文生图提交失败: {resp_data}")

        task_id = resp_data["data"]["task_id"]
        log.info("[文生图] task_id=%s, 开始轮询 (ref_mode=%s)", task_id, used_ref_mode or "none")

        delay = 3
        elapsed = 0
        max_wait = 180

        while elapsed < max_wait:
            await asyncio.sleep(delay)
            elapsed += delay

            st = await c.get(
                f"{cfg.kling_base_url}/v1/images/generations/{task_id}",
                headers=kling_image_headers(),
            )
            data = st.json().get("data", {})
            status = data.get("task_status")

            if status == "succeed":
                images = data.get("task_result", {}).get("images", [])
                if not images:
                    raise RuntimeError("文生图成功但无图片返回")

                image_url = images[0]["url"]
                log.info("[文生图] 成功，下载图片")

                raw = await c.get(image_url)
                os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(raw.content)

                result = ImageResult(
                    url=image_url,
                    local_path=output_path,
                    width=images[0].get("width", 0),
                    height=images[0].get("height", 0),
                    model=model,
                )
                return await _attach_clip_consistency(
                    result,
                    image_path=output_path,
                    prompt=full_prompt,
                    storyboard_id=storyboard_id,
                    clip_no=clip_no,
                )

            if status == "failed":
                raise RuntimeError(f"文生图失败: {data.get('task_status_msg')}")

            delay = min(delay * 1.5, 15)

    raise RuntimeError(f"文生图超时（{max_wait}秒），task_id={task_id}")


# ── R5.1 多候选生成（不动旧 generate_image）───────────────────────────────


async def _attach_clip_consistency(
    result: ImageResult,
    *,
    image_path: str,
    prompt: str,
    storyboard_id: str | None = None,
    clip_no: int | None = None,
) -> ImageResult:
    """Attach best-effort CLIP consistency warning metadata to ImageResult."""
    cfg = get_settings()
    if not cfg.clip_consistency_enabled:
        return result
    try:
        from layers.L3_visual.clip_consistency import check_consistency

        consistency = await check_consistency(
            image_path=image_path,
            prompt=prompt,
            storyboard_id=storyboard_id,
            clip_no=clip_no,
        )
        result.clip_score = consistency.score
        result.clip_passed = consistency.passed
        result.clip_warning = consistency.warning_message
    except Exception as exc:
        log.warning("[clip] consistency check failed, continue without blocking: %s", exc)
    return result


async def generate_with_candidates(
    prompt: str,
    output_dir: str,
    *,
    n: int = 3,
    aspect_ratio: str = "9:16",
    character_ref_path: str | None = None,
    base_filename: str = "candidate",
    timeout_per_image: int = 60,
) -> list[ImageResult]:
    """Generate N image candidates concurrently.

    Each candidate receives a tiny prompt variation to nudge provider-side
    randomness. Individual failures are logged and skipped.
    """
    log.info("[multi_candidate] n=%d prompt=%s...", n, prompt[:80])
    os.makedirs(output_dir, exist_ok=True)

    async def _one(idx: int) -> ImageResult | None:
        variation_hint = ["", "subtle variation", "alternative angle perspective"][idx % 3]
        prompt_var = f"{prompt}, {variation_hint}" if variation_hint else prompt
        out_path = os.path.join(output_dir, f"{base_filename}_{idx:02d}.png")
        try:
            return await asyncio.wait_for(
                generate_image(
                    prompt=prompt_var,
                    output_path=out_path,
                    aspect_ratio=aspect_ratio,
                    character_ref_path=character_ref_path,
                ),
                timeout=timeout_per_image,
            )
        except Exception as e:
            log.warning("[multi_candidate] idx=%d failed: %s", idx, e)
            return None

    results = await asyncio.gather(*[_one(i) for i in range(n)])
    successful = [r for r in results if r is not None]
    log.info("[multi_candidate] %d/%d candidates succeeded", len(successful), n)
    return successful
