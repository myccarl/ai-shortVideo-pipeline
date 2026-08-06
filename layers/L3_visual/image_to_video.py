from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import tempfile

from config.settings import get_settings
from layers.L2_creative.style_engine import StyleTemplate
from core.langfuse_client import observe
from layers.L3_visual.prompt_safety import (
    TEXTLESS_NEGATIVE_PROMPT,
    TEXTLESS_VISUAL_GUARD,
    fit_visual_prompt,
    sanitize_visual_prompt,
)
from layers.L3_visual.providers.base import VideoResult
from layers.L3_visual.providers.atlascloud import (
    image_to_video as _atlascloud_image_to_video,
    is_atlascloud_provider,
)
from layers.L3_visual.providers.kling_v3 import image_to_video as _kling_image_to_video
from layers.L3_visual.text_artifact_guard import inspect_text_artifacts
from layers.L3_visual.text_to_image import generate_image

log = logging.getLogger(__name__)

_VIDEO_ARTIFACT_MAX_ATTEMPTS = 2
_VIDEO_ARTIFACT_SAMPLE_FPS = 1
_VIDEO_ARTIFACT_MAX_SEC = 8
_VIDEO_ARTIFACT_MAX_FRAMES = 6
_VIDEO_ARTIFACT_MIN_FRAME_HITS = 2
_TEXT_ARTIFACT_GUARD_BLOCKING = False
_PROMPT_MAX_LEN = 2400
_SINGLE_FRAME_BLOCK_TYPES = {"readable_text", "logo", "watermark", "subtitle"}


async def image_to_video(
    image_path: str,
    prompt: str,
    output_path: str,
    duration_sec: int = 5,
    aspect_ratio: str = "9:16",
    quality: str = "standard",
    character_ref_path: str | None = None,
    camera_control: dict | None = None,
) -> VideoResult:
    """Route image-to-video generation to the configured visual backend."""
    cfg = get_settings()
    if is_atlascloud_provider(getattr(cfg, "visual_video_provider", "kling")):
        return await _atlascloud_image_to_video(
            image_path=image_path,
            prompt=prompt,
            output_path=output_path,
            duration_sec=duration_sec,
            aspect_ratio=aspect_ratio,
            quality=quality,
            character_ref_path=character_ref_path,
            camera_control=camera_control,
        )
    return await _kling_image_to_video(
        image_path=image_path,
        prompt=prompt,
        output_path=output_path,
        duration_sec=duration_sec,
        aspect_ratio=aspect_ratio,
        quality=quality,
        character_ref_path=character_ref_path,
        camera_control=camera_control,
    )


def _fit_prompt_length(prompt: str, max_len: int = _PROMPT_MAX_LEN) -> str:
    return fit_visual_prompt(prompt, max_len=max_len)


def _stronger_textless_prompt(prompt: str) -> str:
    base = str(prompt or "").strip()
    marker = f", {TEXTLESS_VISUAL_GUARD}"
    if base.endswith(marker):
        base = base[: -len(marker)].rstrip(" ,")
    elif base == TEXTLESS_VISUAL_GUARD:
        base = ""
    return _fit_prompt_length(sanitize_visual_prompt(
        f"{base}, replace every text-like area with pure geometric light, "
        "blank glass, clean highlights, no glyph-like marks, no logo-like symbols, "
        "no product branding, no interface shapes with letters"
    ))


def _inspect_video_text_artifacts(
    video_path: str,
    *,
    fps: int = _VIDEO_ARTIFACT_SAMPLE_FPS,
    max_sec: int = _VIDEO_ARTIFACT_MAX_SEC,
    max_frames: int = _VIDEO_ARTIFACT_MAX_FRAMES,
):
    frame_dir = tempfile.mkdtemp(prefix="artifact_frames_")
    try:
        frames = extract_frames(video_path, frame_dir, fps=fps, max_sec=max_sec)
        artifact_reports = []
        for frame_path in frames[:max_frames]:
            report = inspect_text_artifacts(frame_path)
            if report.has_artifacts:
                report.frame_path = frame_path
                artifact_reports.append(report)
                artifact_type = getattr(report, "artifact_type", "")
                if artifact_type in _SINGLE_FRAME_BLOCK_TYPES:
                    return report
                if len(artifact_reports) >= _VIDEO_ARTIFACT_MIN_FRAME_HITS:
                    return report
    finally:
        for root, _dirs, files in os.walk(frame_dir, topdown=False):
            for name in files:
                try:
                    os.unlink(os.path.join(root, name))
                except OSError:
                    pass
        try:
            os.rmdir(frame_dir)
        except OSError:
            pass
    return None


@observe(name="image_to_video", as_type="generation")
async def generate_clip(
    image_prompt: str,
    kling_prompt: str,
    output_path: str,
    style: StyleTemplate,
    duration_sec: int = 5,
    quality: str = "standard",
    character_ref_path: str | None = None,
    first_frame_path: str | None = None,
    camera_control: dict | None = None,
    storyboard_id: str | None = None,
    clip_no: int | None = None,
) -> VideoResult:
    """
    Complete pipeline: text/image prompt -> first frame -> image-to-video.
    """
    work_dir = os.path.dirname(output_path) or "."
    os.makedirs(work_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(output_path))[0]
    generated_first_frame_path = os.path.join(work_dir, f"{base_name}_first_frame.png")
    safe_image_prompt = _fit_prompt_length(sanitize_visual_prompt(style.enrich_image_prompt(image_prompt)))
    safe_kling_prompt = _fit_prompt_length(sanitize_visual_prompt(kling_prompt))
    last_report = None
    first_frame_clip_score: float | None = None
    first_frame_clip_passed = True
    first_frame_clip_warning = ""

    for clip_attempt in range(_VIDEO_ARTIFACT_MAX_ATTEMPTS):
        image_prompt_for_attempt = safe_image_prompt
        kling_prompt_for_attempt = safe_kling_prompt
        chained_first_frame = first_frame_path

        if clip_attempt:
            image_prompt_for_attempt = _stronger_textless_prompt(safe_image_prompt)
            kling_prompt_for_attempt = _stronger_textless_prompt(safe_kling_prompt)
            chained_first_frame = None
            if first_frame_path:
                log.warning("[pipeline] video artifact retry drops chained first frame reuse")

        if chained_first_frame:
            if not os.path.isfile(chained_first_frame):
                raise FileNotFoundError(f"链式首帧不存在: {chained_first_frame}")
            image_path_for_video = chained_first_frame
            log.info("[pipeline] step1 reuse chained first frame -> %s", chained_first_frame)
        else:
            log.info(
                "[pipeline] step1 generate first frame -> %s (char_ref=%s)",
                generated_first_frame_path,
                bool(character_ref_path),
            )
            for frame_attempt in range(2):
                attempt_prompt = image_prompt_for_attempt
                if frame_attempt:
                    attempt_prompt = _stronger_textless_prompt(image_prompt_for_attempt)
                image_result = await generate_image(
                    prompt=attempt_prompt,
                    output_path=generated_first_frame_path,
                    negative_prompt=style.get_negative_prompt(TEXTLESS_NEGATIVE_PROMPT),
                    aspect_ratio=style.aspect_ratio,
                    character_ref_path=character_ref_path,
                    storyboard_id=storyboard_id,
                    clip_no=clip_no,
                )
                first_frame_clip_score = image_result.clip_score
                first_frame_clip_passed = image_result.clip_passed
                first_frame_clip_warning = image_result.clip_warning
                report = await asyncio.to_thread(inspect_text_artifacts, generated_first_frame_path)
                if not report.has_artifacts:
                    break
                log.warning(
                    "[pipeline] first frame text artifact advisory, retry=%d blocking=%s type=%s confidence=%.2f evidence=%s reason=%s",
                    frame_attempt + 1,
                    _TEXT_ARTIFACT_GUARD_BLOCKING,
                    getattr(report, "artifact_type", "unknown"),
                    float(getattr(report, "confidence", 1.0)),
                    getattr(report, "evidence", ""),
                    report.reason,
                )
                if not _TEXT_ARTIFACT_GUARD_BLOCKING:
                    break
            image_path_for_video = generated_first_frame_path

        log.info("[pipeline] step2 image-to-video duration=%ds -> %s", duration_sec, output_path)
        result = await image_to_video(
            image_path=image_path_for_video,
            prompt=kling_prompt_for_attempt,
            output_path=output_path,
            duration_sec=duration_sec,
            aspect_ratio=style.aspect_ratio,
            quality=quality,
            character_ref_path=character_ref_path,
            camera_control=camera_control,
        )
        result.clip_score = first_frame_clip_score
        result.clip_passed = first_frame_clip_passed
        result.clip_warning = first_frame_clip_warning

        artifact_report = await asyncio.to_thread(_inspect_video_text_artifacts, output_path)
        if not artifact_report:
            return result

        last_report = artifact_report
        log.warning(
            "[pipeline] sampled video text artifact advisory, clip attempt=%d blocking=%s type=%s confidence=%.2f evidence=%s reason=%s frame=%s",
            clip_attempt + 1,
            _TEXT_ARTIFACT_GUARD_BLOCKING,
            getattr(artifact_report, "artifact_type", "unknown"),
            float(getattr(artifact_report, "confidence", 1.0)),
            getattr(artifact_report, "evidence", ""),
            artifact_report.reason,
            getattr(artifact_report, "frame_path", ""),
        )
        if not _TEXT_ARTIFACT_GUARD_BLOCKING:
            return result

    raise RuntimeError(
        f"video text artifact guard failed after {_VIDEO_ARTIFACT_MAX_ATTEMPTS} attempts: "
        f"{last_report.reason if last_report else 'unknown artifact'}"
    )


MAX_SHOTS_DEFAULT = 4  # 单角色连续镜数硬上限（与 characters.yaml 默认值对齐）


async def generate_clip_sequence(
    clips: list[dict],
    output_dir: str,
    style: StyleTemplate,
    character_ref_path: str | None = None,
    max_shots: int | None = None,
    chain_frames: bool = True,
) -> list[str]:
    """
    按评估结果依次生成多个视频片段。

    clips 格式:
        [
            {
                "clip_no": 1,
                "duration_sec": 5,
                "scene_summary": "...",
                "image_prompt": "...",
                "kling_prompt": "...",
                "quality": "standard"
            },
        ]

    max_shots: 单角色连续镜数上限（默认 MAX_SHOTS_DEFAULT=4）。
    超过时直接抛出 ValueError，不静默截断（修复硬伤：超限必须显式报错）。
    """
    limit = max_shots if max_shots is not None else MAX_SHOTS_DEFAULT
    if len(clips) > limit:
        raise ValueError(
            f"镜数硬限制触发：请求 {len(clips)} 镜，当前角色上限 {limit} 镜。"
            f"请减少 clips 数量或在 characters.yaml 中调整 max_consecutive_shots。"
            f"（不静默失败，必须显式处理）"
        )

    os.makedirs(output_dir, exist_ok=True)
    paths = []
    prev_tail: str | None = None

    for idx, clip in enumerate(clips):
        clip_no = clip["clip_no"]
        duration = clip.get("duration_sec", 5)
        quality = clip.get("quality", "standard")
        image_prompt = clip.get("image_prompt", clip.get("scene_summary", ""))
        kling_prompt = clip.get("kling_prompt", clip.get("scene_summary", ""))
        camera_control = clip.get("camera_control")
        out_path = os.path.join(output_dir, f"clip_{clip_no:02d}.mp4")
        first_frame_override = prev_tail if chain_frames and prev_tail else None

        log.info(
            "[sequence] clip=%d | %ds | %s | chain=%s | %s...",
            clip_no,
            duration,
            quality,
            bool(first_frame_override),
            image_prompt[:40],
        )
        await generate_clip(
            image_prompt=image_prompt,
            kling_prompt=kling_prompt,
            output_path=out_path,
            style=style,
            duration_sec=duration,
            quality=quality,
            character_ref_path=character_ref_path,
            first_frame_path=first_frame_override,
            camera_control=camera_control,
        )
        paths.append(out_path)

        if chain_frames and idx + 1 < len(clips):
            prev_tail = os.path.join(output_dir, f"tail_{clip_no:02d}.png")
            try:
                extract_last_frame(out_path, prev_tail)
                log.info("[sequence] extracted tail frame for clip %d -> %s", clip_no, os.path.basename(prev_tail))
            except Exception as exc:
                log.warning("[sequence] tail frame extraction failed, next clip falls back to text-to-image: %s", exc)
                prev_tail = None

    return paths


def extract_last_frame(video_path: str, output_path: str) -> str:
    """Extract a clip tail frame for chaining into the next image-to-video call.

    可灵 mp4 容器有 'Late SEI is not implemented' 警告，紧贴末尾 seek 抽不到 keyframe。
    采用 3 策略链式 fallback，命中任一即返回；全失败抛 RuntimeError。
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    def _has_frame(p: str) -> bool:
        return os.path.exists(p) and os.path.getsize(p) > 0

    # 策略 1（首选）：-sseof -1.0 + -update 1
    # 反向 seek 到结尾前 1 秒，-update 1 让 ffmpeg 把每个解码帧覆盖写到同一个文件
    # 解码完成后 output_path 自然就是最后一帧。对 Late SEI 容器最鲁棒
    try:
        log.info("[抽尾帧] strategy=sseof_update path=%s", video_path)
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-sseof", "-1.0",
                "-i", video_path,
                "-update", "1",
                "-q:v", "2",
                output_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if _has_frame(output_path):
            return output_path
        log.warning("[抽尾帧] strategy=sseof_update 输出为空，降级到 ffprobe_seek")
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        log.warning("[抽尾帧] strategy=sseof_update 失败: %s", exc)

    # 策略 2：ffprobe duration - 0.5s + input-side -ss（旧 ffprobe 路径加大 buffer）
    try:
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        duration_sec = float(probe.stdout.strip())
        if duration_sec <= 0:
            raise ValueError(f"invalid duration={duration_sec}")
        # 关键修复：从 0.04s 改成 0.5s，避开 Late SEI 容器最后段 keyframe 缺失
        seek_sec = max(0.0, duration_sec - 0.5)
        log.info(
            "[抽尾帧] strategy=ffprobe_seek duration=%.2fs seek=%.2fs path=%s",
            duration_sec, seek_sec, video_path,
        )
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", f"{seek_sec:.3f}",
                "-i", video_path,
                "-frames:v", "1",
                "-q:v", "2",
                output_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if _has_frame(output_path):
            return output_path
        log.warning("[抽尾帧] strategy=ffprobe_seek 输出为空，降级到 sseof_legacy")
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as exc:
        log.warning("[抽尾帧] strategy=ffprobe_seek 失败: %s", exc)

    # 策略 3（最后兜底）：旧 -sseof -0.04 路径（理论上最差，但万一前两个都坏可能它能撑住）
    try:
        log.info("[抽尾帧] strategy=sseof_legacy path=%s", video_path)
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-sseof", "-0.04",
                "-i", video_path,
                "-update", "1",
                "-frames:v", "1",
                "-q:v", "2",
                output_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if _has_frame(output_path):
            return output_path
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        log.warning("[抽尾帧] strategy=sseof_legacy 失败: %s", exc)

    raise RuntimeError(f"抽末帧失败或文件为空: {output_path}")


def concat_clips(clip_paths: list[str], output_path: str) -> str:
    abs_paths = [os.path.abspath(p) for p in clip_paths]
    abs_output = os.path.abspath(output_path)
    list_content = "\n".join(f"file '{p}'" for p in abs_paths)

    list_file = os.path.join(os.path.dirname(abs_output), "concat_list.txt")
    with open(list_file, "w") as f:
        f.write(list_content)

    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", abs_output],
        check=True,
    )
    os.unlink(list_file)
    return abs_output


def extract_frames(
    video_path: str,
    output_dir: str,
    fps: int = 1,
    max_sec: int = 30,
) -> list[str]:
    import glob as globmod

    os.makedirs(output_dir, exist_ok=True)
    out_pattern = os.path.join(output_dir, "frame_%03d.jpg")
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vf", f"fps={fps},scale=720:-2", "-t", str(max_sec), out_pattern],
        check=True,
    )
    return sorted(globmod.glob(os.path.join(output_dir, "frame_*.jpg")))
