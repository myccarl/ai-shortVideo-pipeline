from __future__ import annotations

from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # ── DeepSeek ──
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"

    # ── GLM (ZhipuAI) ──
    glm_api_key: str = ""
    glm_model_id: str = "glm-4-plus"
    glm_vision_model: str = "glm-4v"

    # ── 可灵 Kling（图生视频，v2-5-turbo std） ──
    kling_access_key: str = ""
    kling_secret_key: str = ""
    kling_base_url: str = "https://api-beijing.klingai.com"
    kling_video_model: str = "kling-v2-5-turbo"
    kling_cost_5s: float = 1.20
    kling_cost_10s: float = 2.40
    # Phase P P11: Kling 3.0 native audio PoC. Default keeps the stitched i2v path.
    visual_generation_mode: str = "stitched_i2v"  # "stitched_i2v" | "kling3_native_audio"
    kling3_base_url: str = ""    # Empty means reuse kling_base_url.
    kling3_access_key: str = ""  # Empty means reuse kling_access_key.
    kling3_secret_key: str = ""  # Empty means reuse kling_secret_key.
    kling3_model: str = "kling-v3.0"

    # Langfuse end-to-end observability.
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_enabled: bool = False

    # D41-B: Chinese-CLIP keyframe/prompt consistency warning.
    clip_consistency_enabled: bool = True
    clip_consistency_threshold: float = 0.22

    # D41-C: AV sync rescue after hard drift.
    av_rescue_enabled: bool = True
    av_rescue_tempo_max: float = 1.30
    av_rescue_pad_max_sec: float = 5.0
    av_rescue_rewrite_max_drift_sec: float = 15.0

    # ── 可灵文生图 (Image 3.0) ──
    kling_image_access_key: str = ""
    kling_image_secret_key: str = ""
    kling_image_model: str = "kling-v1-5"

    # ── Optional visual backends ──
    visual_image_provider: str = "kling"  # "kling" | "atlascloud"
    visual_video_provider: str = "kling"  # "kling" | "atlascloud" | "muapi"
    atlascloud_api_key: str = ""
    atlascloud_media_base_url: str = "https://api.atlascloud.ai/api/v1"
    atlascloud_image_model: str = "bytedance/seedream-v5.0-lite"
    atlascloud_video_model: str = "bytedance/seedance-2.0-fast/image-to-video"
    atlascloud_video_resolution: str = "720p"
    atlascloud_video_generate_audio: bool = True
    muapi_api_key: str = ""
    muapi_media_base_url: str = "https://api.muapi.ai/api/v1"
    muapi_video_model: str = "seedance-2-image-to-video-fast"

    # ── 飞书 Feishu ──
    feishu_app_id: str = ""
    feishu_app_secret: str = ""

    # ── 火山引擎·豆包 TTS ──
    volcengine_tts_appid: str = ""
    volcengine_tts_access_token: str = ""
    volcengine_tts_secret_key: str = ""
    volcengine_tts_cluster: str = "volcano_tts"
    volcengine_tts_default_voice: str = "zh_female_wenrouxiaoya_uranus_bigtts"

    # ── PostgreSQL ──
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/myaivideos"

    # ── Redis ──
    redis_url: str = "redis://localhost:6379/0"

    # ── 热搜 ──
    trending_fetch_interval_min: int = 30
    trending_platforms: list[str] = Field(default=["douyin", "weibo", "bilibili"])

    # ── 路径 ──
    data_dir: Path = Field(default_factory=lambda: Path("data"))
    output_dir: Path = Field(default_factory=lambda: Path("output"))
    memory_path: Path = Field(default_factory=lambda: Path("cat_memory.json"))

    # ── 风格模板 ──
    style_templates_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent / "style_templates"
    )

    # ── 角色 IP ──
    characters_config_path: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent / "characters.yaml"
    )

    # ── 并发 ──
    max_concurrent_jobs: int = 1
    enable_burn_captions: bool = False
    # Phase P P3：是否走旧链路（先视频后 TTS）。默认 False = 走 P3 新链路（先 TTS 估时再选视频档）
    # 旧链路保留兜底：如果 P3 出 bug 影响生产，env 设 USE_LEGACY_AV_PIPELINE=true 可一键回退
    use_legacy_av_pipeline: bool = False

    # ── yt-dlp ──
    ytdlp_cookies_file: str = ""   # 可选：浏览器导出的 cookies.txt 路径，B站需要登录态

    # ── 内容安全 ──
    blocked_keywords: list[str] = Field(
        default=["政治", "选举", "政党", "领导人", "抗议", "赌博", "色情", "毒品", "诈骗"]
    )

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
