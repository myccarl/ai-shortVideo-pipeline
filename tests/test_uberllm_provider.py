from __future__ import annotations

from types import SimpleNamespace

import pytest

from layers.L3_visual.providers.base import ImageResult, VideoResult

BASE = "https://uberllm.example/v1"

IMAGE_OPTIONS = {
    "parameters": {
        "properties": {
            "model": {"type": "string"},
            "prompt": {"type": "string"},
            "aspect_ratio": {"enum": ["1:1", "16:9", "9:16"], "type": "string"},
        }
    }
}

FAST_VIDEO_OPTIONS = {
    "parameters": {
        "properties": {
            "audio": {"type": "boolean"},
            "duration_seconds": {"enum": [5, 10, 15], "type": "number"},
            "image_url": {"type": "string"},
            "resolution": {"enum": ["480p", "720p"], "type": "string"},
        }
    }
}

PRO_VIDEO_OPTIONS = {
    "parameters": {
        "properties": {
            "aspect_ratio": {"enum": ["16:9", "1:1", "9:16"], "type": "string"},
            "duration_seconds": {"enum": [5, 10, 15], "type": "number"},
            "image_url": {"type": "string"},
            "resolution": {"enum": ["480p", "720p", "1080p"], "type": "string"},
        }
    }
}


class FakeResponse:
    def __init__(self, data=None, *, content: bytes = b""):
        self._data = data if data is not None else {}
        self.content = content

    def json(self):
        return self._data

    def raise_for_status(self):
        return None


class FakeAsyncClient:
    def __init__(self, *, routes, **kwargs):
        self.routes = {key: list(value) for key, value in routes.items()}
        self.posts = []
        self.gets = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def _next(self, method, url):
        return self.routes[(method, url)].pop(0)

    async def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        return self._next("POST", url)

    async def get(self, url, **kwargs):
        self.gets.append((url, kwargs))
        return self._next("GET", url)


def _settings(**overrides):
    values = {
        "uberllm_api_key": "ull-key",
        "uberllm_base_url": BASE,
        "uberllm_image_model": "venice/seedream-v5-lite",
        "uberllm_video_model": "bytedance/seedance-2-0-fast-image-to-video",
        "uberllm_video_resolution": "720p",
        "uberllm_video_audio": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    import layers.L3_visual.providers.uberllm as module

    async def _sleep(_delay):
        return None

    monkeypatch.setattr(module.asyncio, "sleep", _sleep)


@pytest.mark.anyio
async def test_uberllm_text_to_image_fits_payload_to_model_options(monkeypatch, tmp_path):
    import layers.L3_visual.providers.uberllm as module

    artifact = f"{BASE}/files/file_img1/content"
    fake_client = FakeAsyncClient(routes={
        ("GET", f"{BASE}/models/venice/seedream-v5-lite"): [FakeResponse(IMAGE_OPTIONS)],
        ("POST", f"{BASE}/images/generations"): [FakeResponse({"data": [{"url": artifact}]})],
        ("GET", artifact): [FakeResponse(content=b"png-bytes")],
    })
    monkeypatch.setattr(module, "get_settings", lambda: _settings())
    monkeypatch.setattr(module.httpx, "AsyncClient", lambda **kwargs: fake_client)

    out = tmp_path / "frame.png"
    result = await module.text_to_image(
        "cinematic frame",
        str(out),
        negative_prompt="text, watermark",
        aspect_ratio="9:16",
        positive_suffix="photoreal",
    )

    # Callers get the downloaded copy, never the key-gated, 24h link.
    assert result == ImageResult(url=str(out), local_path=str(out), model="venice/seedream-v5-lite")
    assert artifact not in result.url
    assert out.read_bytes() == b"png-bytes"
    url, kwargs = fake_client.posts[0]
    assert url == f"{BASE}/images/generations"
    assert kwargs["json"] == {
        "model": "venice/seedream-v5-lite",
        "prompt": "cinematic frame, photoreal",
        "n": 1,
        "aspect_ratio": "9:16",
    }
    assert kwargs["headers"]["Authorization"] == "Bearer ull-key"
    # The artifact link is under the API, so it is downloaded with the key.
    assert fake_client.gets[-1][1]["headers"]["Authorization"] == "Bearer ull-key"


def test_uberllm_image_payload_maps_ratio_to_pixel_size():
    import layers.L3_visual.providers.uberllm as module

    options = {"size": {"enum": ["1024x1024", "1536x1024", "1024x1536"], "type": "string"}}
    payload = module._image_payload(
        options, model_id="venice/z-image-turbo", prompt="frame", negative_prompt="", aspect_ratio="9:16"
    )
    assert payload == {"model": "venice/z-image-turbo", "prompt": "frame", "n": 1, "size": "1024x1536"}


@pytest.mark.anyio
async def test_uberllm_image_to_video_uploads_frame_and_snaps_duration(monkeypatch, tmp_path):
    import layers.L3_visual.providers.uberllm as module

    model = "bytedance/seedance-2-0-fast-image-to-video"
    artifact = f"{BASE}/files/file_vid1/content"
    fake_client = FakeAsyncClient(routes={
        ("GET", f"{BASE}/models/{model}"): [FakeResponse(FAST_VIDEO_OPTIONS)],
        ("POST", f"{BASE}/files"): [FakeResponse({"id": "file_abc", "object": "file", "url": "u"})],
        ("POST", f"{BASE}/video/generations"): [FakeResponse({"id": "job-1", "status": "queued"})],
        ("GET", f"{BASE}/video/generations/job-1"): [
            FakeResponse({"id": "job-1", "status": "running"}),
            FakeResponse({
                "id": "job-1",
                "status": "succeeded",
                "model": model,
                "results": [{"url": artifact, "content_type": "video/mp4", "duration_seconds": 5}],
            }),
        ],
        ("GET", artifact): [FakeResponse(content=b"mp4-bytes")],
    })
    monkeypatch.setattr(module, "get_settings", lambda: _settings())
    monkeypatch.setattr(module.httpx, "AsyncClient", lambda **kwargs: fake_client)

    image = tmp_path / "first.png"
    image.write_bytes(b"image")
    out = tmp_path / "clip.mp4"
    result = await module.image_to_video(str(image), "slow dolly in", str(out), duration_sec=3, aspect_ratio="9:16")

    assert result == VideoResult(url=str(out), local_path=str(out), duration_sec=5, task_id="job-1", model=model)
    # The video link is on the API, so it is fetched server-side with the key.
    assert fake_client.gets[-1][0] == artifact
    assert fake_client.gets[-1][1]["headers"]["Authorization"] == "Bearer ull-key"
    assert out.read_bytes() == b"mp4-bytes"
    upload_url, upload_kwargs = fake_client.posts[0]
    assert upload_url == f"{BASE}/files"
    assert upload_kwargs["files"]["file"][0] == "first.png"
    video_url, video_kwargs = fake_client.posts[1]
    assert video_url == f"{BASE}/video/generations"
    assert video_kwargs["headers"]["Idempotency-Key"]
    # This model takes no aspect_ratio (the start frame sets the shape) and only 5/10/15 s clips.
    assert video_kwargs["json"] == {
        "model": model,
        "prompt": "slow dolly in",
        "image_url": "file_abc",
        "duration_seconds": 5,
        "resolution": "720p",
        "audio": False,
    }


def test_uberllm_video_payload_pro_model_keeps_shape_and_drops_unknown_values(monkeypatch):
    import layers.L3_visual.providers.uberllm as module

    monkeypatch.setattr(module, "get_settings", lambda: _settings())
    assert module._video_model("pro") == module.PRO_VIDEO_MODEL
    assert module._video_model("standard") == module.DEFAULT_VIDEO_MODEL

    options = PRO_VIDEO_OPTIONS["parameters"]["properties"]
    payload = module._video_payload(
        options,
        model_id=module.PRO_VIDEO_MODEL,
        prompt="motion",
        image_ref="file_abc",
        duration_sec=12,
        aspect_ratio="4:3",
        resolution="1080p",
        audio=True,
    )
    assert payload == {
        "model": module.PRO_VIDEO_MODEL,
        "prompt": "motion",
        "image_url": "file_abc",
        "duration_seconds": 15,
        "resolution": "1080p",
    }
    payload = module._video_payload(
        options,
        model_id=module.PRO_VIDEO_MODEL,
        prompt="motion",
        image_ref="file_abc",
        duration_sec=30,
        aspect_ratio="9:16",
        resolution="720p",
        audio=False,
    )
    assert payload["duration_seconds"] == 15
    assert payload["aspect_ratio"] == "9:16"


@pytest.mark.anyio
async def test_uberllm_video_job_failure_raises(monkeypatch, tmp_path):
    import layers.L3_visual.providers.uberllm as module

    model = "bytedance/seedance-2-0-fast-image-to-video"
    fake_client = FakeAsyncClient(routes={
        ("GET", f"{BASE}/models/{model}"): [FakeResponse(FAST_VIDEO_OPTIONS)],
        ("POST", f"{BASE}/files"): [FakeResponse({"id": "file_abc"})],
        ("POST", f"{BASE}/video/generations"): [FakeResponse({"id": "job-2", "status": "queued"})],
        ("GET", f"{BASE}/video/generations/job-2"): [
            FakeResponse({"id": "job-2", "status": "failed", "error": {"message": "content rejected"}}),
        ],
    })
    monkeypatch.setattr(module, "get_settings", lambda: _settings())
    monkeypatch.setattr(module.httpx, "AsyncClient", lambda **kwargs: fake_client)

    image = tmp_path / "first.png"
    image.write_bytes(b"image")
    with pytest.raises(RuntimeError, match="content rejected"):
        await module.image_to_video(str(image), "motion", str(tmp_path / "clip.mp4"))


@pytest.mark.anyio
async def test_generate_image_routes_to_uberllm(monkeypatch, tmp_path):
    import layers.L3_visual.text_to_image as module

    captured = {}

    async def fake_uberllm_text_to_image(**kwargs):
        captured.update(kwargs)
        return ImageResult(url="https://cdn.example/image.png", local_path=kwargs["output_path"], model="ull")

    monkeypatch.setattr(module, "get_settings", lambda: SimpleNamespace(
        visual_image_provider="uberllm",
        uberllm_image_model="venice/seedream-v5-lite",
        kling_image_model="kling-v1-5",
        clip_consistency_enabled=False,
    ))
    monkeypatch.setattr(module, "uberllm_text_to_image", fake_uberllm_text_to_image)

    out = tmp_path / "frame.png"
    result = await module.generate_image("prompt", str(out), aspect_ratio="16:9")

    assert result.model == "ull"
    assert captured["prompt"] == "prompt"
    assert captured["aspect_ratio"] == "16:9"
    assert captured["model"] == "venice/seedream-v5-lite"


@pytest.mark.anyio
async def test_image_to_video_routes_to_uberllm(monkeypatch, tmp_path):
    import layers.L3_visual.image_to_video as module

    captured = {}

    async def fake_uberllm_image_to_video(**kwargs):
        captured.update(kwargs)
        return VideoResult(url="https://cdn.example/video.mp4", local_path=kwargs["output_path"], model="ull")

    monkeypatch.setattr(module, "get_settings", lambda: SimpleNamespace(visual_video_provider="uberllm"))
    monkeypatch.setattr(module, "_uberllm_image_to_video", fake_uberllm_image_to_video)

    result = await module.image_to_video(
        image_path="first.png",
        prompt="motion",
        output_path=str(tmp_path / "clip.mp4"),
        duration_sec=5,
        aspect_ratio="9:16",
    )

    assert result.model == "ull"
    assert captured["image_path"] == "first.png"
    assert captured["aspect_ratio"] == "9:16"


@pytest.mark.parametrize(
    "base_url, artifact, expected",
    [
        ("https://api.uberllm.dev/v1", "https://api.uberllm.dev/v1/files/file_1/content", True),
        ("https://api.uberllm.dev/v1", "https://API.UBERLLM.DEV:443/v1/files/file_1/content", True),
        ("https://api.uberllm.dev/v1", "https://api.uberllm.dev/v1", True),
        ("https://api.uberllm.dev", "https://api.uberllm.dev/v1/files/file_1/content", True),
        # Lookalike hosts and userinfo tricks: a string prefix would match these.
        ("https://api.uberllm.dev", "https://api.uberllm.dev.evil.example/steal", False),
        ("https://api.uberllm.dev", "https://api.uberllm.dev@evil.example/steal", False),
        ("https://api.uberllm.dev/v1", "https://api.uberllm.dev/v1evil/steal", False),
        ("https://api.uberllm.dev/v1", "https://api.uberllm.dev/v1.evil.example/x", False),
        ("https://api.uberllm.dev/v1", "https://api.uberllm.dev/other/x", False),
        # Same host, different scheme or port is another origin.
        ("https://api.uberllm.dev/v1", "http://api.uberllm.dev/v1/files/file_1/content", False),
        ("https://api.uberllm.dev/v1", "https://api.uberllm.dev:8443/v1/files/file_1/content", False),
        ("https://api.uberllm.dev/v1", "not a url", False),
    ],
)
def test_uberllm_api_key_only_sent_to_api_origin(monkeypatch, base_url, artifact, expected):
    import layers.L3_visual.providers.uberllm as module

    monkeypatch.setattr(module, "get_settings", lambda: _settings(uberllm_base_url=base_url))
    assert module._is_api_url(artifact) is expected


@pytest.mark.anyio
async def test_uberllm_download_from_lookalike_host_sends_no_key(monkeypatch, tmp_path):
    import layers.L3_visual.providers.uberllm as module

    base = "https://api.uberllm.dev"
    artifact = "https://api.uberllm.dev.evil.example/file.png"
    fake_client = FakeAsyncClient(routes={
        ("GET", f"{base}/models/venice/seedream-v5-lite"): [FakeResponse(IMAGE_OPTIONS)],
        ("POST", f"{base}/images/generations"): [FakeResponse({"data": [{"url": artifact}]})],
        ("GET", artifact): [FakeResponse(content=b"png-bytes")],
    })
    monkeypatch.setattr(module, "get_settings", lambda: _settings(uberllm_base_url=base))
    monkeypatch.setattr(module.httpx, "AsyncClient", lambda **kwargs: fake_client)

    out = tmp_path / "frame.png"
    result = await module.text_to_image("frame", str(out))

    assert fake_client.gets[-1][0] == artifact
    assert fake_client.gets[-1][1]["headers"] is None
    assert result.url == str(out)
