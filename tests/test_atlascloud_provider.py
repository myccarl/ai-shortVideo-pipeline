from __future__ import annotations

from types import SimpleNamespace

import pytest

from layers.L3_visual.providers.base import ImageResult, VideoResult


class FakeResponse:
    def __init__(self, data=None, *, content: bytes = b""):
        self._data = data if data is not None else {}
        self.content = content

    def json(self):
        return self._data

    def raise_for_status(self):
        return None


class FakeAsyncClient:
    def __init__(self, *, post_response=None, get_responses=None, **kwargs):
        self.post_response = post_response or FakeResponse()
        self.get_responses = list(get_responses or [])
        self.posts = []
        self.gets = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        return self.post_response

    async def get(self, url, **kwargs):
        self.gets.append((url, kwargs))
        if self.get_responses:
            return self.get_responses.pop(0)
        return FakeResponse({"status": "processing"})


def _settings(**overrides):
    values = {
        "atlascloud_api_key": "atlas-key",
        "atlascloud_media_base_url": "https://atlas.example/api/v1",
        "atlascloud_image_model": "bytedance/seedream-v5.0-lite",
        "atlascloud_video_model": "bytedance/seedance-2.0-fast/image-to-video",
        "atlascloud_video_resolution": "720p",
        "atlascloud_video_generate_audio": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    import layers.L3_visual.providers.atlascloud as module

    async def _sleep(_delay):
        return None

    monkeypatch.setattr(module.asyncio, "sleep", _sleep)


@pytest.mark.anyio
async def test_atlascloud_text_to_image_submits_schema_payload(monkeypatch, tmp_path):
    import layers.L3_visual.providers.atlascloud as module

    fake_client = FakeAsyncClient(
        post_response=FakeResponse({"id": "img-task"}),
        get_responses=[
            FakeResponse({"status": "completed", "outputs": ["https://cdn.example/image.png"]}),
            FakeResponse(content=b"png-bytes"),
        ],
    )
    monkeypatch.setattr(module, "get_settings", lambda: _settings())
    monkeypatch.setattr(module.httpx, "AsyncClient", lambda **kwargs: fake_client)

    out = tmp_path / "frame.png"
    result = await module.text_to_image("cinematic frame", str(out), aspect_ratio="9:16")

    assert result == ImageResult(
        url="https://cdn.example/image.png",
        local_path=str(out),
        model="bytedance/seedream-v5.0-lite",
    )
    assert out.read_bytes() == b"png-bytes"
    assert fake_client.posts[0][0] == "https://atlas.example/api/v1/model/generateImage"
    payload = fake_client.posts[0][1]["json"]
    assert payload == {
        "model": "bytedance/seedream-v5.0-lite",
        "prompt": "cinematic frame",
        "size": "1600*2848",
        "output_format": "png",
    }
    assert fake_client.posts[0][1]["headers"]["Authorization"] == "Bearer atlas-key"


@pytest.mark.anyio
async def test_atlascloud_image_to_video_submits_i2v_payload(monkeypatch, tmp_path):
    import layers.L3_visual.providers.atlascloud as module

    fake_client = FakeAsyncClient(
        post_response=FakeResponse({"data": {"task_id": "video-task"}}),
        get_responses=[
            FakeResponse({"data": {"status": "completed", "outputs": ["https://cdn.example/video.mp4"]}}),
            FakeResponse(content=b"mp4-bytes"),
        ],
    )
    monkeypatch.setattr(module, "get_settings", lambda: _settings())
    monkeypatch.setattr(module.httpx, "AsyncClient", lambda **kwargs: fake_client)

    image = tmp_path / "first.png"
    image.write_bytes(b"image")
    out = tmp_path / "clip.mp4"
    result = await module.image_to_video(
        str(image),
        "slow dolly in",
        str(out),
        duration_sec=3,
        aspect_ratio="9:16",
        quality="pro",
    )

    assert result == VideoResult(
        url="https://cdn.example/video.mp4",
        local_path=str(out),
        duration_sec=4,
        task_id="video-task",
        model="bytedance/seedance-2.0-fast/image-to-video",
    )
    assert out.read_bytes() == b"mp4-bytes"
    assert fake_client.posts[0][0] == "https://atlas.example/api/v1/model/generateVideo"
    payload = fake_client.posts[0][1]["json"]
    assert payload["model"] == "bytedance/seedance-2.0-fast/image-to-video"
    assert payload["image"] == "aW1hZ2U="
    assert payload["prompt"] == "slow dolly in"
    assert payload["duration"] == 4
    assert payload["resolution"] == "720p"
    assert payload["ratio"] == "9:16"
    assert payload["bitrate_mode"] == "high"
    assert payload["generate_audio"] is True
    assert payload["watermark"] is False


@pytest.mark.anyio
async def test_generate_image_routes_to_atlascloud(monkeypatch, tmp_path):
    import layers.L3_visual.text_to_image as module

    captured = {}

    async def fake_atlascloud_text_to_image(**kwargs):
        captured.update(kwargs)
        return ImageResult(url="https://cdn.example/image.png", local_path=kwargs["output_path"], model="atlas")

    monkeypatch.setattr(module, "get_settings", lambda: SimpleNamespace(
        visual_image_provider="atlascloud",
        atlascloud_image_model="bytedance/seedream-v5.0-lite",
        kling_image_model="kling-v1-5",
        clip_consistency_enabled=False,
    ))
    monkeypatch.setattr(module, "atlascloud_text_to_image", fake_atlascloud_text_to_image)

    out = tmp_path / "frame.png"
    result = await module.generate_image("prompt", str(out), aspect_ratio="16:9")

    assert result.model == "atlas"
    assert captured["prompt"] == "prompt"
    assert captured["output_path"] == str(out)
    assert captured["aspect_ratio"] == "16:9"
    assert captured["model"] == "bytedance/seedream-v5.0-lite"


@pytest.mark.anyio
async def test_image_to_video_routes_to_atlascloud(monkeypatch, tmp_path):
    import layers.L3_visual.image_to_video as module

    captured = {}

    async def fake_atlascloud_image_to_video(**kwargs):
        captured.update(kwargs)
        return VideoResult(url="https://cdn.example/video.mp4", local_path=kwargs["output_path"], model="atlas")

    monkeypatch.setattr(module, "get_settings", lambda: SimpleNamespace(visual_video_provider="atlascloud"))
    monkeypatch.setattr(module, "_atlascloud_image_to_video", fake_atlascloud_image_to_video)

    result = await module.image_to_video(
        image_path="first.png",
        prompt="motion",
        output_path=str(tmp_path / "clip.mp4"),
        duration_sec=5,
        aspect_ratio="9:16",
    )

    assert result.model == "atlas"
    assert captured["image_path"] == "first.png"
    assert captured["prompt"] == "motion"
    assert captured["aspect_ratio"] == "9:16"
