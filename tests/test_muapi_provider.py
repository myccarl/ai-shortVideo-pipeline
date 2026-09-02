from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from layers.L3_visual.providers.base import VideoResult


class FakeResponse:
    def __init__(self, data=None, *, content: bytes = b""):
        self._data = data if data is not None else {}
        self.content = content

    def json(self):
        return self._data

    def raise_for_status(self):
        return None


class FakeAsyncClient:
    def __init__(self, **kwargs):
        self.posts = []
        self.gets = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        if url.endswith("/upload_file"):
            return FakeResponse({"url": "https://files.example/start.png"})
        return FakeResponse({"request_id": "muapi-task", "status": "processing"})

    async def get(self, url, **kwargs):
        self.gets.append((url, kwargs))
        if "/predictions/" in url:
            return FakeResponse(
                {
                    "request_id": "muapi-task",
                    "status": "completed",
                    "output": {"video_url": "https://cdn.example/clip.mp4"},
                }
            )
        return FakeResponse(content=b"video-bytes")


def _settings(**overrides):
    values = {
        "muapi_api_key": "muapi-key",
        "muapi_media_base_url": "https://api.muapi.example/api/v1",
        "muapi_video_model": "seedance-2-image-to-video-fast",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.asyncio
async def test_muapi_image_to_video_uploads_start_frame_and_polls(monkeypatch, tmp_path):
    import layers.L3_visual.providers.muapi as module

    fake_client = FakeAsyncClient()
    monkeypatch.setattr(module, "get_settings", lambda: _settings())
    monkeypatch.setattr(module.httpx, "AsyncClient", lambda **kwargs: fake_client)

    async def _no_sleep():
        return None

    monkeypatch.setattr(module.asyncio, "sleep", lambda _delay: _no_sleep())

    image = tmp_path / "start.png"
    image.write_bytes(b"image-bytes")
    output = tmp_path / "clip.mp4"

    result = await module.image_to_video(
        str(image),
        "slow camera movement",
        str(output),
        duration_sec=3,
        aspect_ratio="9:16",
        quality="standard",
    )

    assert result == VideoResult(
        url="https://cdn.example/clip.mp4",
        local_path=str(output),
        duration_sec=4,
        task_id="muapi-task",
        model="seedance-2-image-to-video-fast",
    )
    assert output.read_bytes() == b"video-bytes"
    assert fake_client.posts[0][0] == "https://api.muapi.example/api/v1/upload_file"
    assert fake_client.posts[0][1]["headers"] == {"x-api-key": "muapi-key"}
    assert fake_client.posts[0][1]["files"]["file"][0] == "start.png"
    assert fake_client.posts[0][1]["files"]["file"][1] == b"image-bytes"
    assert fake_client.posts[1][0] == "https://api.muapi.example/api/v1/seedance-2-image-to-video-fast"
    assert fake_client.posts[1][1]["headers"] == {
        "x-api-key": "muapi-key",
        "Content-Type": "application/json",
    }
    assert fake_client.posts[1][1]["json"] == {
        "prompt": "slow camera movement",
        "images_list": ["https://files.example/start.png"],
        "aspect_ratio": "9:16",
        "duration": 4,
    }
    assert fake_client.gets[0][0] == "https://api.muapi.example/api/v1/predictions/muapi-task/result"


def test_muapi_pro_quality_uses_standard_model(monkeypatch):
    import layers.L3_visual.providers.muapi as module

    monkeypatch.setattr(module, "get_settings", lambda: _settings())

    assert module._model("pro") == "seedance-2-image-to-video"
    assert module._model("standard") == "seedance-2-image-to-video-fast"


@pytest.mark.asyncio
async def test_muapi_rejects_insecure_output_url(monkeypatch, tmp_path):
    import layers.L3_visual.providers.muapi as module

    class InsecureClient(FakeAsyncClient):
        async def get(self, url, **kwargs):
            self.gets.append((url, kwargs))
            return FakeResponse(
                {
                    "status": "completed",
                    "output": {"video_url": "http://cdn.example/clip.mp4"},
                }
            )

    fake_client = InsecureClient()
    monkeypatch.setattr(module, "get_settings", lambda: _settings())
    monkeypatch.setattr(module.httpx, "AsyncClient", lambda **kwargs: fake_client)

    image = tmp_path / "start.png"
    image.write_bytes(b"image-bytes")

    with pytest.raises(RuntimeError, match="insecure URL"):
        await module.image_to_video(str(image), "motion", str(tmp_path / "clip.mp4"))


@pytest.mark.asyncio
async def test_image_to_video_routes_muapi(monkeypatch, tmp_path):
    fake_llm_client = types.ModuleType("integrations.llm_client")
    fake_llm_client.call_glm4v = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "integrations.llm_client", fake_llm_client)

    import layers.L3_visual.image_to_video as module

    captured = {}

    async def fake_muapi_image_to_video(**kwargs):
        captured.update(kwargs)
        return VideoResult(url="https://cdn.example/clip.mp4", local_path=kwargs["output_path"], model="muapi")

    monkeypatch.setattr(module, "get_settings", lambda: SimpleNamespace(visual_video_provider="muapi"))
    monkeypatch.setattr(module, "_muapi_image_to_video", fake_muapi_image_to_video)

    result = await module.image_to_video(
        image_path="start.png",
        prompt="motion",
        output_path=str(tmp_path / "clip.mp4"),
        duration_sec=5,
        aspect_ratio="9:16",
    )

    assert result.model == "muapi"
    assert captured["image_path"] == "start.png"
    assert captured["prompt"] == "motion"
    assert captured["aspect_ratio"] == "9:16"
