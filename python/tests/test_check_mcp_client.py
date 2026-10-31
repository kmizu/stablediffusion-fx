import asyncio
import base64
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


def load_harness():
    path = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "check_mcp_client.py"
    fake_mcp = ModuleType("mcp")
    fake_mcp.Client = object
    fake_mcp.StdioServerParameters = object
    previous = sys.modules.get("mcp")
    sys.modules["mcp"] = fake_mcp
    try:
        spec = importlib.util.spec_from_file_location("check_mcp_client_under_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            sys.modules.pop("mcp", None)
        else:
            sys.modules["mcp"] = previous


class Model:
    def __init__(self, value):
        self.value = value

    def model_dump(self, mode="json"):
        return self.value


class Tool(Model):
    def __init__(self, name):
        super().__init__({"name": name})
        self.name = name


class Result:
    def __init__(self, data=None, content=None):
        self.data = data or {}
        self.content = content or []

    def model_dump(self, mode="json"):
        return {"structuredContent": self.data, "content": self.content, "isError": False}


class StopExercise(Exception):
    pass


class FakeClient:
    protocol_version = "2025-06-18"
    server_info = Model({"name": "fixture"})

    def __init__(self, image_path, image_content, require_progress):
        self.image_path = str(image_path)
        self.image_content = image_content
        self.require_progress = require_progress

    async def list_tools(self):
        names = ["txt2img", "img2img", "get_job", "cancel_job", "list_models", "list_samplers", "list_scripts"]
        return SimpleNamespace(tools=[Tool(name) for name in names])

    async def call_tool(self, name, arguments, read_timeout_seconds=None, progress_callback=None):
        if name == "txt2img":
            if self.require_progress:
                assert progress_callback is not None, "txt2img must use the official SDK progress callback"
            if progress_callback is not None:
                await progress_callback(50.0, 100.0, "base step 1/2")
                await progress_callback(100.0, 100.0, "done")
            return Result({"images": [{"path": self.image_path}]})
        if name == "get_image":
            return Result({"path": self.image_path}, self.image_content)
        if name == "img2img":
            raise StopExercise
        return Result()


def args():
    return SimpleNamespace(require_extensions=False, checkpoint="fixture", side=64, steps=1, dry_run=True)


def test_exercise_uses_sdk_progress_and_validates_decoded_image_block(tmp_path):
    harness = load_harness()
    image = tmp_path / "image.png"
    image.write_bytes(b"fixture")
    client = FakeClient(image, [{"type": "image", "mime_type": "image/png",
                                "data": base64.b64encode(b"png bytes").decode()}], require_progress=True)
    receipt = {}
    with pytest.raises(StopExercise):
        asyncio.run(harness.exercise(client, args(), receipt))
    assert [event["progress"] for event in receipt["progress_events"]] == [50.0, 100.0]
    assert receipt["image_blocks"] == [{"mime_type": "image/png", "decoded_bytes": 9}]


def test_exercise_rejects_missing_image_content_block(tmp_path):
    harness = load_harness()
    image = tmp_path / "image.png"
    image.write_bytes(b"fixture")
    client = FakeClient(image, [{"type": "text", "text": "no image"}], require_progress=False)
    with pytest.raises(AssertionError, match="image content block"):
        asyncio.run(harness.exercise(client, args(), {}))
