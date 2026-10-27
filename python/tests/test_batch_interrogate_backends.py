"""Internal backend dispatcher tests; no real model inference or network."""
import pytest
from PIL import Image
from sdfx.batch_interrogate_backends import BackendRegistry, BackendStatus


class Context:
    def __init__(self): self.cancelled = False
    def check_interrupt(self):
        if self.cancelled: raise RuntimeError("cancelled")


def test_default_inventory_is_unavailable_without_loading_or_network():
    registry = BackendRegistry()
    statuses = registry.available_backends()
    assert {s.name for s in statuses} == {"clip", "deepbooru", "clip_ext", "wd_ext"}
    assert all(not s.available and s.reason for s in statuses)
    with pytest.raises(ValueError, match="unavailable"):
        registry.interrogate("unused.png", ("clip",), Context())


def test_injected_backends_receive_rgb_and_preserve_requested_order(tmp_path):
    image = tmp_path / "image.png"
    Image.new("RGBA", (2, 3), (4, 5, 6, 7)).save(image)
    calls = []
    class Backend:
        def __init__(self, name): self.name = name
        def interrogate(self, image, ctx):
            calls.append((self.name, image.mode, image.size, image.getpixel((0, 0))))
            return self.name
        def unload(self): calls.append((self.name, "unload"))
    registry = BackendRegistry({"clip": Backend("clip"), "deepbooru": Backend("deepbooru")})
    assert registry.interrogate(str(image), ("deepbooru", "clip"), Context()) == ["deepbooru", "clip"]
    assert calls == [("deepbooru", "RGB", (2, 3), (4, 5, 6)), ("clip", "RGB", (2, 3), (4, 5, 6))]
    registry.unload(("clip", "deepbooru"))
    assert calls[-2:] == [("clip", "unload"), ("deepbooru", "unload")]


def test_invalid_selection_and_unbounded_results_are_rejected_before_next_model(tmp_path):
    image = tmp_path / "image.png"; Image.new("RGB", (1, 1)).save(image)
    calls = []
    class Backend:
        def interrogate(self, image, ctx): calls.append(1); return "x" * 131_073
        def unload(self): pass
    registry = BackendRegistry({"clip": Backend()})
    with pytest.raises(ValueError): registry.interrogate(str(image), ("clip", "missing"), Context())
    assert calls == []
    with pytest.raises(ValueError, match="(?i)(result|limit)"):
        registry.interrogate(str(image), ("clip",), Context())
    assert calls == [1]


def test_cancellation_between_models_and_unload_failure_are_explicit(tmp_path):
    image = tmp_path / "image.png"; Image.new("RGB", (1, 1)).save(image)
    calls = []
    class Backend:
        def interrogate(self, image, ctx): calls.append("run"); ctx.cancelled = True; return "one"
        def unload(self): raise RuntimeError("unload failed")
    registry = BackendRegistry({"clip": Backend(), "deepbooru": Backend()})
    with pytest.raises(RuntimeError, match="cancelled"):
        registry.interrogate(str(image), ("clip", "deepbooru"), Context())
    assert calls == ["run"]
    with pytest.raises(RuntimeError, match="unload failed"):
        registry.unload(("clip",))


def test_owned_backend_scope_unloads_on_success_error_and_cancel():
    from sdfx.protocol import EngineError
    class Backend:
        calls = 0
        def interrogate(self, *args): return "result"
        def unload(self): self.calls += 1
    backend = Backend(); registry = BackendRegistry({"clip": backend})
    with registry.owned(("clip",)):
        pass
    for error in (ValueError("invalid"), EngineError("interrupted", "cancelled")):
        with pytest.raises(type(error)):
            with registry.owned(("clip",)):
                raise error
    assert backend.calls == 3


def test_owned_backend_scope_preserves_primary_error_and_unload_failure():
    from sdfx.protocol import EngineError
    class Backend:
        def interrogate(self, *args): return "result"
        def unload(self): raise RuntimeError("release failed")
    registry = BackendRegistry({"clip": Backend()})
    with pytest.raises(EngineError) as error:
        with registry.owned(("clip",)):
            raise EngineError("interrupted", "cancelled")
    assert error.value.code == "interrupted" and "cancelled" in error.value.message
    assert "release failed" in error.value.message


def test_owned_scope_freezes_selection_and_releases_shared_provider_once():
    class Backend:
        def __init__(self): self.calls = 0
        def interrogate(self, *args): return "result"
        def unload(self): self.calls += 1
    shared, other = Backend(), Backend()
    registry = BackendRegistry({"clip": shared, "deepbooru": shared, "wd_ext": other})
    selection = ["clip", "deepbooru"]
    with registry.owned(selection):
        selection[:] = ["wd_ext"]
    assert shared.calls == 1 and other.calls == 0


@pytest.mark.parametrize("failure", [ValueError("invalid input"),
                                    type("OutOfMemoryError", (Exception,), {})("gpu full")])
def test_unload_failure_keeps_original_exception_type_and_reason(failure):
    class Backend:
        def interrogate(self, *args): return "result"
        def unload(self): raise RuntimeError("release failed")
    registry = BackendRegistry({"clip": Backend()})
    with pytest.raises(type(failure)) as caught:
        with registry.owned(("clip",)):
            raise failure
    assert str(failure).split(";")[0] in str(caught.value)
    assert "release failed" in str(caught.value)


def test_exif_orientation_matches_img2img_input(tmp_path):
    from sdfx.images import load_rgb
    image = tmp_path / "oriented.png"
    pixels = Image.new("RGB", (3, 2))
    pixels.putdata([(255, 0, 0), (0, 255, 0), (0, 0, 255),
                    (255, 255, 0), (0, 255, 255), (255, 0, 255)])
    exif = Image.Exif()
    exif[274] = 6
    pixels.save(image, exif=exif)
    pixels.close()
    expected = load_rgb(str(image))
    try:
        assert expected.size == (2, 3)
        expected_pixels = list(expected.getdata())
    finally:
        expected.close()

    class Backend:
        def interrogate(self, rgb, ctx):
            assert rgb.size == (2, 3)
            assert list(rgb.getdata()) == expected_pixels
            return "caption"
        def unload(self): pass

    registry = BackendRegistry({"clip": Backend()})
    assert registry.interrogate(str(image), ("clip",), Context()) == ["caption"]
