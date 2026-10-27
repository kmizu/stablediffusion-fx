"""Behavioral tests for reference-guided temporal frame processing."""
import base64
import importlib.util
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from sdfx.protocol import EngineError, INTERRUPTED, INVALID_PARAMS, UNSUPPORTED
from sdfx.temporal_lab import run


needs_opencv = pytest.mark.skipif(importlib.util.find_spec("cv2") is None, reason="OpenCV is required for this operation")


class Context:
    def __init__(self, interrupt_after=None):
        self.calls = 0
        self.interrupt_after = interrupt_after
        self.events = []

    def check_interrupt(self):
        self.calls += 1
        if self.interrupt_after is not None and self.calls >= self.interrupt_after:
            raise EngineError(INTERRUPTED, "interrupted")

    def progress(self, step, total, pass_="base", **kwargs):
        self.events.append((step, total, pass_))


def images(directory, frames):
    directory.mkdir()
    for index, frame in enumerate(frames, 1):
        Image.fromarray(np.asarray(frame, dtype=np.uint8), "RGB").save(directory / f"frame{index:04d}.png")
    return directory


def solid(value, size=32):
    return np.full((size, size, 3), value, dtype=np.uint8)


def open_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def output_images(result):
    return sorted(Path(result["output_dir"]).glob("*.png"))


def make_params(tmp_path, op, **fields):
    output = tmp_path / "outputs"
    output.mkdir(exist_ok=True)
    return {"operation": op, "output_parent": str(output), "output_name": op, **fields}


def reference_and_generated(tmp_path):
    ref1 = solid(50)
    ref2 = ref1.copy()
    ref2[8:24, 8:24] = 100
    ref3 = ref2.copy()
    refs = images(tmp_path / "references", [ref1, ref2, ref3])
    gens = images(tmp_path / "generated", [solid(40), solid(200), solid(100)])
    return refs, gens


def decode_mask(result):
    with Image.open(BytesIO(base64.b64decode(result["mask_png_base64"], validate=True))) as image:
        return np.asarray(image.convert("L"))


@needs_opencv
def test_dfi_respects_motion_and_preserves_frame_count_and_input(tmp_path):
    refs, gens = reference_and_generated(tmp_path)
    before = [(p, p.read_bytes()) for p in sorted(gens.iterdir())]
    params = make_params(tmp_path, "dfi", reference_dir=str(refs), generated_dir=str(gens),
                         inter_denoise=1, inter_denoise_size=1, inter_blur=1)
    ctx = Context()
    result = run(params, ctx)
    files = output_images(result)
    assert result["operation"] == "dfi" and result["frame_count"] == len(files) == 3
    assert tuple(result["dimensions"]) == (32, 32)
    first, second = map(open_rgb, files[:2])
    assert np.array_equal(first, solid(40))
    # Reference stayed still at the corner; moving center follows new generation more.
    assert abs(int(second[0, 0, 0]) - 40) < abs(int(second[16, 16, 0]) - 40)
    assert all(p.read_bytes() == data for p, data in before)
    assert ctx.events and ctx.events[-1][0] == ctx.events[-1][1] == 3


@needs_opencv
def test_preview_is_a_stable_region_mask_and_writes_no_output(tmp_path):
    refs, gens = reference_and_generated(tmp_path)
    params = {"operation": "preview", "reference_dir": str(refs), "generated_dir": str(gens),
              "frame_index": 1, "tolerance": 5, "expand": 0, "smooth": 1}
    result = run(params, Context())
    mask = decode_mask(result)
    assert mask.shape == (32, 32)
    assert mask[0, 0] > mask[16, 16]
    assert Path(result["generated_previous"]).name == "frame0001.png"
    assert Path(result["generated_current"]).name == "frame0002.png"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["generated", "references"]


@needs_opencv
def test_preview_tolerance_expand_and_smoothing_change_the_mask(tmp_path):
    ref1 = solid(50)
    ref2 = ref1.copy()
    ref2[8:24, 8:24] = 60
    refs = images(tmp_path / "references", [ref1, ref2])
    gens = images(tmp_path / "generated", [solid(40), solid(200)])
    base = {"operation": "preview", "reference_dir": str(refs), "generated_dir": str(gens),
            "frame_index": 1, "tolerance": 5, "expand": 0, "smooth": 1}
    initial = decode_mask(run(base, Context()))
    tolerant = decode_mask(run({**base, "tolerance": 15}, Context()))
    expanded = decode_mask(run({**base, "expand": 5}, Context()))
    smoothed = decode_mask(run({**base, "smooth": 7}, Context()))
    assert tolerant[16, 16] > initial[16, 16]
    assert np.count_nonzero(expanded < 128) > np.count_nonzero(initial < 128)
    assert np.any((smoothed > 0) & (smoothed < 255))


@needs_opencv
def test_source_denoise_changes_stability_map_for_noisy_reference(tmp_path):
    ref1 = solid(80)
    ref2 = ref1.copy()
    ref2[::2, ::2] = 88
    refs = images(tmp_path / "references", [ref1, ref2])
    gens = images(tmp_path / "generated", [solid(40), solid(200)])
    base = {"operation": "preview", "reference_dir": str(refs), "generated_dir": str(gens),
            "frame_index": 1, "tolerance": 5, "expand": 0, "smooth": 1}
    raw = decode_mask(run({**base, "source_denoise": 0}, Context()))
    filtered = decode_mask(run({**base, "source_denoise": 9}, Context()))
    assert not np.array_equal(raw, filtered)


@needs_opencv
def test_intermediate_denoise_and_blur_controls_act_on_output(tmp_path):
    refs = images(tmp_path / "references", [solid(0), solid(100), solid(200)])
    noisy = solid(120)
    noisy[::2, ::2] = 145
    noisy[1::2, 1::2] = 95
    gens = images(tmp_path / "generated", [noisy, noisy, noisy])
    base = make_params(tmp_path, "dfi", reference_dir=str(refs), generated_dir=str(gens),
                       inter_denoise=1, inter_denoise_size=1, inter_denoise_frequency=1, inter_blur=1)
    plain = output_images(run({**base, "output_name": "plain"}, Context()))
    strong = output_images(run({**base, "output_name": "strong", "inter_denoise": 15,
                                "inter_denoise_size": 15, "inter_blur": 5}, Context()))
    assert not np.array_equal(open_rgb(plain[1]), open_rgb(strong[1]))
    less_frequent = output_images(run({**base, "output_name": "less-frequent", "inter_denoise": 15,
                                       "inter_denoise_size": 15, "inter_blur": 5,
                                       "inter_denoise_frequency": 3}, Context()))
    assert any(not np.array_equal(open_rgb(a), open_rgb(b)) for a, b in zip(strong[1:], less_frequent[1:]))


@needs_opencv
def test_dfi_frame_limit_and_refresh_change_output(tmp_path):
    refs = images(tmp_path / "references", [solid(50)] * 3)
    gens = images(tmp_path / "generated", [solid(20), solid(160), solid(240)])
    base = make_params(tmp_path, "dfi", reference_dir=str(refs), generated_dir=str(gens),
                       inter_denoise=1, inter_denoise_size=1, inter_blur=1)
    stable = run({**base, "output_name": "stable", "frame_limit": 2}, Context())
    refresh = run({**base, "output_name": "refresh", "refresh_frequency": 1, "preserve": 0}, Context())
    assert len(output_images(stable)) == 2
    assert len(output_images(refresh)) == 3
    assert open_rgb(output_images(refresh)[1]).mean() > open_rgb(output_images(stable)[1]).mean()


def test_dfi_upstream_white_count_dynamic_blur_and_dissolve_fixture(tmp_path):
    """Derived from upstream 247-255 and 281-299, 339-367 plus IM dissolve docs.

    Static pixels become 255 before smoothing; only exact 255 values count
    toward the automatically selected opacity. Dynamic blur is applied after
    that count, so the arithmetic mean of gray pixels is not the percentage.
    """
    cv = pytest.importorskip("cv2")
    ref1 = solid(50)
    ref2 = ref1.copy()
    ref2[:, 16:] = 120
    refs = images(tmp_path / "references", [ref1, ref2])
    gens = images(tmp_path / "generated", [solid(0), solid(200)])
    params = make_params(tmp_path, "dfi", reference_dir=str(refs), generated_dir=str(gens),
                         tolerance=5, smooth=3, inter_denoise=1, inter_denoise_size=1,
                         inter_blur=1, refresh_frequency=0)
    result = run(params, Context())
    observed = open_rgb(output_images(result)[1])[:, :, 0].astype(int)
    binary = np.zeros((32, 32), np.uint8)
    binary[:, :16] = 255
    smoothed = cv.GaussianBlur(binary, (3, 3), 0)
    percent = round(float(np.count_nonzero(smoothed == 255)) / smoothed.size * 100, 1)
    kernel = int(np.ceil((100 - percent) / 3))
    if kernel % 2 == 0:
        kernel += 1
    dynamic = cv.GaussianBlur(smoothed, (kernel, kernel), 0).astype(float) / 255
    # Over composite: previous black with opacity dynamic * percent over new gray.
    expected = np.rint(200 * (1 - dynamic * percent / 100)).astype(int)
    assert abs((smoothed / 255).mean() - percent / 100) > 0.01
    assert np.abs(observed - expected).max() <= 1


def test_refresh_percent_multiplies_dynamic_mask_opacity(tmp_path):
    cv = pytest.importorskip("cv2")
    ref1 = solid(50)
    ref2 = ref1.copy()
    ref2[:, 16:] = 120
    refs = images(tmp_path / "references", [ref1, ref2])
    gens = images(tmp_path / "generated", [solid(0), solid(200)])
    params = make_params(tmp_path, "dfi", reference_dir=str(refs), generated_dir=str(gens),
                         refresh_frequency=1, preserve=25, inter_denoise=1,
                         inter_denoise_size=1, inter_blur=1)
    observed = open_rgb(output_images(run(params, Context()))[1])[:, :, 0].astype(int)
    mask = np.zeros((32, 32), np.uint8)
    mask[:, :16] = 255
    dynamic = cv.GaussianBlur(mask, (17, 17), 0).astype(float) / 255
    expected = np.rint(200 * (1 - dynamic * 0.25)).astype(int)
    assert np.abs(observed - expected).max() <= 1


def test_pair_count_and_dimension_errors_publish_nothing(tmp_path):
    refs = images(tmp_path / "references", [solid(10), solid(20)])
    gens = images(tmp_path / "generated", [solid(30)])
    params = make_params(tmp_path, "dfi", reference_dir=str(refs), generated_dir=str(gens))
    with pytest.raises(EngineError) as exc:
        run(params, Context())
    assert exc.value.code == INVALID_PARAMS
    assert not (tmp_path / "outputs" / "dfi").exists()
    Image.fromarray(solid(20, 16)).save(gens / "frame0002.png")
    with pytest.raises(EngineError):
        run(params, Context())
    assert not (tmp_path / "outputs" / "dfi").exists()


@needs_opencv
def test_reference_is_center_cropped_and_resized_to_generated_frame(tmp_path):
    # Upstream center-crops the reference aspect ratio, then resizes it to the
    # generated frame; motion solely in cropped-out margins is invisible.
    ref1 = np.full((32, 64, 3), 50, dtype=np.uint8)
    ref2 = ref1.copy()
    ref2[:, :16] = 220
    ref2[:, 48:] = 220
    refs = images(tmp_path / "references", [ref1, ref2])
    gens = images(tmp_path / "generated", [solid(20), solid(200)])
    before = {p: p.read_bytes() for p in refs.iterdir()}
    preview = run({"operation": "preview", "reference_dir": str(refs),
                   "generated_dir": str(gens), "frame_index": 1}, Context())
    assert np.all(decode_mask(preview) == 255)
    assert tuple(preview["dimensions"]) == (32, 32)
    params = make_params(tmp_path, "dfi", reference_dir=str(refs), generated_dir=str(gens),
                         inter_denoise=1, inter_denoise_size=1, inter_blur=1)
    result = run(params, Context())
    assert tuple(result["dimensions"]) == (32, 32)
    assert np.array_equal(open_rgb(output_images(result)[1]), solid(20))
    assert all(p.read_bytes() == data for p, data in before.items())


@pytest.mark.parametrize("field,value", [
    ("source_denoise", -1), ("tolerance", 0), ("tolerance", float("nan")),
    ("expand", 51), ("smooth", 2), ("inter_denoise", 26),
    ("inter_denoise_size", 2), ("inter_denoise_frequency", 0),
    ("inter_blur", 6), ("refresh_frequency", 31), ("preserve", 101),
    ("frame_limit", -1), ("frame_limit", True),
])
def test_dfi_rejects_invalid_controls_before_publishing(tmp_path, field, value):
    refs = images(tmp_path / "references", [solid(50), solid(50)])
    gens = images(tmp_path / "generated", [solid(20), solid(160)])
    params = make_params(tmp_path, "dfi", reference_dir=str(refs), generated_dir=str(gens), **{field: value})
    with pytest.raises(EngineError) as exc:
        run(params, Context())
    assert exc.value.code == INVALID_PARAMS
    assert not (tmp_path / "outputs" / "dfi").exists()


@pytest.mark.parametrize("field,value", [(None, None), ("overlay", 0.5),
                                         pytest.param("blend", 0.5, marks=needs_opencv),
                                         pytest.param("normalize", True, marks=needs_opencv)])
def test_deflicker_noop_and_each_stage(tmp_path, field, value):
    frames = images(tmp_path / "frames", [solid(30), solid(180), solid(80)])
    base = make_params(tmp_path, "deflicker", frames_dir=str(frames), blend=0, overlay=0, normalize=False)
    noop = output_images(run({**base, "output_name": "noop"}, Context()))
    assert len(noop) == 3
    assert all(np.array_equal(open_rgb(p), open_rgb(q)) for p, q in zip(noop, sorted(frames.iterdir())))
    if field is not None:
        files = output_images(run({**base, "output_name": field, field: value}, Context()))
        assert len(files) == 3
        assert any(not np.array_equal(open_rgb(x), open_rgb(y)) for x, y in zip(noop, files)), field


@needs_opencv
def test_blend_uses_adjacent_stage_inputs_not_previous_output(tmp_path):
    # Source dyndef traverses from last to first, using unmodified neighbours.
    frames = images(tmp_path / "frames", [solid(20), solid(100), solid(220)])
    params = make_params(tmp_path, "deflicker", frames_dir=str(frames), blend=0.5)
    result = output_images(run(params, Context()))
    assert [int(open_rgb(p)[0, 0, 0]) for p in result] == [20, 60, 160]


@needs_opencv
def test_normalize_uses_adjacent_luminance_weighted_images(tmp_path):
    # Source norm weights previous/current by their own average brightness.
    frames = images(tmp_path / "frames", [solid(20), solid(100), solid(220)])
    params = make_params(tmp_path, "deflicker", frames_dir=str(frames), normalize=True)
    output = output_images(run(params, Context()))
    observed = [int(open_rgb(p)[0, 0, 0]) for p in output]
    assert observed[0] == 20
    assert abs(observed[1] - 87) <= 1
    assert abs(observed[2] - 183) <= 1


@needs_opencv
def test_normalize_black_pair_stays_black_without_division_error(tmp_path):
    frames = images(tmp_path / "frames", [solid(0), solid(0)])
    params = make_params(tmp_path, "deflicker", frames_dir=str(frames), normalize=True)
    result = output_images(run(params, Context()))
    assert all(np.array_equal(open_rgb(frame), solid(0)) for frame in result)


def test_overlay_uses_prior_frame_branch_and_upstream_brightness_threshold(tmp_path):
    frames = images(tmp_path / "frames", [solid(30), solid(100), solid(220)])
    base = make_params(tmp_path, "deflicker", frames_dir=str(frames))
    low = output_images(run({**base, "output_name": "low", "overlay": 0.39}, Context()))
    high = output_images(run({**base, "output_name": "high", "overlay": 0.4}, Context()))
    # The upstream first output and second output are both the first pair's
    # overlay; 0.4 or above then adds 10 to each channel, clipped at 255.
    lo = [int(open_rgb(p)[0, 0, 0]) for p in low]
    hi = [int(open_rgb(p)[0, 0, 0]) for p in high]
    assert lo[0] == lo[1] and hi[0] == hi[1]
    assert abs(lo[0] - 70) <= 1
    assert abs(hi[0] - 79) <= 1
    assert abs(hi[2] - 211) <= 1


def test_fuse_pairs_frame_order_resizes_and_respects_strength(tmp_path):
    style = images(tmp_path / "style", [solid(80, 16), solid(150, 16)])
    video = images(tmp_path / "video", [solid(140), solid(60)])
    base = make_params(tmp_path, "fuse", style_dir=str(style), video_dir=str(video))
    low = output_images(run({**base, "output_name": "low", "strength": 0.1}, Context()))
    high = output_images(run({**base, "output_name": "high", "strength": 1.0}, Context()))
    assert len(low) == len(high) == 2
    assert open_rgb(high[0]).shape == (32, 32, 3)
    assert not np.array_equal(open_rgb(low[0]), open_rgb(high[0]))


def test_fuse_overlay_branches_on_style_not_video(tmp_path):
    style = images(tmp_path / "style", [solid(200)])
    video = images(tmp_path / "video", [solid(50)])
    params = make_params(tmp_path, "fuse", style_dir=str(style), video_dir=str(video), strength=0.5)
    out = output_images(run(params, Context()))
    assert abs(int(open_rgb(out[0])[0, 0, 0]) - 108) <= 1


def test_interrupt_does_not_publish_partial_or_modify_sources(tmp_path):
    refs, gens = reference_and_generated(tmp_path)
    before = {p: p.read_bytes() for p in gens.iterdir()}
    params = make_params(tmp_path, "dfi", reference_dir=str(refs), generated_dir=str(gens))
    with pytest.raises(EngineError) as exc:
        run(params, Context(interrupt_after=4))
    assert exc.value.code == INTERRUPTED
    assert not (tmp_path / "outputs" / "dfi").exists()
    assert not list((tmp_path / "outputs").iterdir())
    assert all(p.read_bytes() == data for p, data in before.items())


def test_interrupt_during_large_sequence_discovery_stops_before_publish(tmp_path):
    refs = images(tmp_path / "references", [solid(50)] * 10)
    gens = images(tmp_path / "generated", [solid(40)] * 10)
    params = make_params(tmp_path, "dfi", reference_dir=str(refs), generated_dir=str(gens))
    ctx = Context(interrupt_after=4)
    with pytest.raises(EngineError) as exc:
        run(params, ctx)
    assert exc.value.code == INTERRUPTED
    assert ctx.calls == 4
    assert not list((tmp_path / "outputs").iterdir())
    assert len(list(refs.iterdir())) == len(list(gens.iterdir())) == 10


@pytest.mark.parametrize("operation", ["dfi", "preview", "deflicker", "fuse"])
def test_working_memory_limit_fails_before_decoding_or_output_allocation(tmp_path, monkeypatch, operation):
    import sdfx.temporal_lab as lab

    refs, gens = reference_and_generated(tmp_path)
    base = make_params(tmp_path, operation, reference_dir=str(refs), generated_dir=str(gens),
                       frames_dir=str(refs), style_dir=str(refs), video_dir=str(gens), frame_index=1)
    monkeypatch.setattr(lab, "MAX_WORKING_BYTES", 65536)
    monkeypatch.setattr(lab, "_read_rgb", lambda path: pytest.fail("decoded a frame before budget validation"))
    with pytest.raises(EngineError) as exc:
        run(base, Context())
    assert exc.value.code == INVALID_PARAMS
    assert "memory" in exc.value.message.lower()
    assert not list((tmp_path / "outputs").iterdir())


@pytest.mark.parametrize("operation", [[], {}, None, 0, True])
def test_nonstring_operation_is_rejected_without_type_error(operation):
    with pytest.raises(EngineError) as exc:
        run({"operation": operation}, Context())
    assert exc.value.code == INVALID_PARAMS


@pytest.mark.parametrize("bad_metadata", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_video_metadata_is_reported_and_capture_closed(tmp_path, monkeypatch, bad_metadata):
    import sdfx.temporal_lab as lab
    cv2 = pytest.importorskip("cv2")
    video = tmp_path / "video.avi"
    video.write_bytes(b"placeholder")

    class Capture:
        closed = False

        def isOpened(self):
            return True

        def get(self, key):
            return {cv2.CAP_PROP_FPS: 10, cv2.CAP_PROP_FRAME_WIDTH: bad_metadata,
                    cv2.CAP_PROP_FRAME_HEIGHT: 32, cv2.CAP_PROP_FRAME_COUNT: 3}[key]

        def release(self):
            self.closed = True

    capture = Capture()
    class FakeCV:
        CAP_PROP_FPS = cv2.CAP_PROP_FPS
        CAP_PROP_FRAME_WIDTH = cv2.CAP_PROP_FRAME_WIDTH
        CAP_PROP_FRAME_HEIGHT = cv2.CAP_PROP_FRAME_HEIGHT
        CAP_PROP_FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT

        @staticmethod
        def VideoCapture(path):
            return capture

    monkeypatch.setattr(lab, "_cv", lambda: FakeCV)
    params = make_params(tmp_path, "extract", video_path=str(video))
    with pytest.raises(EngineError) as exc:
        run(params, Context())
    assert exc.value.code == INVALID_PARAMS
    assert capture.closed
    assert not list((tmp_path / "outputs").iterdir())


def test_extract_memory_budget_rejects_before_video_frame_decode(tmp_path, monkeypatch):
    import sdfx.temporal_lab as lab
    cv2 = pytest.importorskip("cv2")
    video = tmp_path / "video.avi"
    video.write_bytes(b"placeholder")

    class Capture:
        closed = False

        def isOpened(self):
            return True

        def get(self, key):
            return {cv2.CAP_PROP_FPS: 10, cv2.CAP_PROP_FRAME_WIDTH: 32,
                    cv2.CAP_PROP_FRAME_HEIGHT: 32, cv2.CAP_PROP_FRAME_COUNT: 3}[key]

        def read(self):
            pytest.fail("decoded a video frame before budget validation")

        def release(self):
            self.closed = True

    capture = Capture()
    class FakeCV:
        CAP_PROP_FPS = cv2.CAP_PROP_FPS
        CAP_PROP_FRAME_WIDTH = cv2.CAP_PROP_FRAME_WIDTH
        CAP_PROP_FRAME_HEIGHT = cv2.CAP_PROP_FRAME_HEIGHT
        CAP_PROP_FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT

        @staticmethod
        def VideoCapture(path):
            return capture

    monkeypatch.setattr(lab, "_cv", lambda: FakeCV)
    monkeypatch.setattr(lab, "MAX_WORKING_BYTES", 65536)
    params = make_params(tmp_path, "extract", video_path=str(video))
    with pytest.raises(EngineError) as exc:
        run(params, Context())
    assert exc.value.code == INVALID_PARAMS
    assert "memory" in exc.value.message.lower()
    assert capture.closed
    assert not list((tmp_path / "outputs").iterdir())


def test_extract_checks_actual_frame_when_decoder_underreports_dimensions(tmp_path, monkeypatch):
    import sdfx.temporal_lab as lab
    cv2 = pytest.importorskip("cv2")
    video = tmp_path / "underreported.avi"
    video.write_bytes(b"placeholder")

    class Capture:
        closed = False

        def isOpened(self):
            return True

        def get(self, key):
            return {cv2.CAP_PROP_FPS: 10, cv2.CAP_PROP_FRAME_WIDTH: 16,
                    cv2.CAP_PROP_FRAME_HEIGHT: 16, cv2.CAP_PROP_FRAME_COUNT: 1}[key]

        def read(self):
            return True, np.zeros((32, 32, 3), dtype=np.uint8)

        def release(self):
            self.closed = True

    capture = Capture()
    class FakeCV:
        CAP_PROP_FPS = cv2.CAP_PROP_FPS
        CAP_PROP_FRAME_WIDTH = cv2.CAP_PROP_FRAME_WIDTH
        CAP_PROP_FRAME_HEIGHT = cv2.CAP_PROP_FRAME_HEIGHT
        CAP_PROP_FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT
        COLOR_BGR2RGB = cv2.COLOR_BGR2RGB

        @staticmethod
        def VideoCapture(path):
            return capture

        @staticmethod
        def cvtColor(*args):
            pytest.fail("converted an oversized decoded frame")

    monkeypatch.setattr(lab, "_cv", lambda: FakeCV)
    monkeypatch.setattr(lab, "MAX_WORKING_BYTES", 131072)
    params = make_params(tmp_path, "extract", video_path=str(video))
    with pytest.raises(EngineError) as exc:
        run(params, Context())
    assert exc.value.code == INVALID_PARAMS
    assert "memory" in exc.value.message.lower()
    assert capture.closed
    assert not list((tmp_path / "outputs").iterdir())


def test_extract_real_video_original_and_lower_rate_without_duplicate_frames(tmp_path):
    cv2 = pytest.importorskip("cv2")
    video = tmp_path / "test.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32))
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG video writer unavailable")
    try:
        for color in (10, 30, 50, 70, 90, 110, 130, 150, 170, 190):
            writer.write(solid(color))
    finally:
        writer.release()
    base = make_params(tmp_path, "extract", video_path=str(video))
    original = run({**base, "output_name": "original", "fps": 0}, Context())
    lower = run({**base, "output_name": "lower", "fps": 5}, Context())
    assert original["frame_count"] == 10
    assert lower["frame_count"] == 5
    assert lower["fps"] == 5 and lower["source_fps"] == pytest.approx(10)
    assert len(output_images(lower)) == 5
    with pytest.raises(EngineError) as exc:
        run({**base, "output_name": "too-fast", "fps": 20}, Context())
    assert exc.value.code == INVALID_PARAMS
    assert not (tmp_path / "outputs" / "too-fast").exists()


def test_missing_opencv_reports_unsupported_and_cleans_staging(tmp_path, monkeypatch):
    import builtins
    original = builtins.__import__
    def blocked(name, *args, **kwargs):
        if name.split(".")[0] == "cv2":
            raise ImportError("cv2 blocked for test")
        return original(name, *args, **kwargs)
    refs, gens = reference_and_generated(tmp_path)
    params = make_params(tmp_path, "dfi", reference_dir=str(refs), generated_dir=str(gens),
                         inter_denoise=1, inter_denoise_size=1, inter_blur=1)
    before = {p: p.read_bytes() for folder in (refs, gens) for p in folder.iterdir()}
    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(EngineError) as error:
        run(params, Context())
    assert error.value.code == UNSUPPORTED and "OpenCV" in error.value.message
    assert list((tmp_path / "outputs").iterdir()) == []
    assert all(p.read_bytes() == data for p, data in before.items())
