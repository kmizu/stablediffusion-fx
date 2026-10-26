"""ADetailer re-implementation: unit config, prompt splitting, mask post-processing."""
import numpy as np
import pytest

from sdfx import adetailer as ad

Image = pytest.importorskip("PIL.Image")


def _mask(size, box):
    from PIL import ImageDraw
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rectangle(box, fill=255)
    return m


def test_units_skip_none_and_xyz_overrides_hit_the_first():
    u = ad.normalize_units({"units": [{"model": "face_yolov8n.pt"}, {"model": "None"}, {"model": "hand_yolov8n.pt",
                                                                                         "confidence": "0.5"}],
                            "denoising_strength": 0.7})
    assert [x["model"] for x in u] == ["face_yolov8n.pt", "hand_yolov8n.pt"]
    assert u[0]["denoising_strength"] == 0.7 and u[1]["denoising_strength"] == 0.4
    assert u[1]["confidence"] == 0.5 and isinstance(u[0]["inpaint_only_masked"], bool)
    assert ad.normalize_units({"model": "opencv_face"})[0]["model"] == "opencv_face"


def test_prompts_placeholder_and_sep():
    assert ad.unit_prompts("", "main", 2) == ["main", "main"]
    assert ad.unit_prompts("[PROMPT], smile", "a girl", 1) == ["a girl, smile"]
    assert ad.unit_prompts("left eye [SEP] right eye", "m", 3) == ["left eye", "right eye", "right eye"]


def test_offset_and_dilate_erode():
    pytest.importorskip("cv2")
    m = _mask((64, 64), (20, 20, 29, 29))
    moved = np.asarray(ad.offset_dilate(m, 5, 5, 0))
    assert moved[15:25, 25:35].min() == 255 and moved[25, 21] == 0   # right 5, up 5
    grown = np.asarray(ad.offset_dilate(m, 0, 0, 5))
    assert (grown > 0).sum() > (np.asarray(m) > 0).sum()
    shrunk = np.asarray(ad.offset_dilate(m, 0, 0, -5))
    assert 0 < (shrunk > 0).sum() < (np.asarray(m) > 0).sum()


def test_merge_invert_and_filters():
    a, b = _mask((32, 32), (0, 0, 9, 9)), _mask((32, 32), (20, 20, 31, 31))
    merged = ad.merge_invert([a, b], "Merge")
    assert len(merged) == 1 and np.asarray(merged[0])[5, 5] == 255 and np.asarray(merged[0])[25, 25] == 255
    inv = np.asarray(ad.merge_invert([a, b], "Merge and Invert")[0])
    assert inv[5, 5] == 0 and inv[15, 15] == 255
    assert ad.merge_invert([a, b], "None") == [a, b]
    d = ad.Detection([(0, 0, 10, 10), (0, 0, 30, 30), (0, 0, 20, 20)], [a, b, a], [0.9, 0.8, 0.7])
    k = ad.filter_k_largest(d, 2)
    assert k.boxes == [(0, 0, 30, 30), (0, 0, 20, 20)]
    r = ad.filter_by_ratio(d, 0.2, 1.0, (32, 32))
    assert r.boxes == [(0, 0, 30, 30), (0, 0, 20, 20)]


def test_resolve_model_paths(tmp_path):
    p = tmp_path / "my_face.pt"
    p.write_bytes(b"x")
    assert ad.resolve_model(str(p)) == str(p)
    assert ad.resolve_model("my_face.pt", [str(tmp_path)]) == str(p)
    assert ad.resolve_model("opencv_face") == "opencv_face"
    with pytest.raises(FileNotFoundError):
        ad.resolve_model("nothing.bin", [str(tmp_path)])


def test_match_colors_restores_contrast():
    pytest.importorskip("cv2")
    rng = np.random.default_rng(0)
    ref = Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8))
    hazy = Image.fromarray((np.asarray(ref) * 0.6 + 60).astype(np.uint8))
    fixed = np.asarray(ad.match_colors(hazy, ref)).astype(float)
    r = np.asarray(ref).astype(float)
    assert abs(fixed.std() - r.std()) < 0.1 * r.std() < abs(np.asarray(hazy).astype(float).std() - r.std())
