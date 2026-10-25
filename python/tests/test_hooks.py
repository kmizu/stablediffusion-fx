import random

import numpy as np
import pytest

from sdfx import hooks as H


def test_hook_config():
    hooks = {"freeu": {"b1": 1.2}, "pag": {"enabled": False}, "freeu_xyz": {"a": 1}, "hypertile": True}
    c = H.hook_config(hooks, "freeu", H.FREEU_DEFAULTS[False])
    assert c == {"b1": 1.2, "b2": 1.6, "s1": 0.9, "s2": 0.2}
    assert H.hook_config(hooks, "pag", H.PAG_DEFAULTS) is None
    assert H.hook_config(hooks, "hypertile", H.HYPERTILE_DEFAULTS) == H.HYPERTILE_DEFAULTS
    assert H.hook_config(hooks, "dynamic_thresholding", H.DYNTHRESH_DEFAULTS) is None


def test_interpret_scale():
    assert H.interpret_scale(7.0, "Constant", 0, 0.5) == 7.0
    assert H.interpret_scale(7.0, "Linear Down", 1.0, 0.5) == pytest.approx(1 + 6 * 0.5)
    assert H.interpret_scale(7.0, "Linear Up", 0.0, 0.25) == pytest.approx(1.75)
    assert H.interpret_scale(7.0, "Power Up", 0.0, 0.5, 2.0) == pytest.approx(7 * 0.25)


def _data(seed=0):
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(2, 4, 16, 16))
    c = u + rng.normal(scale=0.5, size=u.shape)
    return c, u


def test_dynthresh_equals_cfg_when_mimic_equals_cfg():
    c, u = _data()
    out = H.dynthresh(c, u, 7.0, mimic_scale=7.0, threshold_percentile=1.0)
    np.testing.assert_allclose(out, u + 7.0 * (c - u), rtol=1e-10, atol=1e-10)


def test_dynthresh_clamps_to_mimic_range():
    c, u = _data(1)
    cfg, mimic = 15.0, 5.0
    out = H.dynthresh(c, u, cfg, mimic_scale=mimic, threshold_percentile=1.0)
    mim = u + mimic * (c - u)
    flat_o = out.reshape(2, 4, -1)
    flat_m = mim.reshape(2, 4, -1)
    dev_o = np.abs(flat_o - flat_o.mean(axis=2, keepdims=True)).max(axis=2)
    dev_m = np.abs(flat_m - flat_m.mean(axis=2, keepdims=True)).max(axis=2)
    np.testing.assert_allclose(dev_o, dev_m, rtol=1e-8)
    # channel means are those of the full-CFG target
    cfg_t = (u + cfg * (c - u)).reshape(2, 4, -1)
    np.testing.assert_allclose(flat_o.mean(axis=2), cfg_t.mean(axis=2), rtol=1e-8, atol=1e-10)


def test_dynthresh_variants_run():
    c, u = _data(2)
    for kw in ({"variability_measure": "STD"}, {"scaling_startpoint": "ZERO"}, {"separate_feature_channels": False},
               {"threshold_percentile": 0.95, "interpolate_phi": 0.5}, {"mimic_mode": "Cosine Down", "cfg_mode": "Linear Up"}):
        out = H.dynthresh(c, u, 9.0, frac=0.3, mimic_scale=4.0, **kw)
        assert out.shape == c.shape and np.isfinite(out).all()


def test_dynthresh_torch_matches_numpy():
    torch = pytest.importorskip("torch")
    c, u = _data(3)
    a = H.dynthresh(c, u, 12.0, mimic_scale=6.0, threshold_percentile=0.98)
    b = H.dynthresh(torch.tensor(c), torch.tensor(u), 12.0, mimic_scale=6.0, threshold_percentile=0.98)
    np.testing.assert_allclose(a, b.numpy(), rtol=1e-6, atol=1e-8)


def test_random_divisor_and_split():
    assert H.random_divisor(64, 32, 1) == 2
    assert H.random_divisor(64, 200, 1) == 1
    r = random.Random(0)
    assert H.random_divisor(96, 32, 3, r) in (3, 2, 1)
    # 1024x1024 image -> latent 128x128, tile 256 -> latent tile 32 -> 4x4 tiles at depth 0
    assert H.hypertile_split(128 * 128, 128, 128, 256, 1, 0, False) == (4, 4, 128, 128)
    assert H.hypertile_split(64 * 64, 128, 128, 256, 1, 0, False) is None          # depth 1 not enabled
    assert H.hypertile_split(64 * 64, 128, 128, 256, 1, 1, False) == (2, 2, 64, 64)
    # scale_depth doubles the tile at depth 1 -> 64 >= 64 -> a single tile -> no split
    assert H.hypertile_split(64 * 64, 128, 128, 256, 1, 1, True) is None


@pytest.mark.parametrize("v,expected", [
    (True, True), (False, False), ("enable", True), ("disable", False), ("true", True), ("False", False),
    ("on", True), (1, True), (0, False), (None, True),
])
def test_coerce_bool(v, expected):
    assert H.coerce_bool(v, True) is expected


@pytest.mark.parametrize("v,expected", [
    (["mid"], ["mid"]), ("middle", ["mid"]), ("down", ["down"]), ("up", ["up"]),
    ("mid, up.block_0", ["mid", "up.block_0"]), (None, ["mid"]), ("", ["mid"]), (["middle", "down.block_1"], ["mid", "down.block_1"]),
])
def test_coerce_layers(v, expected):
    assert H.coerce_layers(v) == expected


def test_normalizers():
    d = H.normalize_dynthresh({"separate_feature_channels": "disable", "mimic_scale": "5", "variability_measure": "ad"})
    assert d["separate_feature_channels"] is False and d["mimic_scale"] == 5.0 and d["variability_measure"] == "AD"
    assert H.normalize_pag({"scale": 2, "layers": "middle"}) == {"scale": 2.0, "adaptive_scale": 0.0, "layers": ["mid"]}
    assert H.normalize_hypertile({"tile_size": 128.0, "scale_depth": "enable"})["scale_depth"] is True
    assert H.hook_config({"pag": {"enabled": "disable"}}, "pag", H.PAG_DEFAULTS) is None
    assert H.hook_config({"freeu": "enable"}, "freeu", H.FREEU_DEFAULTS[True]) == H.FREEU_DEFAULTS[True]
    c, u = _data(4)
    a = H.dynthresh(c, u, 9.0, mimic_scale=4.0, separate_feature_channels="disable")
    b = H.dynthresh(c, u, 9.0, mimic_scale=4.0, separate_feature_channels=False)
    np.testing.assert_allclose(a, b)
