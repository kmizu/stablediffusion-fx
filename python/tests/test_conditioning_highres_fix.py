import math
import pytest

from sdfx.conditioning_highres_fix import resolve_weight


@pytest.mark.parametrize(('mode', 'expected'), [
    ('disabled', None), ('empty', 1.0), ('lowest', 0.9), ('low', 0.8),
    ('medium', 0.5), ('high', 0.2), ('highest', 0.1), ('full', 0.0),
])
def test_upstream_mode_weights(mode, expected):
    actual = resolve_weight(mode, 0.4)
    assert actual is None if expected is None else actual == pytest.approx(expected)


@pytest.mark.parametrize(('mode', 'low', 'high'), [
    ('empty', 1., 1.), ('lowest', 1., .75), ('low', 1., .5), ('medium', .5, .5),
    ('high', 0., .5), ('highest', 0., .25), ('full', 0., 0.),
])
def test_denoise_endpoints(mode, low, high):
    assert resolve_weight(mode, 0.) == pytest.approx(low)
    assert resolve_weight(mode, 1.) == pytest.approx(high)


def test_missing_strength_does_not_change_existing_conditioning():
    assert resolve_weight('high', None) is None
    assert resolve_weight('disabled', 0.6) is None


@pytest.mark.parametrize('bad', [-.01, 1.01, math.nan, math.inf, True, '0.5'])
def test_invalid_strength_rejected(bad):
    with pytest.raises(ValueError): resolve_weight('high', bad)


def test_unknown_mode_rejected():
    with pytest.raises(ValueError): resolve_weight('wrong', .5)


def test_engine_contract_advertises_hook_in_dry_run():
    from sdfx import hooks
    from sdfx.dryrun import FakeEngine
    assert 'conditioning_highres_fix' in hooks.KNOWN_HOOKS
    assert 'conditioning_highres_fix' in FakeEngine().features()


def test_pass_strength_preserves_user_denoise_not_scheduler_minimum():
    from types import SimpleNamespace
    from sdfx.conditioning_highres_fix import source_denoising
    gp = SimpleNamespace(img2img=SimpleNamespace(denoising_strength=.01), hires=SimpleNamespace(denoising_strength=.2))
    assert source_denoising('base', gp, effective=.1) == .01
    assert source_denoising('hires', gp, effective=.25) == .2
    assert source_denoising('img2img_hires', gp, effective=.1, requested=.02) == .02
    gp.img2img = None
    assert source_denoising('base', gp, effective=1.) is None


def test_scoped_conditioning_interpolates_pixel_input_and_restores_methods():
    import numpy as np
    from sdfx.conditioning_highres_fix import conditioning_scope
    source = np.repeat(np.array([[[[.8, -.4], [.2, -.6]]]], dtype=np.float32), 3, axis=1)
    mask = np.array([[[[0., 1.], [1., 0.]]]], dtype=np.float32)
    class Processor:
        def preprocess(self, image, **kwargs): return image.copy()
    class Pipeline:
        image_processor = Processor()
        def prepare_mask_latents(self, mask, image, *args, **kwargs): return image
    pipe = Pipeline(); processor = pipe.image_processor
    before_pipe, before_processor = dict(pipe.__dict__), dict(processor.__dict__)
    with conditioning_scope(pipe, .25):
        image = processor.preprocess(source)
        actual = pipe.prepare_mask_latents(mask, image * (mask < .5), 1)
        np.testing.assert_allclose(actual, source * (1 - mask * .25), atol=1e-7)
        np.testing.assert_array_equal(image, source)
    assert pipe.__dict__ == before_pipe and processor.__dict__ == before_processor


def test_scope_restores_methods_after_failure_and_weight_one_is_noop():
    from sdfx.conditioning_highres_fix import conditioning_scope
    class Processor:
        def preprocess(self, image, **kwargs): return image
    class Pipeline:
        image_processor = Processor()
        def prepare_mask_latents(self, mask, image, *args, **kwargs): return image
    pipe = Pipeline()
    with conditioning_scope(pipe, 1.):
        assert 'prepare_mask_latents' not in pipe.__dict__
    with pytest.raises(RuntimeError, match='injected'):
        with conditioning_scope(pipe, .5):
            raise RuntimeError('injected')
    assert 'prepare_mask_latents' not in pipe.__dict__
    assert 'preprocess' not in pipe.image_processor.__dict__


def test_scope_restores_existing_instance_overrides_on_interruption():
    from types import SimpleNamespace
    from sdfx.conditioning_highres_fix import conditioning_scope
    previous_preprocess = lambda image, **kwargs: image
    previous_prepare = lambda *args, **kwargs: None
    processor = SimpleNamespace(preprocess=previous_preprocess)
    pipe = SimpleNamespace(image_processor=processor, prepare_mask_latents=previous_prepare)
    with pytest.raises(KeyboardInterrupt):
        with conditioning_scope(pipe, .5):
            raise KeyboardInterrupt()
    assert processor.preprocess is previous_preprocess
    assert pipe.prepare_mask_latents is previous_prepare


def test_scope_reports_unsupported_precomputed_latent_without_leaking_hooks():
    import numpy as np
    from sdfx.protocol import EngineError, UNSUPPORTED
    from sdfx.conditioning_highres_fix import conditioning_scope
    class Processor:
        def preprocess(self, image, **kwargs): return image
    class Pipeline:
        image_processor = Processor()
        def prepare_mask_latents(self, mask, image, *args, **kwargs): return image
    pipe = Pipeline()
    with pytest.raises(EngineError) as error:
        with conditioning_scope(pipe, .5):
            pipe.image_processor.preprocess(np.zeros((1, 4, 8, 8)))
            pipe.prepare_mask_latents(np.zeros((1, 1, 8, 8)), np.zeros((1, 4, 8, 8)))
    assert error.value.code == UNSUPPORTED
    assert 'prepare_mask_latents' not in pipe.__dict__


def test_mask_conditioning_formula_handles_binary_threshold_and_outside_pixels():
    import numpy as np
    from sdfx.conditioning_highres_fix import condition_pixels
    image = np.array([[[[1., -1., .5]]]], dtype=np.float32)
    mask = np.array([[[[.49, .5, .9]]]], dtype=np.float32)
    actual = condition_pixels(image, mask, .4)
    np.testing.assert_allclose(actual, [[[[1., -.6, .3]]]], atol=1e-7)


def test_mask_conditioning_preserves_image_precision():
    import numpy as np
    from sdfx.conditioning_highres_fix import condition_pixels
    for dtype in (np.float16, np.float32):
        image = np.full((1, 3, 8, 8), .5, dtype=dtype)
        mask = np.ones((1, 1, 8, 8), dtype=np.float32)
        result = condition_pixels(image, mask, .25)
        assert result.dtype == image.dtype
        np.testing.assert_array_equal(result, np.full_like(image, .375))
