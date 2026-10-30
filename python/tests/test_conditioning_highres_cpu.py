"""Tiny real Diffusers components; no model downloads or GPU."""
import io
import pytest

torch = pytest.importorskip('torch')
diffusers = pytest.importorskip('diffusers')
from PIL import Image


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_conditioning_tensor_keeps_native_vae_precision(dtype):
    from sdfx.conditioning_highres_fix import condition_pixels
    image = torch.full((1, 3, 8, 8), .5, dtype=dtype)
    mask = torch.ones((1, 1, 8, 8), dtype=torch.float32)
    result = condition_pixels(image, mask, .25)
    assert result.dtype == dtype
    torch.testing.assert_close(result, torch.full_like(image, .375), atol=0, rtol=0)


def test_fp16_safe_vae_latent_source_conditioning_runs_after_unmasked_encode():
    from types import SimpleNamespace
    from diffusers.image_processor import VaeImageProcessor
    from sdfx.latent_upscale import encode
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        vae = diffusers.AutoencoderKL(in_channels=3, out_channels=3, latent_channels=4,
            down_block_types=('DownEncoderBlock2D',) * 4, up_block_types=('UpDecoderBlock2D',) * 4,
            block_out_channels=(8, 8, 8, 8), layers_per_block=1, norm_num_groups=4, force_upcast=False).half()
        pipe = SimpleNamespace(vae=vae, image_processor=VaeImageProcessor(vae_scale_factor=8))
        source = Image.new('RGB', (64, 64), (100, 130, 180))
        generators = [torch.Generator('cpu').manual_seed(173)]
        encode(pipe, source, 64, 64, 'nearest-exact', generators, 'cpu', torch.float16)
        result = encode(pipe, source, 64, 64, 'nearest-exact', generators, 'cpu', torch.float16,
                        mask=Image.new('L', (64, 64), 255), conditioning_weight=.25)
        assert result.dtype == torch.float16 and torch.isfinite(result).all()
        assert all(p.dtype == torch.float16 for p in vae.parameters())
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize('latent_source', [False, True])
@pytest.mark.parametrize(('mode', 'remaining'), [('disabled', 0.), ('empty', 0.), ('high', .8), ('full', 1.)])
@pytest.mark.parametrize('sdxl', [False, True])
@pytest.mark.parametrize('nine_channel', [False, True])
def test_engine_inpaint_conditioning_respects_channel_capability(latent_source, mode, remaining, sdxl, nine_channel):
    from sdfx.engine import Engine
    from sdfx.params import GenerationParams, GenerationRequest, Img2ImgParams
    from sdfx.protocol import Emitter, JobContext
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        torch.manual_seed(1)
        vae = diffusers.AutoencoderKL(in_channels=3, out_channels=3, latent_channels=4,
            down_block_types=('DownEncoderBlock2D',) * 4, up_block_types=('UpDecoderBlock2D',) * 4,
            block_out_channels=(8, 8, 8, 8), layers_per_block=1, norm_num_groups=4)
        unet = diffusers.UNet2DConditionModel(sample_size=8, in_channels=9 if nine_channel else 4, out_channels=4,
            down_block_types=('CrossAttnDownBlock2D',), up_block_types=('CrossAttnUpBlock2D',),
            block_out_channels=(8,), layers_per_block=1, norm_num_groups=4, cross_attention_dim=8, attention_head_dim=2,
            **(dict(addition_embed_type='text_time', addition_time_embed_dim=2, projection_class_embeddings_input_dim=20) if sdxl else {}))
        common = dict(vae=vae, text_encoder=None, tokenizer=None, unet=unet,
                      scheduler=diffusers.EulerDiscreteScheduler(steps_offset=1))
        if sdxl:
            pipe = diffusers.StableDiffusionXLInpaintPipeline(**common, text_encoder_2=None, tokenizer_2=None, add_watermarker=False)
        else:
            pipe = diffusers.StableDiffusionInpaintPipeline(**common, safety_checker=None, feature_extractor=None,
                requires_safety_checker=False)
        pipe.set_progress_bar_config(disable=True)
        engine = Engine(); engine.pipe = pipe
        engine.kind = ('sdxl' if sdxl else 'sd15') + ('-inpaint' if nine_channel else '')
        engine.dtype = torch.float32
        engine.cuda = False; engine.device = torch.device('cpu')
        engine.sdxl = sdxl; engine._extra = {}; engine._ms_overrides = {}; engine.base_sched_config = dict(pipe.scheduler.config)
        gp = GenerationParams(steps=3, batch_size=1, width=64, height=64,
            img2img=Img2ImgParams(init_image='unused', denoising_strength=.4),
            hooks={'conditioning_highres_fix': {'mode': mode},
                   **({'latent_upscale': {'method': 'nearest-exact', 'scheduler': 'simple'}} if latent_source else {})})
        inputs = []
        original_encode = vae.encode
        def capture(image, *args, **kwargs):
            inputs.append(image.detach().cpu().clone())
            return original_encode(image, *args, **kwargs)
        vae.encode = capture
        image = Image.new('RGB', (64, 64), (100, 130, 180))
        mask = Image.new('L', (64, 64), 255)
        embeds = {'prompt_embeds': torch.zeros(1, 4, 8)}
        if sdxl: embeds['pooled_prompt_embeds'] = torch.zeros(1, 8)
        engine._run('inpaint', gp, GenerationRequest(params=gp), JobContext(Emitter(io.BytesIO()), 'cpu'),
            sampler='Euler', schedule='Simple', cfg_scale=1, embeds=embeds,
            steps=3, strength=.4, w=64, h=64, seeds=[1], pass_='base', image_idx=0, images_total=1,
            image=[image], mask=[mask], latent_source=(image, mask) if latent_source else None)
        original = pipe.image_processor.preprocess([image]).float()
        assert inputs
        torch.testing.assert_close(inputs[-1], original * (remaining if nine_channel else 0.), atol=1e-6, rtol=0)
    finally:
        torch.set_num_threads(previous)
