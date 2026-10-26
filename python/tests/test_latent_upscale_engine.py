"""CPU integration using random tiny diffusers components, never downloaded/checkpoint weights."""
import io
from types import SimpleNamespace
import pytest

torch = pytest.importorskip('torch')
diffusers = pytest.importorskip('diffusers')
from PIL import Image
from sdfx.engine import Engine
from sdfx.params import GenerationParams, GenerationRequest, Img2ImgParams
from sdfx.protocol import Emitter, JobContext
from sdfx import latent_upscale


@pytest.mark.parametrize('masked,sdxl,nine_channel', [(False, False, False), (True, False, False),
                                                   (True, False, True), (False, True, False), (True, True, False)])
def test_actual_engine_run_passes_resized_source_latents_and_preserves_batch(masked, sdxl, nine_channel):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
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
            pipe = diffusers.StableDiffusionXLPipeline(**common, text_encoder_2=None, tokenizer_2=None, add_watermarker=False)
        else:
            cls = diffusers.StableDiffusionInpaintPipeline if nine_channel else diffusers.StableDiffusionPipeline
            pipe = cls(**common, safety_checker=None, feature_extractor=None, requires_safety_checker=False)
        pipe.set_progress_bar_config(disable=True)
        engine = Engine()
        engine.pipe, engine.kind, engine.dtype = pipe, 'sdxl' if sdxl else ('sd15-inpaint' if nine_channel else 'sd15'), torch.float32
        engine.sdxl = sdxl
        engine._extra, engine._ms_overrides = {}, {}
        engine.base_sched_config = dict(pipe.scheduler.config)
        gp = GenerationParams(steps=3, batch_size=2, width=64, height=64,
            img2img=Img2ImgParams(init_image='unused', denoising_strength=.4),
            hooks={'latent_upscale': {'method': 'nearest-exact', 'scheduler': 'simple'}})
        source = Image.new('RGB', (48, 32), (100, 130, 180))
        target = source.resize((64, 64))
        mask = Image.new('L', source.size, 255)
        calls = []
        original_encode = vae.encode
        def encode(image, *a, **kw):
            calls.append(tuple(image.shape[-2:]))
            return original_encode(image, *a, **kw)
        vae.encode = encode
        n, strength, actual = latent_upscale.full_steps(3, .4)
        ctx = JobContext(Emitter(io.BytesIO()), 'cpu')
        embeds = {'prompt_embeds': torch.zeros(2, 4, 8)}
        if sdxl:
            embeds['pooled_prompt_embeds'] = torch.zeros(2, 8)
        images, _ = engine._run('inpaint' if masked else 'img2img', gp, GenerationRequest(params=gp), ctx,
            sampler='Euler', schedule='Simple', cfg_scale=1, embeds=embeds,
            steps=n, strength=strength, expected_steps=actual, w=64, h=64, seeds=[1, 2],
            pass_='base', image_idx=0, images_total=2, image=[target] * 2,
            mask=[Image.new('L', (64, 64), 255)] * 2, latent_source=(source, mask if masked else None))
        assert len(images) == 2 and all(im.size == (64, 64) for im in images)
        assert calls and set(calls) == {(32, 48)}, 'no target-sized pixel re-encoding is allowed'
        assert len(calls) == (2 if masked else 1)
    finally:
        torch.set_num_threads(previous_threads)
