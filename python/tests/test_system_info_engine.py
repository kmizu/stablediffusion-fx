"""Actual CPU generation contract with tiny random components, no model downloads/GPU."""
import io
import pytest
torch = pytest.importorskip('torch')
diffusers = pytest.importorskip('diffusers')
from PIL import Image
from sdfx.engine import Engine
from sdfx.protocol import Emitter, JobContext


def test_actual_engine_benchmark_skips_png_io_and_restores_ordinary_rgb(tmp_path, monkeypatch):
    from sdfx import engine as module
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(8)
        vae = diffusers.AutoencoderKL(in_channels=3, out_channels=3, latent_channels=4,
            down_block_types=('DownEncoderBlock2D',) * 4, up_block_types=('UpDecoderBlock2D',) * 4,
            block_out_channels=(8, 8, 8, 8), layers_per_block=1, norm_num_groups=4)
        unet = diffusers.UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4,
            down_block_types=('CrossAttnDownBlock2D',), up_block_types=('CrossAttnUpBlock2D',),
            block_out_channels=(8,), layers_per_block=1, norm_num_groups=4, cross_attention_dim=8, attention_head_dim=2)
        pipe = diffusers.StableDiffusionPipeline(vae=vae, text_encoder=None, tokenizer=None, unet=unet,
            scheduler=diffusers.EulerDiscreteScheduler(steps_offset=1), safety_checker=None,
            feature_extractor=None, requires_safety_checker=False)
        pipe.set_progress_bar_config(disable=True)
        engine = Engine()
        engine.pipe, engine.kind, engine.dtype = pipe, 'sd15', torch.float32
        engine.model = {'name': 'tiny-test', 'path': 'not-read', 'hash': '1234', 'kind': 'sd15'}
        engine.base_sched_config = dict(pipe.scheduler.config)
        # Exercise real sampling/VAE/decode/save. Only text encoding is supplied directly to avoid downloading CLIP.
        engine._encode_scheduled = lambda *a, **kw: {'prompt_embeds': torch.zeros(1, 4, 8), 'negative_prompt_embeds': torch.zeros(1, 4, 8)}
        saved = []
        original_save = module.save_png
        def save(image, path, *a, **kw):
            saved.append(path)
            return original_save(image, path, *a, **kw)
        monkeypatch.setattr(module, 'save_png', save)
        request = {'params': {'steps': 1, 'width': 64, 'height': 64, 'seed': 42, 'sampler_name': 'Euler', 'cfg_scale': 1},
                   'output_dir': str(tmp_path), 'preview_method': 'none'}
        context = lambda: JobContext(Emitter(io.BytesIO()), 'cpu')
        first = engine.txt2img(request, context())['images'][0]['path']
        assert len(saved) == 1
        result = engine.system_benchmark({'level': 'quick', 'steps': 'turbo', 'warmup': False, 'width': 256, 'height': 256}, context())
        assert result['status'] == 'complete' and result['results'][0]['status'] == 'complete'
        assert len(saved) == 1, 'benchmark must not spend measured time encoding/writing PNGs'
        assert not engine._benchmark_no_save
        restored = engine.txt2img(request, context())['images'][0]['path']
        assert len(saved) == 2
        with Image.open(first) as a, Image.open(restored) as b:
            assert a.convert('RGB').tobytes() == b.convert('RGB').tobytes()
    finally:
        torch.set_num_threads(previous)
