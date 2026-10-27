"""CPU regression: real diffusers denoising must retain the requested batch dimension.

Random tiny components need no checkpoint downloads. Only text encoding is stubbed;
request parsing, engine iteration, noise, scheduler, UNet, VAE and PNG saving are real.
"""
import functools
import hashlib
import io
import json

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
from PIL import Image
from sdfx.engine import Engine
from sdfx.protocol import Emitter, JobContext


@pytest.mark.parametrize("sdxl", [False, True], ids=["sd15", "sdxl"])
def test_four_images_are_denoised_together_in_five_iterations(monkeypatch, tmp_path, sdxl):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(1)
        vae = diffusers.AutoencoderKL(in_channels=3, out_channels=3, latent_channels=4,
            down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4,
            block_out_channels=(8, 8, 8, 8), layers_per_block=1, norm_num_groups=4)
        unet = diffusers.UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4,
            down_block_types=("CrossAttnDownBlock2D",), up_block_types=("CrossAttnUpBlock2D",),
            block_out_channels=(8,), layers_per_block=1, norm_num_groups=4,
            cross_attention_dim=8, attention_head_dim=2,
            **(dict(addition_embed_type="text_time", addition_time_embed_dim=2,
                    projection_class_embeddings_input_dim=20) if sdxl else {}))
        common = dict(vae=vae, text_encoder=None, tokenizer=None, unet=unet,
                      scheduler=diffusers.EulerDiscreteScheduler(steps_offset=1))
        if sdxl:
            pipe = diffusers.StableDiffusionXLPipeline(**common, text_encoder_2=None,
                tokenizer_2=None, add_watermarker=False)
        else:
            pipe = diffusers.StableDiffusionPipeline(**common, safety_checker=None,
                feature_extractor=None, requires_safety_checker=False)
        pipe.set_progress_bar_config(disable=True)
        engine = Engine()
        engine.pipe, engine.kind, engine.sdxl = pipe, "sdxl" if sdxl else "sd15", sdxl
        engine.model = {"name": "random-tiny-cpu", "hash": "test-only"}
        engine.base_sched_config = dict(pipe.scheduler.config)

        encoded_batch_sizes = []
        def encode(pos, neg, gp, bs, steps, **kwargs):
            encoded_batch_sizes.append(bs)
            result = {"prompt_embeds": torch.zeros(bs, 4, 8),
                      "negative_prompt_embeds": torch.zeros(bs, 4, 8)}
            if sdxl:
                result.update(pooled_prompt_embeds=torch.zeros(bs, 8),
                              negative_pooled_prompt_embeds=torch.zeros(bs, 8))
            return result
        monkeypatch.setattr(engine, "_encode_scheduled", encode)

        calls = []
        original_call = type(pipe).__call__
        @functools.wraps(original_call)
        def call(self, *args, **kwargs):
            calls.append((kwargs["latents"].shape[0], kwargs["prompt_embeds"].shape[0],
                          [g.initial_seed() for g in kwargs["generator"]]))
            return original_call(self, *args, **kwargs)
        monkeypatch.setattr(type(pipe), "__call__", call)
        unet_batches = []
        handle = unet.register_forward_pre_hook(lambda module, args: unet_batches.append(args[0].shape[0]))
        events = io.BytesIO()
        try:
            result = engine.txt2img({"params": {"seed": 100, "subseed": 200,
                "batch_size": 4, "n_iter": 5, "steps": 2, "sampler_name": "Euler",
                "width": 64, "height": 64, "cfg_scale": 7},
                "output_dir": str(tmp_path), "preview_every": 0}, JobContext(Emitter(events), "batch"))
        finally:
            handle.remove()

        assert encoded_batch_sizes == [4]
        assert calls == [(4, 4, list(range(100 + 4 * i, 104 + 4 * i))) for i in range(5)]
        # Classifier-free guidance doubles each batch of four to eight UNet rows.
        assert unet_batches == [8] * 10
        assert [im["seed"] for im in result["images"]] == list(range(100, 120))
        assert [im["subseed"] for im in result["images"]] == list(range(200, 220))
        assert len({im["path"] for im in result["images"]}) == 20
        pixels = []
        for im in result["images"]:
            with Image.open(im["path"]) as saved:
                assert saved.size == (64, 64)
                pixels.append(hashlib.sha256(saved.tobytes()).hexdigest())
        assert len(set(pixels)) == 20
        progress = [json.loads(line) for line in events.getvalue().splitlines()]
        progress = [ev for ev in progress if ev.get("event") == "progress"]
        assert len(progress) == 10
        assert [ev["image"] for ev in progress] == [i for i in (0, 4, 8, 12, 16) for _ in range(2)]
        assert all(ev["images"] == 20 for ev in progress)
    finally:
        torch.set_num_threads(previous_threads)
