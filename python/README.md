# sdfx engine (Python)

Inference engine for stablediffusion-fx. The Scala/JavaFX app starts it as `python -m sdfx` (cwd = this
directory) and talks JSON Lines over stdin/stdout; the contract is
`core/src/main/scala/sdfx/core/protocol/Protocol.scala`. Only protocol lines are written to stdout;
logs and library output go to stderr.

```
python -m sdfx [--dry-run] [--log-level INFO] [--dry-run-delay-ms N]
```

`--dry-run` never imports torch/diffusers. It writes placeholder PNGs (deterministic per seed) and
emits the same progress/preview events. Use it for UI work and tests.

## Install (real engine)

```
python -m venv .venv
.venv/Scripts/pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/pip install -r requirements.txt
```

`from_single_file` needs the base model configs and tokenizers (`stable-diffusion-v1-5/stable-diffusion-v1-5`,
`stabilityai/stable-diffusion-xl-base-1.0`) from the Hugging Face cache. They are fetched on first load.
After that you can run offline with `HF_HUB_OFFLINE=1`.

## Layout

| module | role |
|---|---|
| `server.py` | stdin reader thread (ping / interrupt / shutdown), single worker queue, error mapping |
| `protocol.py` | envelopes, locked emitter, `JobContext` (progress / preview / log / interrupt) |
| `params.py` | dataclasses mirroring the Scala params, lenient parsing, `validate`, hires target size |
| `models.py` | recursive scans, checkpoint kind from the safetensors header, A1111 short hash (cached in `.cache/`) |
| `schedulers.py`, `sigmas.py` | A1111 sampler/schedule names to diffusers; numpy ports of the A1111 sigma schedules |
| `prompt.py` | A1111 attention syntax, BREAK, 75-token chunks, emphasis, clip skip, SD1.x/SDXL encoders |
| `engine.py`, `hires.py`, `inpaint.py`, `controlnet.py`, `upscalers.py`, `hooks.py`, `preview.py` | real engine |
| `dryrun.py` | fake engine |

## Tests

```
python -m pip install -r requirements-dev.txt
python -m pytest
```

The tests don't need torch or diffusers. Tests that do need them are skipped when they're missing.
