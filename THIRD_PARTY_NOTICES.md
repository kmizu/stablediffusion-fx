# Third-party notices

stablediffusion-fx is licensed under the GNU Affero General Public License v3.0 (see LICENSE), because it ports
code from AGPL-3.0 (AUTOMATIC1111 web UI, Forge, reForge) and GPL-3.0 (ComfyUI, Fooocus) projects.

How each project is used:

- **ported**: code or algorithms were translated from the project; its license applies to those parts.
- **reimplemented**: the feature was written from scratch from the project's documented behaviour; no code was
  copied. The license is listed for attribution.
- **dependency**: installed separately (pip / Maven), not redistributed in this repository.
- **data**: data files included in this repository.

Projects without a published license file were not copied from; their features were re-implemented from the
description of their behaviour only.

| Project | Use | License | What |
|---|---|---|---|
| [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) | ported | AGPL-3.0 ([text](licenses/AUTOMATIC1111__stable-diffusion-webui.txt)) | infotext format, built-in scripts (X/Y/Z, prompt matrix, loopback, SD upscale, outpainting, img2img alternative, custom code, postprocessing), styles.csv, soft inpainting |
| [Panchovix/stable-diffusion-webui-reForge](https://github.com/Panchovix/stable-diffusion-webui-reForge) | ported | AGPL-3.0 ([text](licenses/Panchovix__stable-diffusion-webui-reForge.txt)) | reference implementation; built-in extensions (APG, RescaleCFG, advanced model sampling, Mahiro, random resolutions), sampler / schedule tables |
| [lllyasviel/stable-diffusion-webui-forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) | ported | AGPL-3.0 ([text](licenses/lllyasviel__stable-diffusion-webui-forge.txt)) | forge built-in extensions (FreeU, Dynamic Thresholding, HyperTile, SAG, Kohya HRFix, MultiDiffusion, Latent Modifier, Never OOM, IP-Adapter, ControlLLLite, PhotoMaker, Fooocus inpaint, SVD, Z123, preprocessors) |
| [comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI) | ported | GPL-3.0 ([text](licenses/comfyanonymous__ComfyUI.txt)) | sigma schedules, APG / Mahiro / RescaleCFG guidance math, Deep Shrink, FreeU |
| [crowsonkb/k-diffusion](https://github.com/crowsonkb/k-diffusion) | ported | MIT ([text](licenses/crowsonkb__k-diffusion.txt)) | Karras / exponential / polyexponential sigma schedules |
| [mcmonkeyprojects/sd-dynamic-thresholding](https://github.com/mcmonkeyprojects/sd-dynamic-thresholding) | ported | MIT ([text](licenses/mcmonkeyprojects__sd-dynamic-thresholding.txt)) | Dynamic Thresholding (CFG-Fix) math |
| [tfernd/HyperTile](https://github.com/tfernd/HyperTile) | ported | MIT ([text](licenses/tfernd__HyperTile.txt)) | HyperTile attention tiling |
| [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) | ported | Apache-2.0 ([text](licenses/kohya-ss__sd-scripts.txt)) | ControlNet-LLLite module layout |
| [lllyasviel/Fooocus](https://github.com/lllyasviel/Fooocus) | ported | GPL-3.0 ([text](licenses/lllyasviel__Fooocus.txt)) | Fooocus inpaint patch / head application |
| [threestudio-project/threestudio](https://github.com/threestudio-project/threestudio) | reimplemented | Apache-2.0 ([text](licenses/threestudio-project__threestudio.txt)) | Zero123 diffusers layout and sampling |
| [TencentARC/PhotoMaker](https://github.com/TencentARC/PhotoMaker) | dependency | NOASSERTION ([text](licenses/TencentARC__PhotoMaker.txt)) | PhotoMaker pipeline (optional Python package) |
| [huggingface/diffusers](https://github.com/huggingface/diffusers) | dependency | Apache-2.0 ([text](licenses/huggingface__diffusers.txt)) | inference pipelines |
| [huggingface/transformers](https://github.com/huggingface/transformers) | dependency | Apache-2.0 ([text](licenses/huggingface__transformers.txt)) | text / image encoders, BLIP, Marigold support |
| [huggingface/peft](https://github.com/huggingface/peft) | dependency | Apache-2.0 ([text](licenses/huggingface__peft.txt)) | LoRA adapters |
| [huggingface/accelerate](https://github.com/huggingface/accelerate) | dependency | Apache-2.0 ([text](licenses/huggingface__accelerate.txt)) | model offloading |
| [huggingface/safetensors](https://github.com/huggingface/safetensors) | dependency | Apache-2.0 ([text](licenses/huggingface__safetensors.txt)) | weights I/O |
| [chaiNNer-org/spandrel](https://github.com/chaiNNer-org/spandrel) | dependency | MIT ([text](licenses/chaiNNer-org__spandrel.txt)) | upscaler / face restoration model loading |
| [pytorch/pytorch](https://github.com/pytorch/pytorch) | dependency | NOASSERTION ([text](licenses/pytorch__pytorch.txt)) | tensor runtime |
| [opencv/opencv-python](https://github.com/opencv/opencv-python) | dependency | MIT ([text](licenses/opencv__opencv-python.txt)) | image processing (canny, face detection) |
| [python-pillow/Pillow](https://github.com/python-pillow/Pillow) | dependency | NOASSERTION ([text](licenses/python-pillow__Pillow.txt)) | images |
| [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) | dependency | MIT ([text](licenses/microsoft__onnxruntime.txt)) | WD14 tagger |
| [openjdk/jfx](https://github.com/openjdk/jfx) | dependency | GPL-2.0 ([text](licenses/openjdk__jfx.txt)) | JavaFX UI toolkit |
| [com-lihaoyi/upickle](https://github.com/com-lihaoyi/upickle) | dependency | MIT ([text](licenses/com-lihaoyi__upickle.txt)) | JSON |
| [scala/scala3](https://github.com/scala/scala3) | dependency | Apache-2.0 ([text](licenses/scala__scala3.txt)) | Scala runtime |
| [AUTOMATIC1111/stable-diffusion-webui-extensions](https://github.com/AUTOMATIC1111/stable-diffusion-webui-extensions) | data | no license file published | extension index snapshot (docs/extensions-index.json) |
| [AIrjen/OneButtonPrompt](https://github.com/AIrjen/OneButtonPrompt) | reimplemented | GPL-3.0 ([text](licenses/AIrjen__OneButtonPrompt.txt)) | extension rewritten as One Button Prompt -> `one_button_prompt` |
| [AUTOMATIC1111/stable-diffusion-webui-wildcards](https://github.com/AUTOMATIC1111/stable-diffusion-webui-wildcards) | reimplemented | no license file published | extension rewritten as Wildcards -> `dynamic_prompts` |
| [AlUlkesh/sd_delete_button](https://github.com/AlUlkesh/sd_delete_button) | reimplemented | Unlicense ([text](licenses/AlUlkesh__sd_delete_button.txt)) | extension rewritten as SD Delete Button -> `core:gallery` |
| [AlUlkesh/stable-diffusion-webui-images-browser](https://github.com/AlUlkesh/stable-diffusion-webui-images-browser) | reimplemented | no license file published | extension rewritten as Image browser -> `core:gallery` |
| [ArtVentureX/sd-webui-agent-scheduler](https://github.com/ArtVentureX/sd-webui-agent-scheduler) | reimplemented | no license file published | extension rewritten as Agent Scheduler -> `core:job-queue` |
| [Extraltodeus/test_my_prompt](https://github.com/Extraltodeus/test_my_prompt) | reimplemented | no license file published | extension rewritten as Test my prompt! -> `test_my_prompt` |
| [Haoming02/sd-webui-prompt-format](https://github.com/Haoming02/sd-webui-prompt-format) | reimplemented | MIT ([text](licenses/Haoming02__sd-webui-prompt-format.txt)) | extension rewritten as Prompt Format - by Haoming02 -> `prompt_format` |
| [Kryptortio/SDAtom-WebUi-client-queue-ext](https://github.com/Kryptortio/SDAtom-WebUi-client-queue-ext) | reimplemented | no license file published | extension rewritten as Web UI client queue -> `core:job-queue` |
| [LEv145/--sd-webui-ar-plus](https://github.com/LEv145/--sd-webui-ar-plus) | reimplemented | no license file published | extension rewritten as Aspect Ratio selector plus -> `core:size-presets` |
| [LucasMali/sd-history-slider](https://github.com/LucasMali/sd-history-slider) | reimplemented | Apache-2.0 ([text](licenses/LucasMali__sd-history-slider.txt)) | extension rewritten as sd-history-slider -> `core:gallery` |
| [MINENEMA/sd-webui-quickrecents](https://github.com/MINENEMA/sd-webui-quickrecents) | reimplemented | no license file published | extension rewritten as sd-webui-quickrecents -> `core:gallery` |
| [MarcusNyne/sd-scramble-prompts-m9](https://github.com/MarcusNyne/sd-scramble-prompts-m9) | reimplemented | GPL-3.0 ([text](licenses/MarcusNyne__sd-scramble-prompts-m9.txt)) | extension rewritten as ScramblePrompts [m9] -> `scramble_prompts` |
| [MarcusNyne/sd-tweak-weights-m9](https://github.com/MarcusNyne/sd-tweak-weights-m9) | reimplemented | GPL-3.0 ([text](licenses/MarcusNyne__sd-tweak-weights-m9.txt)) | extension rewritten as TweakWeights [m9] -> `tweak_weights` |
| [Mikubill/sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet) | reimplemented | GPL-3.0 ([text](licenses/Mikubill__sd-webui-controlnet.txt)) | extension rewritten as sd-webui-controlnet -> `core:controlnet` |
| [NoCrypt/sd-fast-pnginfo](https://github.com/NoCrypt/sd-fast-pnginfo) | reimplemented | no license file published | extension rewritten as Fast PNG Info -> `core:png-info` |
| [SirVeggie/extension-style-vars](https://github.com/SirVeggie/extension-style-vars) | reimplemented | no license file published | extension rewritten as Style variables -> `style_variables` |
| [WildBanjos/DiffusionDefender](https://github.com/WildBanjos/DiffusionDefender) | reimplemented | MIT ([text](licenses/WildBanjos__DiffusionDefender.txt)) | extension rewritten as Diffusion Defender -> `prompt_filter` |
| [Yinzo/sd-webui-Lora-queue-helper](https://github.com/Yinzo/sd-webui-Lora-queue-helper) | reimplemented | MIT ([text](licenses/Yinzo__sd-webui-Lora-queue-helper.txt)) | extension rewritten as Lora Queue Helper -> `lora_queue` |
| [adieyal/sd-dynamic-prompts](https://github.com/adieyal/sd-dynamic-prompts) | reimplemented | MIT ([text](licenses/adieyal__sd-dynamic-prompts.txt)) | extension rewritten as Dynamic Prompts -> `dynamic_prompts` |
| [alemelis/sd-webui-ar](https://github.com/alemelis/sd-webui-ar) | reimplemented | no license file published | extension rewritten as Aspect Ratio selector -> `core:size-presets` |
| [altoiddealer/--sd-webui-ar-plusplus](https://github.com/altoiddealer/--sd-webui-ar-plusplus) | reimplemented | no license file published | extension rewritten as Aspect Ratio and Resolution Buttons (ar-plusplus) -> `core:size-presets` |
| [animerl/novelai-2-local-prompt](https://github.com/animerl/novelai-2-local-prompt) | reimplemented | Unlicense ([text](licenses/animerl__novelai-2-local-prompt.txt)) | extension rewritten as novelai-2-local-prompt -> `novelai_prompt` |
| [bit9labs/sd-masonry](https://github.com/bit9labs/sd-masonry) | reimplemented | AGPL-3.0 ([text](licenses/bit9labs__sd-masonry.txt)) | extension rewritten as SD Masonry Gallery -> `core:gallery` |
| [bit9labs/sd-ratio-lock](https://github.com/bit9labs/sd-ratio-lock) | reimplemented | AGPL-3.0 ([text](licenses/bit9labs__sd-ratio-lock.txt)) | extension rewritten as SD Ratio Lock -> `core:size-presets` |
| [bluelovers/sd-webui-pnginfo-beautify](https://github.com/bluelovers/sd-webui-pnginfo-beautify) | reimplemented | no license file published | extension rewritten as PNGINFO Beautify -> `core:png-info` |
| [gutris1/sd-simple-dimension-preset](https://github.com/gutris1/sd-simple-dimension-preset) | reimplemented | MIT ([text](licenses/gutris1__sd-simple-dimension-preset.txt)) | extension rewritten as SD Simple Dimension Preset -> `core:size-presets` |
| [h43lb1t0/SD-WebUI-BatchCheckpointPrompt](https://github.com/h43lb1t0/SD-WebUI-BatchCheckpointPrompt) | reimplemented | MIT ([text](licenses/h43lb1t0__SD-WebUI-BatchCheckpointPrompt.txt)) | extension rewritten as Batch Checkpoints and Prompts -> `batch_checkpoints_prompts` |
| [ilian6806/stable-diffusion-webui-state](https://github.com/ilian6806/stable-diffusion-webui-state) | reimplemented | MIT ([text](licenses/ilian6806__stable-diffusion-webui-state.txt)) | extension rewritten as stable-diffusion-webui-state -> `core:ui-state` |
| [innightwolfsleep/stable-diffusion-webui-randomize](https://github.com/innightwolfsleep/stable-diffusion-webui-randomize) | reimplemented | MIT ([text](licenses/innightwolfsleep__stable-diffusion-webui-randomize.txt)) | extension rewritten as Randomize -> `randomize` |
| [kohya-ss/sd-webui-additional-networks](https://github.com/kohya-ss/sd-webui-additional-networks) | reimplemented | AGPL-3.0 ([text](licenses/kohya-ss__sd-webui-additional-networks.txt)) | extension rewritten as Kohya-ss Additional Networks -> `core:lora` |
| [light-and-ray/sd-webui-toggle-dark-light](https://github.com/light-and-ray/sd-webui-toggle-dark-light) | reimplemented | no license file published | extension rewritten as Toggle light dark -> `core:theme` |
| [luminouspear/sd-webui-modal-info](https://github.com/luminouspear/sd-webui-modal-info) | reimplemented | MIT ([text](licenses/luminouspear__sd-webui-modal-info.txt)) | extension rewritten as SD Web UI Modal Info -> `core:png-info` |
| [muerrilla/stable-diffusion-NPW](https://github.com/muerrilla/stable-diffusion-NPW) | reimplemented | Unlicense ([text](licenses/muerrilla__stable-diffusion-NPW.txt)) | extension rewritten as Negative Prompt Weight -> `negative_prompt_weight` |
| [namkazt/sd-webui-prompt-history](https://github.com/namkazt/sd-webui-prompt-history) | reimplemented | MIT ([text](licenses/namkazt__sd-webui-prompt-history.txt)) | extension rewritten as sd-webui-prompt_history -> `core:gallery` |
| [picobyte/stable-diffusion-webui-wd14-tagger](https://github.com/picobyte/stable-diffusion-webui-wd14-tagger) | reimplemented | no license file published | extension rewritten as WD 1.4 Tagger -> `caption` |
| [richrobber2/canvas-zoom](https://github.com/richrobber2/canvas-zoom) | reimplemented | no license file published | extension rewritten as Canvas Zoom -> `core:mask-canvas` |
| [thomasasfk/sd-webui-aspect-ratio-helper](https://github.com/thomasasfk/sd-webui-aspect-ratio-helper) | reimplemented | no license file published | extension rewritten as Aspect Ratio Helper -> `core:size-presets` |
| [uwidev/sd_extension-prompt_formatter](https://github.com/uwidev/sd_extension-prompt_formatter) | reimplemented | AGPL-3.0 ([text](licenses/uwidev__sd_extension-prompt_formatter.txt)) | extension rewritten as Prompt Formatter - by uwidev -> `prompt_format` |
| [viyiviyi/prompts-filter](https://github.com/viyiviyi/prompts-filter) | reimplemented | no license file published | extension rewritten as prompts filter -> `prompt_filter` |
| [viyiviyi/stable-diffusion-webui-zoomimage](https://github.com/viyiviyi/stable-diffusion-webui-zoomimage) | reimplemented | AGPL-3.0 ([text](licenses/viyiviyi__stable-diffusion-webui-zoomimage.txt)) | extension rewritten as zoom-image -> `core:zoom` |
| [w-e-w/sdwebui-close-confirmation-dialogue](https://github.com/w-e-w/sdwebui-close-confirmation-dialogue) | reimplemented | MIT ([text](licenses/w-e-w__sdwebui-close-confirmation-dialogue.txt)) | extension rewritten as Webpage close confirmation dialogue -> `core:close-confirm` |
| [wcde/sd-webui-kohya-hiresfix](https://github.com/wcde/sd-webui-kohya-hiresfix) | reimplemented | no license file published | extension rewritten as Deep Shrink Hires.fix -> `kohya_hrfix` |
| [xhoxye/sd-webui-ar_xhox](https://github.com/xhoxye/sd-webui-ar_xhox) | reimplemented | no license file published | extension rewritten as Aspect Ratio Resolutions selector-xhox -> `core:size-presets` |
| [yownas/seed_travel](https://github.com/yownas/seed_travel) | reimplemented | NOASSERTION ([text](licenses/yownas__seed_travel.txt)) | extension rewritten as seed travel -> `seed_travel` |
| [yownas/shift-attention](https://github.com/yownas/shift-attention) | reimplemented | NOASSERTION ([text](licenses/yownas__shift-attention.txt)) | extension rewritten as shift-attention -> `shift_attention` |
| [zanllp/sd-webui-infinite-image-browsing](https://github.com/zanllp/sd-webui-infinite-image-browsing) | reimplemented | MIT ([text](licenses/zanllp__sd-webui-infinite-image-browsing.txt)) | extension rewritten as Infinite image browsing -> `core:gallery` |
