"""Coverage of the A1111 / reForge extension index ("Available" tab) in stablediffusion-fx.

Every entry of docs/extensions-index.json (snapshot of the official index) gets one status:
  done:<id>   built into stablediffusion-fx (plugin id, or `core:<feature>`)
  todo:<id>   accepted, planned under that plugin id
  na:<reason> does not apply to a standalone desktop app (reason codes below)

Run `python scripts/extensions_matrix.py` to regenerate docs/EXTENSIONS.md.
"""
import json
import os
import sys

NA_REASONS = {
    "webui-css": "styles / JavaScript tweaks of the Gradio web page; the JavaFX UI has its own theme and layout",
    "webui-ux": "Gradio page behaviour (tabs, buttons, dialogs); the desktop UI provides the equivalent natively",
    "webui-api": "adds HTTP routes to the A1111 server for other tools; this app's integration surface is MCP",
    "account": "needs a third-party account / paid API key and posts data to that service",
    "obsolete": "superseded: the feature is in the base app (A1111 >= 1.5) or the extension is deprecated",
    "external-app": "bridge to a specific external application (Photoshop, Krita, ComfyUI, ...)",
    "server-ops": "operating a shared web server (tunnels, TLS, multi-instance, kill buttons)",
}

# index -> status. Index numbers follow docs/extensions-index.json (official index order).
S = {
    0: "done:core:job-queue", 1: "todo:temporal_kit", 2: "todo:core:localization", 3: "todo:batch_checkpoints_prompts",
    4: "todo:kandinsky", 5: "todo:shap_e", 6: "todo:openpose_editor", 7: "todo:segment_anything",
    8: "todo:aesthetic_gradients", 9: "todo:train_dreambooth", 10: "todo:training_picker", 11: "todo:dataset_tag_editor",
    12: "todo:dreamartist", 13: "todo:hypernetwork_training", 14: "todo:custom_diffusion", 15: "todo:smart_process",
    16: "todo:embedding_editor", 17: "todo:model_merge", 18: "todo:model_converter", 19: "done:core:lora",
    20: "todo:model_merge", 21: "todo:embedding_merge", 22: "todo:model_merge", 23: "todo:lora_block_weight",
    24: "done:core:gallery", 25: "done:core:gallery", 26: "todo:inspiration", 27: "todo:inspiration",
    28: "todo:prompt_gallery", 29: "todo:infinity_grid", 30: "todo:core:presets", 31: "todo:core:presets",
    32: "todo:outpaint_canvas", 33: "na:webui-css", 34: "done:core:size-presets", 35: "na:webui-css", 36: "na:webui-css",
    37: "todo:core:localization", 38: "todo:dynamic_prompts", 39: "todo:unprompted", 40: "todo:stylepile",
    41: "todo:core:tag-autocomplete", 42: "todo:novelai_prompt", 43: "todo:tokenizer", 44: "todo:randomize",
    45: "todo:conditioning_highres_fix", 46: "todo:model_keyword", 47: "todo:prompt_generator", 48: "todo:prompt_generator",
    49: "todo:text2prompt", 50: "na:account", 51: "todo:deforum", 52: "todo:gif2gif", 53: "todo:video_loopback",
    54: "todo:seed_travel", 55: "todo:shift_attention", 56: "todo:prompt_travel", 57: "todo:steps_animation",
    58: "na:external-app", 59: "todo:batch_face_swap", 60: "todo:depth_maps", 61: "todo:multi_subject_render",
    62: "todo:depthmap2mask", 63: "todo:remove_background", 64: "todo:pixel_art", 65: "todo:pixel_art", 66: "todo:haku_img",
    67: "todo:asymmetric_tiling", 68: "todo:latent_mirroring", 69: "todo:sonar", 70: "todo:depth_image_io",
    71: "todo:ultimate_sd_upscale", 72: "todo:prompt_fusion", 73: "todo:neutral_prompt", 74: "done:dynamic_thresholding",
    75: "todo:anti_burn", 76: "done:core:controlnet", 77: "todo:latent_couple", 78: "todo:composable_lora",
    79: "na:server-ops", 80: "todo:booru_prompt", 81: "todo:booru_prompt", 82: "todo:booru_prompt", 83: "todo:nsfw_checker",
    84: "todo:prompt_filter", 85: "na:obsolete", 86: "todo:riffusion", 87: "todo:save_intermediate", 88: "todo:grid_numbers",
    89: "todo:hypernetworks", 90: "todo:core:system-info", 91: "todo:openpose_editor", 92: "na:account", 93: "na:account",
    94: "todo:minecraft_schematic", 95: "todo:aesthetic_score", 96: "todo:aesthetic_score", 97: "todo:aesthetic_score",
    98: "todo:clip_interrogator", 99: "todo:cross_attention_viz", 100: "todo:dump_unet", 101: "todo:openpose_editor",
    102: "todo:llul", 103: "todo:cfg_schedule", 104: "todo:ebsynth_utility", 105: "todo:vram_estimator",
    106: "todo:tiled_diffusion", 107: "todo:openpose_editor", 108: "todo:corridor_crawler", 109: "todo:core:panorama-viewer",
    110: "na:account", 111: "todo:remove_background", 112: "na:server-ops", 113: "todo:openpose_editor", 114: "na:webui-css",
    115: "done:core:ui-state", 116: "todo:text2video", 117: "done:core:size-presets", 118: "done:core:mask-canvas",
    119: "todo:regional_prompter", 120: "todo:prompt_translate", 121: "todo:prompt_translate", 122: "todo:abysz_lab",
    123: "todo:negative_prompt_weight", 124: "na:account", 125: "todo:remove_background", 126: "na:obsolete",
    127: "todo:outpaint_canvas", 128: "todo:infinite_zoom", 129: "todo:adetailer", 130: "todo:core:localization",
    131: "todo:core:localization", 132: "todo:core:localization", 133: "todo:core:localization",
    134: "todo:core:localization", 135: "todo:core:localization", 136: "todo:core:localization",
    137: "todo:core:localization", 138: "todo:core:localization", 139: "todo:core:localization",
    140: "todo:core:localization", 141: "todo:core:localization", 142: "todo:core:localization",
    143: "todo:core:localization", 144: "todo:core:localization", 145: "todo:model_organizer",
    146: "todo:model_presets", 147: "todo:cn_animation", 148: "todo:model_converter", 149: "todo:core:prompt-tools",
    150: "todo:model_organizer", 151: "todo:adetailer", 152: "todo:weight_gradient", 153: "todo:one_button_prompt",
    154: "todo:model_organizer", 155: "todo:image_editor", 156: "todo:stable_sr", 157: "done:core:job-queue",
    158: "na:server-ops", 159: "na:webui-api", 160: "na:webui-ux", 161: "todo:lua_scripting", 162: "todo:model_downloader",
    163: "todo:core:styles-editor", 164: "todo:clip_io", 165: "todo:inpaint_anything", 166: "todo:qr_code",
    167: "todo:face_swap", 168: "todo:color_enhance", 169: "na:account", 170: "done:core:gallery", 171: "todo:pixel_art",
    172: "na:webui-css", 173: "na:webui-css", 174: "todo:llm_prompt", 175: "todo:openpose_editor",
    176: "done:core:size-presets", 177: "done:core:gallery", 178: "todo:qr_code", 179: "todo:sadtalker",
    180: "na:external-app", 181: "todo:model_downloader", 182: "done:caption", 183: "todo:loopback_scaler",
    184: "todo:inpaint_background", 185: "na:external-app", 186: "todo:gpu_temperature_guard", 187: "done:core:zoom",
    188: "na:server-ops", 189: "todo:latent_upscale", 190: "done:core:gallery", 191: "todo:prompt_filter",
    192: "todo:animatediff", 193: "na:webui-ux", 194: "todo:adetailer", 195: "todo:seamless_tile",
    196: "todo:face_swap", 197: "todo:style_selector", 198: "todo:prompt_library", 199: "todo:custom_hires_fix",
    200: "todo:refiner", 201: "todo:deoldify", 202: "todo:wav2lip", 203: "todo:core:presets", 204: "todo:lora_control",
    205: "todo:dynamic_prompts", 206: "done:core:size-presets", 207: "na:account", 208: "na:account",
    209: "todo:core:localization", 210: "todo:vectorscope_cc", 211: "todo:prompt_format", 212: "todo:prompt_format",
    213: "todo:fabric", 214: "done:core:close-confirm", 215: "na:account", 216: "todo:model_downloader",
    217: "todo:batch_interrogate", 218: "done:core:gallery", 219: "todo:depth_library", 220: "todo:model_downloader",
    221: "todo:core:presets", 222: "todo:prompt_library", 223: "todo:model_merge", 224: "done:core:png-info",
    225: "todo:mov2mov", 226: "todo:easyphoto", 227: "todo:rich_text", 228: "todo:remove_background",
    229: "todo:booru_prompt", 230: "todo:model_downloader", 231: "todo:nsfw_checker", 232: "todo:negpip",
    233: "todo:cd_tuner", 234: "na:webui-css", 235: "na:webui-api", 236: "todo:core:compile",
    237: "todo:encrypt_image", 238: "todo:nsfw_checker", 239: "todo:lcm", 240: "todo:cads",
    241: "todo:face_swap", 242: "todo:hotshot_xl", 243: "todo:inpaint_difference", 244: "todo:temporal",
    245: "todo:facechain", 246: "todo:style_selector", 247: "todo:panorama_tools", 248: "todo:core:styles-editor",
    249: "todo:semantic_guidance", 250: "done:core:png-info", 251: "todo:person_mask", 252: "todo:style_variables",
    253: "todo:core:api-payload", 254: "todo:adetailer", 255: "todo:lora_queue", 256: "done:kohya_hrfix",
    257: "todo:test_my_prompt", 258: "na:webui-ux", 259: "na:account", 260: "todo:replacer",
    261: "todo:color_correction", 262: "todo:core:presets", 263: "todo:agent_attention", 264: "todo:deepdanbooru",
    265: "todo:deepdanbooru", 266: "todo:ocr", 267: "todo:characteristic_guidance", 268: "todo:embedding_inspector",
    269: "todo:lama_cleaner", 270: "todo:regional_prompter", 271: "todo:model_organizer", 272: "na:account",
    273: "todo:video_extras", 274: "todo:inpaint_background", 275: "todo:daam", 276: "todo:controlnet_preprocess_extras",
    277: "na:account", 278: "todo:rpg_diffusion", 279: "todo:creaprompt", 280: "todo:img2img_workflow",
    281: "na:server-ops", 282: "na:webui-ux", 283: "todo:dataset_tag_editor", 284: "todo:resharpen", 285: "na:webui-ux",
    286: "todo:core:image-compare", 287: "na:webui-ux", 288: "todo:mosaic_outpaint", 289: "na:webui-ux",
    290: "todo:color_grading", 291: "na:webui-ux", 292: "todo:old_photo_restoration", 293: "na:webui-ux",
    294: "todo:core:prompt-tools", 295: "todo:token_downsampling", 296: "done:core:size-presets",
    297: "todo:hires_fix_tweaks", 298: "todo:manga_maker", 299: "todo:model_organizer", 300: "na:server-ops",
    301: "todo:demofusion", 302: "todo:incantations", 303: "todo:old_sd_firstpasser", 304: "todo:image_editor",
    305: "todo:triposr", 306: "na:webui-ux", 307: "todo:core:prompt-tools", 308: "todo:model_merge",
    309: "todo:hardware_info", 310: "done:core:png-info", 311: "na:webui-css", 312: "todo:metadata_utils",
    313: "todo:prompt_expansion", 314: "todo:core:localization", 315: "na:external-app",
    316: "todo:segment_anything", 317: "done:core:size-presets", 318: "todo:scramble_prompts", 319: "todo:tweak_weights",
    320: "todo:core:prevent-interrupt", 321: "todo:waifu2x", 322: "todo:depth_anything", 323: "na:external-app",
    324: "todo:detail_daemon", 325: "todo:masked_content_extras", 326: "todo:core:prompt-tools",
    327: "todo:masked_content_extras", 328: "done:core:theme", 329: "todo:manga_inpainting", 330: "na:webui-api",
    331: "todo:chara_searcher", 332: "todo:fsr_upscale", 333: "todo:core:nsfw-card-blur", 334: "na:webui-ux",
    335: "todo:face_manipulation", 336: "todo:pnginfo_injection", 337: "todo:llm_prompt", 338: "todo:remove_background",
    339: "todo:img2img_hires_fix", 340: "todo:ditail", 341: "todo:live_portrait", 342: "todo:tipo",
    343: "na:webui-ux", 344: "na:account", 345: "done:core:gallery", 346: "na:account", 347: "todo:booru_prompt",
    348: "na:account", 349: "todo:model_downloader", 350: "done:core:size-presets", 351: "todo:style_selector",
    352: "todo:vae_tools",
}


# plugin ids / core features that are implemented: their `todo:` entries are reported as `done:`
IMPLEMENTED = {
    "dynamic_prompts", "randomize", "prompt_format", "prompt_filter", "negative_prompt_weight", "novelai_prompt",
    "style_variables", "test_my_prompt", "scramble_prompts", "tweak_weights", "lora_queue", "batch_checkpoints_prompts",
    "seed_travel", "shift_attention", "one_button_prompt",
    "detail_daemon", "vectorscope_cc", "latent_mirroring", "asymmetric_tiling", "anti_burn", "resharpen",
    "color_grading", "cfg_schedule", "cads",
    "ultimate_sd_upscale", "adetailer", "lora_block_weight", "model_merge", "model_converter",
    "pixel_art", "color_enhance", "fsr_upscale", "person_mask", "remove_background", "depthmap2mask", "inpaint_background",
}


def status(i: int) -> str:
    st = S[i]
    kind, _, target = st.partition(":")
    return f"done:{target}" if kind == "todo" and target in IMPLEMENTED else st


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    idx = json.load(open(os.path.join(root, "docs", "extensions-index.json"), encoding="utf-8"))
    missing = [i for i in range(len(idx)) if i not in S]
    if missing:
        sys.exit(f"unclassified index entries: {missing}")
    rows, counts = [], {"done": 0, "todo": 0, "na": 0}
    for i, e in enumerate(idx):
        st = status(i)
        kind, _, target = st.partition(":")
        counts[kind] += 1
        note = NA_REASONS.get(target, "") if kind == "na" else ""
        rows.append(f"| {i} | [{e['name']}]({e['url']}) | {e['stars']} | {kind} | `{target}` | {note} |")
    md = ["# A1111 / reForge extension index coverage", "",
          f"Snapshot of the official extension index ({len(idx)} extensions), the list reForge's Extensions > Available "
          "tab installs from. Status per extension: `done` = built in, `todo` = accepted and planned, `na` = does not "
          "apply to a standalone desktop app.", "",
          f"Totals: done {counts['done']}, todo {counts['todo']}, na {counts['na']}.", "",
          "| # | Extension | Stars | Status | stablediffusion-fx | Why not applicable |", "|---|---|---|---|---|---|"] + rows
    md += ["", "Reason codes for `na`:", ""] + [f"- `{k}`: {v}" for k, v in NA_REASONS.items()]
    open(os.path.join(root, "docs", "EXTENSIONS.md"), "w", encoding="utf-8", newline="\n").write("\n".join(md) + "\n")
    status_json = [{"index": i, "name": e["name"], "url": e["url"], "status": status(i)} for i, e in enumerate(idx)]
    with open(os.path.join(root, "docs", "extensions-status.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(status_json, fh, ensure_ascii=False, indent=0)
    print(counts)


if __name__ == "__main__":
    main()
