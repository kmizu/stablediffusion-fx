# PNGINFO Injection verification (#98)

Scope: opt-in native AlwaysOn plugin `pnginfo_injection` records generation-time resource Hashes. Existing Hashes JSON is merged and current confirmed facts take precedence. Names, quotes and Unicode round-trip through PNG infotext. Disabled mode omits the injected result/metadata fields and preserves pixels. Internal protocol accepts older results without resource_hashes.

## Source and distribution

Behavior reference: bluelovers/sd-webui-pnginfo-injection revision 93bf780426d3571056c6ef96fb7734943f16e711. No published license file was found. No code, assets or static wildcard mapping table copied. THIRD_PARTY_NOTICES.md records the reference; scripts/third_party.py completed for 145 projects. ADetailer runtime optional package is ultralytics 8.4.66; test detector is separately downloaded Bingsu/adetailer face_yolov8n.pt, not bundled.

## Measured acceptance

- Native MainWindow Generate and saved PNG checks, both SD1.5 animekawa_v10 and SDXL waiIllustriousSDXL_v170: outputs/verify-pnginfo-native-sd15 and outputs/verify-pnginfo-native-sdxl. Saved model hash equals loaded AutoV2 prefix; disabled has no Hashes; decoded RGB exact. SDXL output inspected.
- Native positive and negative wildcard expansion, both models: outputs/verify-pnginfo-wildcard-sd15 and outputs/verify-pnginfo-wildcard-sdxl. Independent file SHA-256 matches saved wildcards_file entries, enabled/off RGB exact. Both images inspected.
- ADetailer actual detector loading, both models: outputs/verify-pnginfo-ad-sd15/pnginfo_ad_sd15.json and outputs/verify-pnginfo-ad-sdxl/pnginfo_ad_sdxl.json. Detector hash 70b640f8f6; independent whole-file digest agrees, disabled omits resource_hashes, RGB exact. SDXL inspected. SD1.5 rerun passes after cache-identity correction.
- Synthetic zero-vector textual inversions loaded through actual tokenizers/encoders on both models: outputs/verify-pnginfo-embed-sd15/pnginfo_embed_sd15.json and outputs/verify-pnginfo-embed-sdxl/pnginfo_embed_sdxl.json. Correct used embedding hash and exact enabled/off RGB. These test provenance, not embedding artistic quality.
- Single-file SD1.5 ControlNet: outputs/verify-pnginfo-cn-sd15/pnginfo_controlnet_sd15.json. Real lllyasviel/control_v11p_sd15_canny weights loaded and reused from cache, independent prefix be713fb941 agrees, enabled/off RGB exact; image inspected.
- Core ResourceHashesSuite: 3 tests. PromptPluginsSuite: 14 tests including exact consumed-file bytes, cached reads and negative wildcards. Resource snapshots reject changed/replaced backing files, omission emits warning. Related resource/ADetailer tests: 14 passed. Resource/LoRA/composable tests: 36 passed.

## Behavior limits

Civitai upload/network identification is not tested. Wildcards_file entries identify local files consumed during expansion; they do not claim upstream bundle identity. Glob/combination expansion can read a pool of files rather than only a selected line. Upstream Template Generated/Grid and Wildcard Prompt WIP are not represented as completed features. Unsupported directory resources are warned and omitted rather than assigned pathname hashes. SDXL ControlNet hardware acceptance is not established here; model/ADetailer/embedding/native wildcard checks cover both architectures.

Loaded model and explicit VAE use their load-time hashes. Embeddings, single-file ControlNet, YOLO detectors and ordinary LoRA adapters retain load-time file identity; a later changed backing file is omitted with a reload warning. Deliberately restoring all filesystem identity fields after byte mutation is outside the stat identity check. Composable LoRA creates a per-job adapter snapshot; its provenance uses that job's loaded paths.

## Regression status

Whole Python passed after ordinary LoRA cache correction (exit 0, outputs/pnginfo-full-python.log). Whole Scala completed 1270 tests with 5 LocalPreviewFilesSuite Windows drive-probe timeouts (other suites passed, one skip). Focused LocalPreviewFilesSuite recheck passed all 22 tests (outputs/pnginfo-local-preview-recheck.log). Assembly passed; target/stablediffusion-nui.jar SHA-256 is 37a36c9ebde47e148ebdeed93c63f6a5b7b971799c053d44db849019b9db2e46. Final native rerun passed both models after all cache corrections: outputs/pnginfo-native-final.log, outputs/verify-pnginfo-final-sd15 and outputs/verify-pnginfo-final-sdxl. Each saved model/positive/negative wildcard hash agrees and enabled/off RGB is exact.
