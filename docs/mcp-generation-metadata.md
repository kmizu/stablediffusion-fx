# MCP PNG generation metadata inspection

`inspect_generation_metadata` reads PNG `parameters` from an allowed local
path or a base64 PNG and proposes `txt2img` arguments plus an `always_on` object.
It does not execute a script, select a checkpoint, load models, or generate an
image. The caller must review the response and submit a separate generation.
Input paths use the MCP image allowlist; UNC/network paths and protected PNGs
are rejected. PNG bytes are limited to 16 MiB; total text and ICC-profile
expansion is limited to 1 MiB, including compressed PNG text variants.

| GUI restore capability | MCP inspection |
| --- | --- |
| Positive/negative prompt, steps, sampler/scheduler, CFG, seed, size, batch, clip skip | Typed txt2img candidates from `Infotext.toParams`, subject to MCP limits |
| Native Hires scale, size, upscaler, steps, denoising, prompts, sampler, scheduler and CFG | Typed Hires candidates |
| Conditioning Highres Fix, img2img Hires Fix, Token Downsampling, Model Keyword replay disable | Explicit typed `always_on` candidates from the same GUI replay helpers |
| Checkpoint / VAE choice | Not resolved or loaded; check catalog and metadata manually |
| Variation seed/strength, sampler eta and manual LoRA selections | Typed candidates; adapter names are validated against the live catalog at generation |
| Inpaint input/mask and ControlNet image references | Not recoverable from PNG infotext alone; caller supplies them. Typed ControlNet units accept only catalog models and allowed images |
| Hires seed tweaks, Composable LoRA and grid numbers | Explicit `always_on` candidates when their recognized metadata is present |
| FreeU v2, HyperTile, PAG | Bounded candidates parsed from native plugin-emitted fields; these are additional MCP interpretations, not existing GUI restore functions. Incomplete or unrecognized metadata stays unapplied |
| Latent upscale and other plugin controls | Not yet returned as replay candidates |
| Arbitrary infotext fields and unrecognized extensions | Returned as `unapplied_fields` only; no plugin/script dispatch |

This is finite metadata interpretation, not a guarantee of pixel-identical
generation: checkpoint weights, embeddings, adapter files, runtime versions,
original inputs and random state may differ. External stdio and HTTP client
acceptance must be tested separately from direct tool calls.

## Desktop features not fully exposed through MCP on base d837505

| Desktop feature | MCP state |
| --- | --- |
| PNG Info inspect and restore into an editable form | This tool proposes parameters; it does not manipulate the GUI or automatically run them |
| Model Keyword mappings editor | Lookup is exposed; mapping save/delete and catalog download are not exposed by that lookup tool |
| Embedding Inspector and Embedding Merge workbenches | No dedicated MCP workbench API; generation prompt syntax alone is not equivalent |
| Image Comparison, gallery selection and Styles editor | Recent image listing/image retrieval exist; comparison UI, selection and style CRUD are not exposed |
| Metadata Utils, Temporal Lab and Extras | Their desktop workbenches are not exposed as equivalent standalone MCP tools |
| Session/password controls for protected PNG | MCP rejects protected image input; it does not use the desktop password session |
| Model merge / converter scripts | Explicitly GUI-only in `McpTools.GuiOnlyScripts` |
| Tokenizer, System Info and VRAM Estimator tabs | No dedicated MCP equivalents in this base tool list |

The separately developed common AlwaysOn API may change this matrix after
integration; it does not itself make every desktop workbench remotely available.

Focused JDK21 direct-tool tests cover local/base64 metadata equivalence, typed
values, allowlist and UNC refusal, unknown arguments, malicious model text,
MCP bounds and compressed-text expansion limits. These tests do not establish
external transport or real-model replay; those require separate acceptance.

Generation accepts `subseed`, `subseed_strength`, `eta`, structured
`manual_loras`, independent Hires prompt/negative/CFG/sampler/scheduler, and
typed `controlnet` units. Existing `loras` remains the prompt-tag interface.
ControlNet model names and preprocessors are checked against the catalog;
images use the normal MCP allowlist/base64 policy. No raw hooks are exposed.
The focused JDK21 suites passed 36 tests, including fake-worker receipt and
ControlNet temporary-file cleanup on success and validation failure. This is
not an external SDK or real-model acceptance result.
