# v0.03 source and notice audit

Checked 2026-10-27 against the actual open issues #23, #24 and #141 and pinned
upstream README/LICENSE files. This audit does not accept features, change the
extension ledger, close issues or substitute for native/real-model checks.

## New notice entries

| Scope | Pinned source | License and archived original SHA-256 |
|---|---|---|
| #23 named generation states | [SenshiSentou/sd-webui-state-manager, 2af62ecd5287438bbc309f7e840d34aa5c8e3cb6](https://github.com/SenshiSentou/sd-webui-state-manager/tree/2af62ecd5287438bbc309f7e840d34aa5c8e3cb6) | MIT; copyright 2023 Patrick Boelens (bluetomato); `3fe9f2a7f539d60ed0e8edaab5f9645c0ee144f9df7f3cf29296153fc643bc4e` |
| #24 local CSV tag completion | [DominikDoom/a1111-sd-webui-tagcomplete, 4170882f90b47be130a0ff9314f663c230b9153d](https://github.com/DominikDoom/a1111-sd-webui-tagcomplete/tree/4170882f90b47be130a0ff9314f663c230b9153d) | MIT; copyright 2022 Dominik Reh; `c1532c29734187eddd3948321379fa95ba79191bc3cb41d2787d7dff6e5913b5` |

The original LICENSE bytes are archived under `licenses/` and pinned in
`scripts/third_party.py`. Both features are independently implemented; no
upstream code, screenshots, icons or tag datasets are bundled. The MIT
copyright and permission texts are retained in the distribution notices.

The state-manager README describes whole-state capture including quicksettings,
generation and script/accordion controls, manual saves, an inspector, and
automatic history with thumbnails. v0.03 covers manual named settings,
save/list/inspect/restore/delete and typed native controls; history/autosave,
thumbnails, browser IndexedDB and upstream import/storage compatibility are
excluded. Init-image and mask pixels are not archived. The existing native
selected-field presets remain a separate feature and data store.

The tag-completion README describes keyboard completion from local tag CSVs,
alongside extra networks, wildcards, chants, translations and previews. v0.03
covers local CSV tag/alias completion in positive/negative prompt editors;
extra-network names/trigger words, wildcard trees, chants, translation files,
network previews and upstream configuration parity are excluded. User-selected
CSV data is not copied into the repository or release. Pinned README SHA-256:
state-manager `5ade6613e5e84bcdff0c3ea2d98655bb6b14d0f3dd78ec5b8c5970b276784e65`;
tag-completion `eae4ec82917aa27a2fd3766aafea007fc79beebab9959cb065eb3972c7a775c3`.

## #141 WD PNG metadata

The child issue explicitly requires per-image effective prompts, generation
facts and WD identity inside PNGs, existing PNG Info/Restore compatibility
after restart, and metadata before owned publication. Other interrogators,
LoRA/ControlNet/scripts remain under #140. The metadata implementation adds
no copied upstream code, asset or dependency; A1111 infotext attribution and
the existing Python image/runtime dependencies already have notice entries.

The existing [batch-interrogator MIT license at ed91cec663bcddd7ca5514264a61483c7d80f8d2](https://github.com/Alvi-alvarez/sd-Img2img-batch-interrogator/blob/ed91cec663bcddd7ca5514264a61483c7d80f8d2/LICENSE)
was fetched again and matches the archived bytes exactly (SHA-256
`73ce7c491c7d76436e5594ee337689d37d57d7c50b8aa70b1d1347ed498502f6`).
The [WD model card at 7f6b584d0bd3f55c4531f14ba3d4761b2bccdc0f](https://huggingface.co/SmilingWolf/wd-vit-tagger-v3/blob/7f6b584d0bd3f55c4531f14ba3d4761b2bccdc0f/README.md)
still declares Apache-2.0. WD weights and selected-tags data remain externally
supplied and excluded from the release; this audit does not authorize a new
redistribution of those files or claim independent dataset-rights verification.

## Acceptance still required

The issue requirements retain native UI integration, boundary/regression
checks, real SD1.5/SDXL generation where applicable, visual inspection and
recording measured results. #141 specifically needs fresh-session PNG
Info/Restore evidence. Notices alone do not satisfy those checks. Only after
the release coordinator accepts results should the corresponding ledger and
issue statuses be changed.
