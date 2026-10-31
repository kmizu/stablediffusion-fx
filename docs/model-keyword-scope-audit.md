# Model Keyword #85: scope and acceptance audit

Status: implementation and finite acceptance evidence are recorded below;
independent review reported no concrete blocker. The isolated worktree includes
accepted #47, #16 and #87 through main `5934d8183bb838839e31e0af331c120f8387113f`.
Pre-integration backups and stashes remain available. Evidence for the current
merged source includes focused Python CPU and Scala plugin/engine/MCP/UI tests,
plus native SD1.5/SDXL MainWindow and real MCP/mode checks after #87 integration.
Fresh package and CPU regression results are recorded in verification; full
Scala initially failed and only affected suites were rerun successfully. Earlier native/package runs
and guarded GPU diagnostics establish only the source versions and conditions
recorded in `model-keyword-verification.md`. The final-checks document contains
the remaining commands and guard evidence criteria.

## Source scope

Reference: [upstream README at c3d0833](https://github.com/mix1009/model-keyword/blob/c3d0833a9434bb41ecd7ea715b5abd5e6b43a5af/README.md)
and [its script](https://github.com/mix1009/model-keyword/blob/c3d0833a9434bb41ecd7ea715b5abd5e6b43a5af/scripts/model_keyword.py).

| Upstream behavior | Current native state and evidence | Remaining limit |
| --- | --- | --- |
| Always-on in txt2img/img2img; enabled by default | Native AlwaysOn plugin and per-tab controls; first-use default matches upstream; persisted explicit false is retained. MCP sync/async/script generation selects this plugin alone from registry values, with per-request `model_keyword` boolean override. Focused Scala plus mounted SD1.5/SDXL first-use and MCP default/opt-out passed after #87 integration. | Fresh package remains. MCP uses registry settings, not unsaved live desktop per-tab form state. |
| Checkpoint old-hash lookup and collision name similarity | 64 KiB from offset 1 MiB; ties choose later entry. Pinned checkpoint table and custom table lookup, tests for collisions and edits. | Checkpoint database row-by-row provenance is not independently established. |
| All, random, iterate or individual checkpoint keywords; four placements | Planner reserves per-image choices and emits effective prompts and source metadata. CPU and pre-#47 SD15/SDXL MainWindow checks exist. | Full-script/extension combinations cannot be inferred from a finite run. |
| Single selected LoRA with weight, individual/all/random/iterate keywords | Catalog selection, insertion, per-image planning and first-image batch adapter activation. The pinned upstream LoRA catalog now ships offline; a user catalog takes precedence. | Fresh package verification remains. Nonstandard extra-network kinds and unsupported adapter formats fail explicitly. |
| Select TI, choose TI/model ordering | Embedding catalog control and prompt composition; native real-model tests used labeled synthetic SD15/SDXL TI fixtures. | Fixtures establish loading/path behavior, not all trained TI conventions. |
| Check, save, delete custom checkpoint and LoRA mappings | Native mappings tab reports file/hash/source and handles atomic edits with backups. LoRA-user mappings are distinct from downloaded catalog. | Non-cooperating external editors are not proven safe. |
| `GET /model_keyword/get_keywords` | Catalog-restricted read-only MCP `model_keyword_lookup` uses the worker's `check` action and returns hash/keywords/source. Focused MCP tests passed after #87 integration. | This is a native MCP equivalent, not the A1111 HTTP URL; real package acceptance remains. |
| Persist controls and restore outputs | Plugin settings participate in native settings; raw effective prompts and replay metadata are stored; pre-#47 replay RGB comparisons passed. | Verify first-use enabled default, old persisted false and PNG restore do not rewrite twice after #47 merge. |

## Data and redistribution

The upstream repository declares MIT; the exact license is included in
`licenses/mix1009__model-keyword.txt` and registered in third-party notices.
[GitHub contribution terms](https://docs.github.com/en/site-policy/github-terms/github-terms-of-service#6-contributions-under-repository-license)
support the grant contributors purport to make; this does not independently
prove rights to all externally sourced content. [Upstream issue #24](https://github.com/mix1009/model-keyword/issues/24)
describes a Civitai-JSON-derived submission and maintainer API/cron updates.
[Civitai terms source](https://github.com/civitai/civitai/blob/292a6600234dce01167164f23b5a47aabe68f61a/src/static-content/tos.md)
distinguishes service access, service materials and user content with applicable
poster/model licenses. Neither these terms nor the issue establish that all rows
may or may not be redistributed. The LoRA corpus includes descriptions, not
only terse triggers; treating uncertainty as a blanket prohibition or as complete
clearance would both go beyond the evidence.

The pinned checkpoint table ships as 373,667 exact bytes, SHA256
`9e35fc373c397a1c288e8f6d10cd12f244b20ab84b831b410f261eabf8dcacdf`;
its complete row provenance is also not individually verified. The pinned LoRA
table (31,395,702 bytes, SHA256
`b20edcfd8f121c6a5ce9012e4be7ae587a4a66b85f2cd0328768523df78db5ec`)
is bundled unchanged with the upstream MIT notice. The pinned upstream tree
contains no separate license or notice for this file, and the file header only
warns about edits being overwritten. This is evidence of the provider's stated
distribution terms; it does not independently establish rights for every
third-party record. A user-confirmed download from the pinned upstream URL
checks exact size/hash and does not overwrite; an actual local download and
286,212-entry parse succeeded. Civitai hash-lookup primitive has synthetic callback tests only;
it is not an exposed network transport or substitute for the offline corpus.
The packaged catalog still needs verification, and individual source-data rights
have not been independently established.

## Cross-feature and verification boundaries

- Ordinary keyword generation, Hires replay, img2img/inpaint, protected session,
  cancellation/recovery and extracted package were exercised with SD15 and SDXL
  before #47 and before the latest edits. Actual image inspections are recorded
  in `model-keyword-verification.md`; synthetic DoRA images were noise and do not
  establish trained-model quality.
- Composable with signed NegPiP and img2img Alternative has worker-level SD15/SDXL
  execution checks; these are not full numerical upstream equivalence. Guarded
  genuine SDXL PhotoMaker diagnostics exercised heterogeneous Composable branches,
  schedules, CFG≤1 and >1, merge, offload and interruption; limited signed
  NegPiP and MultiDiffusion cases also ran with gates temporarily bypassed.
  They do not establish full parity, and all production compatibility guards
  remain. MultiDiffusion at 768px/20 steps produced visible seams and duplicated
  faces, including in a MultiDiffusion+Composable control without PhotoMaker.
  Pinned PhotoMaker v1 requires one trigger in each encoded branch.
- Before #85, Composable LoRA with PhotoMaker and Neutral Prompt with PhotoMaker
  were already explicitly rejected. The additional offload + MultiDiffusion +
  PhotoMaker + Embedding Merge guard is new and applies only when Model Keyword
  is enabled; disabling it retains the previous execution path, whose visual
  quality and behavior for that combination are not established by this change.
- Genuine PhotoMaker v1 heterogeneous keyword batching passed before #47 without
  Composable, but #47 later identified and fixed existing shared-adapter leakage.
  Dynamic-rank/DoRA and special tensor formats have explicit limitations; tests
  with synthetic magnitude values do not prove general compatibility.
- The pinned upstream uses one selected LoRA. Ordinary native batch activation
  follows the upstream host's first-image adapter behavior; Composable has its
  own per-branch semantics. Hires fallback, styles and PNG replay must be
  preserved across later integrations.

## Integration and verification record

The integrated source retains the #47 scopes and earlier session, Hires,
metadata, conditioning and preset work. The verification record distinguishes
native/package checks, CPU regressions, the initially failing full Scala run,
and experimental guard-bypassed combinations. The final ZIP differs from the
real-model-tested ZIP only in the notice file. Independent review reported no
concrete blocker. Parent coordination covers integration, push, issue closure
and external MCP transport verification; MCP calls within the UI harness are not
an external transport test.
