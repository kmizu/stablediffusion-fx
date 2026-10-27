# Native tag autocomplete (#24)

Plan: implement and unit-test CSV parsing and active-token replacement; connect a local CSV chooser and shared asynchronous dictionary to both prompt editors; add keyboard and IME-aware popup handling; let the release coordinator run the focused Scala/native checks before acceptance.

Behavior reference: DominikDoom/a1111-sd-webui-tagcomplete README at
`4170882f90b47be130a0ff9314f663c230b9153d` (MIT), retrieved through GitHub API on 2026-10-27.
https://github.com/DominikDoom/a1111-sd-webui-tagcomplete/tree/4170882f90b47be130a0ff9314f663c230b9153d

Independent native implementation; no upstream code or tag data copied. CSV supports
`name,type,count,"alias1,alias2"`, optional count/aliases, UTF-8 and quoted fields.
Header rows containing `name` and `type` or `category` are also supported (including
WD CSVs with `tag_id,name,category,count`); columns are selected by header name.
Choose a local file through **Tags…** beside the prompt. No automatic downloads.
Suggestions match tag/alias prefixes (spaces and underscores equivalent), ranked by count.
Up/Down selects, plain Enter/Tab accepts, Escape dismisses. Ctrl+Enter remains generation.
Only the active comma/bracket-delimited token is replaced, preserving weights and surrounding text.
Selection crossing tokens suppresses completion. Composition events suppress the popup and
its key handling until composition is committed. Existing native undo uses replaceText.

Scope does not include upstream translation downloads, wildcards, chants, network previews,
or full extension settings parity. Files up to 32 MiB / 300,000 rows are admitted.
Verification is pending the coordinator's focused tests and native checks.
