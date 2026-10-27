# Native tag autocomplete (#24)

Stable Diffusion NUI can suggest tags from a local CSV in both positive and
negative prompt editors. The selected file is shared by the window and its path
is saved in Settings; tag data is loaded asynchronously, not copied into the app.

## Use

1. Click **Tags…** beside the prompt, then **Choose CSV…** and select a local UTF-8 file.
2. Type the beginning of a tag or alias inside a prompt. Use Up/Down to choose a
   suggestion, then plain Enter or Tab to insert it. Clicking a suggestion also inserts it.
3. Press Escape to dismiss the popup. Ctrl+Enter remains the generation shortcut
   when a completion popup or IME composition is not active.
4. After editing the CSV externally, use **Tags… → Reload**. **Disable** clears the
   configured path and suggestions; it does not delete the CSV or change prompt text.

## Supported data and editing

Behavior reference: DominikDoom/a1111-sd-webui-tagcomplete README at
`4170882f90b47be130a0ff9314f663c230b9153d` (MIT), retrieved through GitHub API on 2026-10-27.
https://github.com/DominikDoom/a1111-sd-webui-tagcomplete/tree/4170882f90b47be130a0ff9314f663c230b9153d

Independent native implementation; no upstream code or tag data copied. CSV supports
`name,type,count,"alias1,alias2"`, optional count/aliases, UTF-8 and quoted fields.
Header rows containing `name` and `type` or `category` are also supported (including
WD CSVs with `tag_id,name,category,count`); columns are selected by header name.
Choose a local file through **Tags…** beside the prompt. No automatic downloads.
Suggestions match tag/alias prefixes (spaces and underscores equivalent), ranked by count.
Only the active comma/bracket-delimited token is replaced, preserving weights and surrounding text.
Selection crossing tokens suppresses completion. Composition events suppress the popup and
its key handling until composition is committed. Existing native undo uses replaceText.

Files up to 32 MiB / 300,000 rows are admitted. A failed load is reported rather
than silently downloading a replacement dictionary.

## Scope and verification

This is local tag/alias completion. Upstream extra-network names and trigger words,
translation files, wildcards, chants, network previews and full extension settings
parity are outside the v0.03 scope. No upstream code or datasets are bundled.

The recorded native helper passed 10 checks using actual JavaFX controls with
synthetic keyboard and IME events (`outputs/v003-tags.log`). A physical Japanese
IME was **not** exercised; synthetic composition checks do not establish behavior
for every installed IME. Focused Scala regression and remaining release gates are
listed in [the v0.03 scope/evidence record](V003_RELEASE_SCOPE.md). Final integrated
native/model/package acceptance is recorded separately by the release coordinator.
