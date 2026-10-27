# Named generation states

Use **Named generation states…** in a generation tab to keep a complete supported
settings snapshot for that mode. This is separate from **Settings presets**, whose
tracked fields can selectively overlay a form. A named state restores its saved
settings together, including inactive script and extension controls.

## Save, restore and delete

1. Open txt2img, img2img or Inpaint and edit its settings. Click **Named generation
   states…**; the window heading identifies the mode being saved.
2. Enter a name and click **Save named state**. Saving an existing name asks before
   replacing it. Names are isolated by mode.
3. Select a saved entry to inspect its fields and any unavailable resources.
4. Click **Restore state** and confirm. Shared checkpoint, VAE and clip skip also
   change. Restoration does not start generation.
5. Select an entry and use **Delete** to remove it from that mode's list and disk.
   Deletion asks for confirmation.

The library lives in `<app home>/config/generation-states`, separate from existing
preset collections. Reopening the application reads that library. **Reload** reads
external changes or retries after a storage error. **Restore backup** replaces the
whole named-state library with its preceding saved version, after confirmation;
it is not limited to the selected entry or mode.

## What is saved

- Prompt and negative prompt, sampler/scheduler, steps, CFG, dimensions, counts,
  seeds, variation settings and retained sampler eta.
- Shared checkpoint/VAE selections and clip skip.
- Mode-specific hires or img2img/inpaint settings, including disabled drafts.
- Script selection and every supported native script/extension control, including
  controls in inactive forms. Empty dynamic model choices remain empty.
- ControlNet units, their enabled states and referenced image paths; img2img WD
  batch enablement, input folder and output folder name.

The current init image and painted mask stay in place: neither their pixels nor
their source paths are restored. Other image paths are references, not archived
copies. A snapshot does not copy models. History, generated images, global
Settings and window layout are outside its scope. The variation-section expanded
state is retained with its controls.

## Missing resources and blocked restoration

Before changing the form, restoration checks the saved fields, available catalog
choices, prompt LoRA references, referenced input files and an enabled WD batch
input folder. Unavailable resources block the whole restoration and appear in
the detail/status area. It does not quietly drop the missing setting. Keep the
referenced files available or restore their catalog entries and retry. Mask
export must finish before a state can be applied.

Checks are repeated after the confirmation dialog because the form or catalog
can change while it is open. Storage errors are reported; reload the library
before trying another write. Init-image and mask preservation does not imply
that a different image is suitable for the saved inpaint settings.

## Scope and evidence

This feature independently implements manual named generation settings from the
[State Manager reference](https://github.com/SenshiSentou/sd-webui-state-manager/tree/2af62ecd5287438bbc309f7e840d34aa5c8e3cb6).
It does not import upstream IndexedDB/text history, provide automatic generation
history or thumbnails, or claim upstream storage-format compatibility. The MIT
notice is archived; see [the source/license audit](v003-source-license-audit.md).

The actual native Save, Restore, missing-checkpoint refusal, fresh-window reload
and Delete checks passed in `outputs/v003-resumed-states.log`. The Restore check
includes dimensions, eta, disabled FreeU values and preservation of the selected
init image. These are bounded acceptance cases, not every plugin/model
combination. See [the v0.03 evidence record](V003_RELEASE_SCOPE.md) for remaining
release gates.
