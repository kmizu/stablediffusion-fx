# Batch generation and ZIP export

Set **Batch size = 4** and **Batch count = 5** to generate four images together, repeated five times, for twenty images. The Size card shows the calculation explicitly. Batch size consumes additional VRAM; count repeats the batch sequentially.

Live preview shows every member of the current batch as a grid, updated at the preview interval in Settings. Both approximate and TAESD previews support this. Tiles are at most 256 pixels per side, with a 1024-pixel canvas limit; four square images appear in a 512×512 grid. The status reports the batch and image range, such as `Batch 2/5 · images 5–8/20`.

The gallery automatically selects the new generation. **Save ZIP… (20)** saves its twenty individual PNGs through the native save dialog. Search does not restrict the archive, and composite grids and unrelated history are excluded. Choose **All images** for historical browsing, or a previous generation from the selector to export it.

Generation groups last for the current app session. Exporting before completion saves the currently available count shown on the button. Deleted images are removed from the group; a missing file causes an explicit export error. PNG bytes and embedded metadata are preserved. Duplicate filenames receive unique archive names, and a failed export does not replace an existing archive.
