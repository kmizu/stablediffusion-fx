# sdfx MCP server

Lets AI agents (Claude Code, other MCP clients) generate images with Stable Diffusion FX.

Two transports:

- **stdio (headless)**: `sdfx.mcp.McpMain` starts its own engine. `initialize` is answered at once; tools wait for the engine (`*_async` return at once with status `waiting_for_engine`).
  Options: `--home DIR`, `--dry-run`, `--http [PORT]` (serve HTTP instead of stdio), `--token TOKEN`, `--allow-dir DIR`, `--drain-seconds N`.
  stdout carries only MCP messages; all logs go to stderr.
- **Streamable HTTP (inside the GUI)**: enable *MCP HTTP server* in the settings (`mcp_http_enabled`, port
  `mcp_http_port`, default 7861, optional `mcp_token`). The endpoint is `http://127.0.0.1:7861/mcp`. Jobs go
  through the GUI's queue, so they show up in its preview and gallery. It binds to 127.0.0.1 only, rejects
  non-loopback `Origin`/`Host` headers, and requires `Authorization: Bearer <token>` when a token is set.

## Security and limits

- **Image paths.** `init_image`, `mask`, `get_image` and `run_script` `params.init_image` accept file paths only
  inside the outputs folder (including its `.tmp`), plus folders you allow explicitly with `--allow-dir <dir>`
  (repeatable, headless) or the env var `SDFX_MCP_ALLOWED_DIRS` (list separated by `;` on Windows, `:` elsewhere;
  this also applies to the GUI's HTTP server). Paths are resolved to real paths (symlinks/junctions cannot escape)
  and relative paths resolve against the outputs folder. UNC / network paths (`\\host\share`, `//host`) are always
  refused, so no SMB connection is ever opened. The error is the same for "missing" and "not allowed". Base64 and
  `data:` URIs are always accepted.
- **HTTP token.** The bearer token is optional. Without one, **any local process or other user on this machine** can
  generate images, switch the model, interrupt or clear the GUI's queue and read images from the allowed folders.
  Browsers are blocked (only absent or `http(s)://localhost|127.0.0.1|[::1]` Origins and loopback Host headers pass),
  but set `mcp_token` on shared machines. Headless: `--token` or env `SDFX_MCP_TOKEN` (preferred: not visible in the
  process list).
- **Sessions** (HTTP) expire after 30 minutes idle; when 64 sessions exist only idle ones are evicted.
- **Limits:** request body / stdio line 64 MB; at most 32 concurrent tool calls (-32000 "server busy" beyond);
  input images 4096x4096 (checked from the header before decoding); width/height and hires target 4096 per side;
  `width*height*batch_size` <= 16.7 MP and total output <= 64 MP per job; 16 images per call; 32 pending MCP jobs;
  inline previews <= 20 MB of base64 per response (the rest are returned as paths - use `get_image`).
- **stdio hygiene.** stdout carries only JSON-RPC. Launch the JVM with
  `-Xlog:disable -Xlog:all=warning:stderr -Djava.awt.headless=true` so JVM warnings cannot reach stdout, and keep any
  launcher script silent (`@echo off`, no banner, no `pause`).
- On stdin EOF, in-flight calls get `--drain-seconds` (default 2) to finish before the process exits.

## Claude Code setup

```sh
# headless (stdio), via the launcher script
claude mcp add stablediffusion-fx -- D:/stable-diffusion-webui-reForge/stablediffusion-fx/run.bat --mcp
# headless (stdio), using the jar directly
claude mcp add stablediffusion-fx -- java -cp D:/stable-diffusion-webui-reForge/stablediffusion-fx/target/stablediffusion-fx.jar sdfx.mcp.McpMain --home D:/stable-diffusion-webui-reForge/stablediffusion-fx
# the running GUI (HTTP)
claude mcp add --transport http stablediffusion-fx-gui http://127.0.0.1:7861/mcp
# ... with a token
claude mcp add --transport http stablediffusion-fx-gui http://127.0.0.1:7861/mcp --header "Authorization: Bearer <token>"
```

Or copy `../.mcp.json.example` to `.mcp.json` in your project and adjust the paths.
Development: `sbt "mcp/runMain sdfx.mcp.McpMain --dry-run"` (no GPU needed).

## Tools

| Tool | Purpose |
|---|---|
| `txt2img` | Generate from a prompt (size, steps, CFG, sampler, scheduler, seed, batch, clip skip, checkpoint, VAE, hires fix, `loras: [{name, weight}]`, styles). Waits; returns previews + paths + infotext. |
| `img2img` | Same plus `init_image` (base64, data URI, or a path inside the allowed folders), `mask`, `denoising_strength`, `resize_mode`, `inpainting_fill`, `inpaint_full_res`, `mask_blur`. |
| `txt2img_async` / `img2img_async` | Queue and return `{job_id}` immediately, for clients with short tool timeouts. |
| `get_job` | Status / progress / images of an async job (`wait_seconds` to block). |
| `cancel_job` | Cancel or interrupt an MCP job. |
| `list_models` / `set_model` | Checkpoints (kind, loaded) and VAEs; load a checkpoint. |
| `list_loras` | LoRAs with ready-made `<lora:..>` tags. |
| `list_samplers` | Samplers, schedule types, hires upscalers, upscalers, styles. |
| `list_recent` / `get_image` | Browse the outputs folder (only files inside it can be read). |
| `get_status` | Engine state, device, loaded model, current job, queue length. |
| `interrupt` | Stop the running job (`clear_queue` drops pending ones). |
| `list_scripts` / `run_script` | Script plugins (X/Y/Z plot, prompt matrix, loopback, SD upscale...) with their controls; `async: true` returns a job_id. |

Generation results: `content` = one `image` block per image (downscaled to `max_image_side`, default 1024;
`return_images: false` to skip) + a `text` block with each saved path and its infotext; `structuredContent` =
`{job_id, images: [{path, seed, infotext, grid}]}`. At most 16 images per call. Progress is reported with
`notifications/progress` when the call carries `_meta.progressToken`; `notifications/cancelled` interrupts the job.
