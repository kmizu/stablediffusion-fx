# Stable Diffusion NUI MCP server

Lets AI agents (Claude Code, other MCP clients) generate images with Stable Diffusion NUI.

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

## Client setup

The Windows package supports stdio directly with `C:/apps/SDNUI/SDNUI.exe --mcp --home C:/apps/SDNUI`, using its bundled Java runtime. Configure the Python environment and model folders in that home first. `.mcp.json.example` shows this packaged connection; replace the installation path. Use one server configuration for a GPU to avoid starting duplicate model engines.

The commands below show source-checkout and GUI HTTP alternatives. For another MCP host, use the same executable/arguments or HTTP endpoint in its MCP configuration.

### Claude Code examples

```sh
# headless (stdio), via the launcher script
claude mcp add stablediffusion-nui -- C:/path/to/stablediffusion-nui/run.bat --mcp
# headless (stdio), using the jar directly
claude mcp add stablediffusion-nui -- java -cp C:/path/to/stablediffusion-nui/target/stablediffusion-nui.jar sdfx.mcp.McpMain --home C:/path/to/stablediffusion-nui
# the running GUI (HTTP)
claude mcp add --transport http stablediffusion-nui-gui http://127.0.0.1:7861/mcp
# ... with a token
claude mcp add --transport http stablediffusion-nui-gui http://127.0.0.1:7861/mcp --header "Authorization: Bearer <token>"
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
| `list_always_on` | Discover AlwaysOn extensions and typed values for the generation `always_on` object; applies to sync, async and script sub-generations. |
| `inspect_generation_metadata` | Inspect a PNG without resaving it; returns generation arguments, AlwaysOn replay candidates, and unapplied metadata. Nothing executes automatically. |
| `model_keyword_lookup` / `model_presets` | Look up checkpoint/LoRA keywords and manage model-bound preset libraries. |
| `list_scripts` / `run_script` | Script plugins (X/Y/Z plot, prompt matrix, loopback, SD upscale...) with their controls; `async: true` returns a job_id. |

Generation results: `content` = one `image` block per image (downscaled to `max_image_side`, default 1024;
`return_images: false` to skip) + a `text` block with each saved path and its infotext; `structuredContent` =
`{job_id, images: [{path, seed, infotext, grid}]}`. At most 16 images per call. Progress is reported with
`notifications/progress` when the call carries `_meta.progressToken`; `notifications/cancelled` interrupts the job.

## Agent workflow

1. Call `list_models`, `list_samplers`, `list_always_on` and `list_scripts` to discover this installation.
2. Call `set_model` with `{ "name": "<catalog checkpoint>" }`.
3. Submit `txt2img_async` or `img2img_async`, poll `get_job` using `job_id`, and retrieve saved images with `get_image` (`max_side`, not `max_image_side`). Terminal statuses are `done`, `failed`, and `cancelled`. A cancelled job result carries `isError=true` and a structured `status="cancelled"`.
4. Use `cancel_job` for that MCP-owned job. `interrupt` is a broader queue operation.
5. To replay a PNG, inspect it with `inspect_generation_metadata`, review `unapplied_fields` and warnings, resolve catalog model/LoRA choices, and submit the returned arguments and supported `always_on` values. img2img/inpaint additionally need an explicit source image and mask.

Example per-job extension configuration (no desktop settings are changed):

```json
{
  "prompt": "a ceramic teapot on a wooden table",
  "seed": 173,
  "steps": 20,
  "always_on": {
    "freeu": { "enabled": true, "b1": 1.15, "b2": 1.2 }
  }
}
```

AlwaysOn control values use declared defaults plus the supplied fields; saved desktop image paths are not inherited. The existing `model_keyword` boolean keeps its upstream-enabled default and saved setting for compatibility. Use either that boolean or `always_on.model_keyword`, not both.

Typed generation options include variation (`subseed`, `subseed_strength`), separate Hires prompts/CFG/sampler/scheduler, `manual_loras`, and `controlnet`; consult `tools/list` for the authoritative schema. Existing `loras` remains the prompt-tag convenience API.

GUI coverage and replay limits are tracked in [mcp-generation-metadata.md](../docs/mcp-generation-metadata.md). External SDK verification is tracked in [mcp-agent-verification.md](../docs/mcp-agent-verification.md).
