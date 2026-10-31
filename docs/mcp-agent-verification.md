# External MCP client verification

## Scope and authority

The 2026-10-31 goal is practical AI-agent access, including model/capability discovery, model selection, generation/editing, job progress/results, cancellation and supported extensions over both stdio and Streamable HTTP. The verification client is the official `mcp==2.3.0` Python SDK in an isolated Python 3.12 environment. `scripts/dev/check_mcp_client.py` starts a separate JVM or packaged launcher; it does not invoke Scala tools directly.

## Baseline before the new MCP surface

Baseline accepted source: `d8375052e07df4295bc593938b63cb36844bdd47`. External-client runs against the accepted jar passed for SD1.5 (`animekawa_v10`, 512px) and SDXL (`waiIllustriousSDXL_v170`, 768px), 12 steps, over each transport:

- `outputs/mcp-sdk-real-stdio-sd15/receipt.json`
- `outputs/mcp-sdk-real-http-sd15/receipt.json`
- `outputs/mcp-sdk-real-stdio-sdxl/receipt.json`
- `outputs/mcp-sdk-real-http-sdxl/receipt.json`

Each receipt records tool discovery, model/sampler/script listing, set_model, txt2img, get_image, img2img, async progress/completion, cancellation and successful generation after cancellation. Each run exited 0. Both model families' representative images were viewed. HTTP runs use a separate headless HTTP server, not the embedded GUI HTTP instance.

The existing packaged `SDNUI.exe --mcp` also passed official-SDK stdio dry-run workflows (`outputs/mcp-sdk-packaged-launcher-stdio`), using the bundled Java runtime and no launcher-config modification.

## New MCP surface

AlwaysOn discovery and per-job configuration, PNG metadata inspection/replay, variation, independent Hires fields, manual LoRA and ControlNet are integrated. Final independent review found no remaining Critical or Important issues after one fix/re-review round. The nested AlwaysOn schema was validated against the documented payload; pre-queue semantic validation, shared replay work limits, finite-number checks, consumed-field reporting, and SDK progress/image assertions were strengthened.

The merged source passed 85 MCP tests, four focused plugin tests, eight focused Python tests, and JDK21 assembly (`outputs/mcp-main-integration.log`). The broader pre-merge fix wave also passed 1,264 no-torch Python tests with 136 skipped and three torch-only tests deselected. Assembly produced jar SHA-256 `e85e15c1d010b009799d6f2ed4fe169e4b1cd98c40f4f9bd1c3404db963f2f22`.

A fresh Windows app-image ZIP was extracted and checked against the merged source: all 116 Python payload files and 160 other source/licence/document files matched byte-for-byte, as did the packaged jar. The real-model acceptance ZIP SHA-256 was `58c6762c0459673c9ad0df89af9a569f22e75c0bf99b5ccb75439ec7d4bb8100` (`outputs/mcp-final-payload-verification.json`). The extracted `SDNUI.exe` and bundled runtime passed official-SDK stdio and HTTP dry-runs with `--require-extensions`; receipts are in `outputs/mcp-sdk-final-packaged-dry-{stdio,http}/receipt.json`.

Final real-model acceptance used the same extracted launcher and jar with official `mcp==2.3.0`, protocol `2025-06-18`, 12 steps, SD1.5 `animekawa_v10` at 512px and SDXL `waiIllustriousSDXL_v170` at 768px. All four runs exited 0:

- `outputs/mcp-sdk-final-stdio-sd15/receipt.json`
- `outputs/mcp-sdk-final-http-sd15/receipt.json`
- `outputs/mcp-sdk-final-stdio-sdxl/receipt.json`
- `outputs/mcp-sdk-final-http-sdxl/receipt.json`

Every run covered tool/model/sampler/script/AlwaysOn discovery, model selection, txt2img with SDK progress notifications through 100, decoded PNG image content, ordinary and FreeU PNG candidate replay with two exact RGB-equality checks, img2img, mask inpaint, async polling/completion, cancellation, and successful generation after cancellation. Representative txt2img, FreeU, img2img and inpaint images from both model families and transports were viewed. `outputs/mcp-sdk-final-acceptance-summary.json` records the consolidated checks. HTTP evidence uses a separate headless server, not the embedded GUI HTTP instance. No test server or engine process was left running.

## Limits

- MCP has its own per-request configuration and does not inherit live unsaved GUI controls or its password session.
- Custom Code and model conversion/merge restrictions remain explicit.
- Metadata replay returns candidates and unapplied fields, never executes untrusted metadata automatically.
- See `mcp-generation-metadata.md` for the current GUI/MCP coverage matrix.
- Model-image compatibility and the quality of a prompt remain model dependent; a passing transport test does not prove every extension combination.
