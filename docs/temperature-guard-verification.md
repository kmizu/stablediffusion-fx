# GPU temperature guard acceptance — 2026-10-30

Index 186 / issue #54: native AlwaysOn `gpu_temperature_guard`, shared sampling-step callback, hysteresis, check/recheck intervals, maximum pause, physical device selection, optional paused readings, explicit pause/resume/timeout messages and interruptible waiting.

## Evidence

- Python whole suite passed with CUDA_VISIBLE_DEVICES=-1: `outputs/temperature-full-python.log`. Subsequent sensor type validation plus pin checks: 23 focused tests passed.
- Scala whole suite passed: 1264 passed, 1 skipped, zero failures; `outputs/temperature-full-scala.log`, exit 0. The native bulk scanner required about eight minutes; it was not restarted.
- Actual SD1.5 and SDXL off/cool/bounded-wait comparisons passed exact RGB parity. Actual img2img and hires comparisons also passed exact parity. Reports under `outputs/verify-temperature-*`; see investigation for precise directories.
- Real engine interruption during an unlimited low-threshold wait completed in 0.141 seconds (SD1.5), 0.110 seconds (SDXL), followed by successful ordinary generation.
- Actual native Generate/Interrupt/Cancelled/recovery paths passed on both models. SDXL interruption-only latency: 0.1588864 seconds. The earlier SD1.5 printed timing included recovery and is not interruption-only. Recovery PNGs were saved and inspected; native SDXL artifact is `outputs/verify-temperature-native-sdxl/outputs/txt2img/2026-10-30/00000-7.png`.
- Candidate assembly and Java 21 launch passed; expanded form inspected at `outputs/verify-temperature-ui-panel.png`. sbt jar hash: `ba9468da04301d337552100ee9af053ca26cb283`.
- Notices regenerated for 144 projects. Upstream AGPL license pinned to f388d7081945576ca2361c365d16f5edb3250412 with SHA-256 248a1ab6a57e6f677770a1498efc7a567f22aaae68923ae84c47d8a3db0ee699.

## Limits and operating semantics

NVIDIA hardware was tested on RTX 3090. ROCm and OpenHardwareMonitor adapters have fixture/schema coverage only; AMD hardware was unavailable. External monitors are not bundled or installed. WMI namespace/fields were verified against source; OHM's zero sentinel is rejected. No code was copied from the external monitors. A missing sensor stops processing instead of reporting cool. A provider subprocess can take up to its 2/3-second timeout; interrupt checks inside temperature sleep occur every 0.1 seconds.

Checks occur at sampling-step end, not during a single long GPU kernel. Maximum pause resumes generation even above the wake temperature and emits a warning, matching the source feature's explicit semantics. Zero interval or wake above sleep disables queries. Sensor numbers are physical provider IDs, not CUDA-visible ordinals. Waiting preserves the latent state; no artificial heating was used. PhotoMaker-specific callback compatibility and AMD live execution are not claimed.

See `temperature-guard-investigation.md` for primary source pins, historical intermediate results and exact test routes. Final ledger/document and remote integration are separate steps.
