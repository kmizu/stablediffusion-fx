# AGENTS.md

Instructions for coding agents (Codex and others) working in this repository.

**Start with `docs/HANDOFF.md`.** It has the current goal (bundle every A1111 extension-index entry as a built-in
plugin, with licence notices), progress, the next task, environment paths and verification commands.

## Project

stablediffusion-fx is a standalone Stable Diffusion app with four parts:

- `core/`, `engine/`, `mcp/`, `plugins/`, `ui/`: the Scala 3 + JavaFX client, built with sbt.
- `python/sdfx/`: the diffusers engine. The client starts it as `python -m sdfx` and talks JSON Lines over stdio
  (protocol v1).
- `mcp/`: an MCP server, so agents can generate images.
- The plugin system. Built-ins live in `plugins/src/main/scala/sdfx/plugins/`, registered in `StandardPlugins.scala`.

The parent directory (reForge) is reference material only. Never import or launch it.

## Commands

- Scala tests: `%LOCALAPPDATA%\Coursier\data\bin\sbt.bat -batch test`
- Fat jar: `sbt.bat -batch ui/assembly`, which writes `target/stablediffusion-fx.jar` (run it with Java 21).
- Python tests:
  - `python/.venv-smoke/Scripts/python.exe -m pytest` (GPU venv, torch).
  - `python/.venv-test/Scripts/python.exe -m pytest` (no torch).
- Extension ledger and docs: `python scripts/extensions_matrix.py`
- Licences and notices: `python scripts/third_party.py` (needs a logged-in `gh`)

## Rules

- **Plugin ids**: a plugin's `id` equals its target name in `scripts/extensions_matrix.py`.
- **Engine hooks**: every new hook name goes in `hooks.KNOWN_HOOKS`, `Engine.features()` and `dryrun.features()`.
- **Licences**: re-implement from documented behaviour. Never copy code from repositories without a licence. Add
  optional dependencies to `scripts/third_party.py` and as commented lines in `python/requirements.txt`.
- **Verification**: GPU-check new features on real SD1.5 and SDXL models, and look at the images before marking a
  target done.
- **Line endings**: write files with LF (`.gitattributes`).
- **Commits**: one commit per wave, saying what was added and how it was verified. Push to `main` of the private
  repo `kmizu/stablediffusion-fx`.
