# MCP Agent Workflows Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement the tasks and verify each result.

**Goal:** Let an AI discover available generation capabilities, configure them, generate/edit images, inspect progress/results, cancel work, and replay generation data through actual MCP clients over stdio and Streamable HTTP.

**Architecture:** Keep AppServices/JobQueue/Generator as the execution authority. Publish the existing declarative plugin controls as typed MCP discovery and validated per-job values. Use the official Python MCP client SDK against a separate server process for transport acceptance; do not substitute direct Scala calls for protocol evidence.

**Tech Stack:** Scala 3, existing MCP transports, Python official MCP SDK for isolated verification.

**Spec:** User goal set 2026-10-31: practical AI-agent MCP access, both transports, actual SD1.5/SDXL workflows, explicit GUI/MCP gaps, issue resolution and release every ten issues.

## Global constraints

- Preserve existing internal sdfx/SDFX names and settings; never import or launch parent reForge.
- Keep GUI and MCP jobs on the same queue; preserve per-job cancellation ownership and image path policy.
- Keep Custom Code and model conversion/merge restrictions explicit; do not silently expose arbitrary local file access or code execution through plugin values.
- Use real SD1.5 and SDXL models and inspect generated images for acceptance. Distinguish dry-run, direct service, external protocol, and packaged launcher evidence.
- Current issue batch has nine accepted issues. MCP integration changes do not count as a closed issue unless tracked and resolved with its own evidence.

## Review focus

- Image controls in plugins must use the same allowed-root/base64 policy as ordinary MCP image inputs.
- Malformed/unknown/disabled plugins or controls must fail before queueing; no accidental fallback to saved GUI values.
- Async/script/sync generation must receive identical plugin configuration and dispose temporary inputs on failure/cancellation.
- Existing MCP default behavior and user settings must remain documented; forthcoming Model Keyword defaults need integration.
- Real external SDK clients must handle structured results, image blocks, progress, job polling and cancellation over both transports.

## Tasks

- [x] Establish external official-SDK stdio/HTTP dry-run baseline; record tools and structured results.
- [x] Add failing tests for AlwaysOn discovery, typed values, bad inputs, image policy, sync/async/script forwarding.
- [x] Implement shared plugin discovery and per-job configuration in the MCP layer; retain host validation.
- [x] Add complete generation fields and PNG inspection/replay with parameter validation, without treating infotext as trusted paths.
- [x] Run real SD1.5/SDXL external-client workflows over both transports, including async progress, results and cancellation.
- [x] Update connection examples, tool reference, GUI/MCP gap matrix and exact evidence/limitations.
- [x] Review, integrate accepted concurrent changes, build/package and verify the distributable.
