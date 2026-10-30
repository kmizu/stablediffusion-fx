# Ten-issue release goal

The user reset the prior goal on 2026-10-30 and requested ten issue resolutions followed by a release. This tracker counts further distinct issues completed after that reset; the ten waves in `twenty-issue-release-progress.md` are earlier project history.

Baseline local and remote `main`: `83b00b598f48606eafe7266f03fcbab4c1b9c1a7`. At reset, #48 image scrambling was in progress and uncommitted. Accepted under this goal: **3 / 10**.

## Accepted

- [#48 image scrambling](https://github.com/kmizu/stablediffusion-nui/issues/48): commit [`b8a623fdad1d78317a10b37149192a59ccff3499`](https://github.com/kmizu/stablediffusion-nui/commit/b8a623fdad1d78317a10b37149192a59ccff3499) was pushed to `origin/main`; local HEAD and remote HEAD matched and the working tree was clean at verification. Issue closed 2026-10-30 15:17:54 UTC. See `encrypt-image-verification.md` for evidence and scope: SVD has no actual SVD-model test, metadata and temporary files may be readable, and 16-bit LA is unsupported. The mode is pixel scrambling, not secure encryption.

- [#61 img2img Hires Fix](https://github.com/kmizu/stablediffusion-nui/issues/61): commit [`65b4ddcc11e8c2438c30cdd57b3d7cfc3732e019`](https://github.com/kmizu/stablediffusion-nui/commit/65b4ddcc11e8c2438c30cdd57b3d7cfc3732e019) was pushed to `origin/main`; local HEAD and remote HEAD matched and the working tree was clean at verification. Issue closed 2026-10-30T16:36:42Z. Python 1,507 passed/1 skipped and Scala 1,437 passed/1 skipped; real SD1.5/SDXL native and extracted-package checks passed. See `img2img-hires-fix-verification.md` for source-faithful pixel refinement, preserved native latent Hires and explicit scope limits.

- [#82 metadata utilities](https://github.com/kmizu/stablediffusion-nui/issues/82): commit [`777d128cd02762d4b8749ddefe3d7d70c83fc353`](https://github.com/kmizu/stablediffusion-nui/commit/777d128cd02762d4b8749ddefe3d7d70c83fc353) was pushed to `origin/main`; local HEAD and remote HEAD matched and the working tree was clean at verification. Issue closed 2026-10-30T19:15:43Z. Python 1,535 passed/1 skipped and Scala 1,441 passed/1 skipped; native and extracted-package SD1.5/SDXL checks passed, including source-bound editing, cancellation/recovery and exact generated RGB. See `metadata-utils-verification.md` for safety limits and retained earlier timeout evidence.

## In progress

- #85 is underway in isolated worktree `C:/Users/Mizushima/.codex/worktrees/d919/stablediffusion-nui`, now based on `777d128cd02762d4b8749ddefe3d7d70c83fc353`. No integration into main has occurred yet.
- #46 is underway in isolated worktree `C:/Users/Mizushima/.codex/worktrees/embedding-inspector-main82/stablediffusion-nui`, based on `777d128cd02762d4b8749ddefe3d7d70c83fc353`; earlier worktrees are retained. No integration into main has occurred yet.
- #47 is underway in isolated worktree `C:/Users/Mizushima/.codex/worktrees/690e/stablediffusion-nui` at the same base. No integration into main has occurred yet.

## Release gate

The release remains pending. Verify ten distinct resolutions and clean local/remote state, then build and test the distributable, publish it, and verify the remote tag and downloaded artifact hashes.
