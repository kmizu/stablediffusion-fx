# Ten-issue release goal

The user reset the prior goal on 2026-10-30 and requested ten issue resolutions followed by a release. This tracker counts further distinct issues completed after that reset; the ten waves in `twenty-issue-release-progress.md` are earlier project history.

Baseline local and remote `main`: `83b00b598f48606eafe7266f03fcbab4c1b9c1a7`. At reset, #48 image scrambling was in progress and uncommitted. Accepted under this goal: **2 / 10**.

## Accepted

- [#48 image scrambling](https://github.com/kmizu/stablediffusion-nui/issues/48): commit [`b8a623fdad1d78317a10b37149192a59ccff3499`](https://github.com/kmizu/stablediffusion-nui/commit/b8a623fdad1d78317a10b37149192a59ccff3499) was pushed to `origin/main`; local HEAD and remote HEAD matched and the working tree was clean at verification. Issue closed 2026-10-30 15:17:54 UTC. See `encrypt-image-verification.md` for evidence and scope: SVD has no actual SVD-model test, metadata and temporary files may be readable, and 16-bit LA is unsupported. The mode is pixel scrambling, not secure encryption.

- [#61 img2img Hires Fix](https://github.com/kmizu/stablediffusion-nui/issues/61): commit [`65b4ddcc11e8c2438c30cdd57b3d7cfc3732e019`](https://github.com/kmizu/stablediffusion-nui/commit/65b4ddcc11e8c2438c30cdd57b3d7cfc3732e019) was pushed to `origin/main`; local HEAD and remote HEAD matched and the working tree was clean at verification. Issue closed 2026-10-30T16:36:42Z. Python 1,507 passed/1 skipped and Scala 1,437 passed/1 skipped; real SD1.5/SDXL native and extracted-package checks passed. See `img2img-hires-fix-verification.md` for source-faithful pixel refinement, preserved native latent Hires and explicit scope limits.

## In progress

- #85 is underway in isolated worktree `C:/Users/Mizushima/.codex/worktrees/d919/stablediffusion-nui`. No integration into main has occurred yet.
- #46 is underway in isolated worktree `C:/Users/Mizushima/.codex/worktrees/embedding-inspector-integrated/stablediffusion-nui`, based on `65b4ddcc11e8c2438c30cdd57b3d7cfc3732e019`; the earlier worktree is retained. No integration into main has occurred yet.
- #82 passed local acceptance in worktree `C:/Users/Mizushima/.codex/worktrees/d912/stablediffusion-nui` and is being integrated into main. Do not count it until push and issue closure are verified; see `metadata-utils-verification.md`.

## Release gate

The release remains pending. Verify ten distinct resolutions and clean local/remote state, then build and test the distributable, publish it, and verify the remote tag and downloaded artifact hashes.
