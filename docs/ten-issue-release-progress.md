# Ten-issue release goal

The user reset the prior goal on 2026-10-30 and requested ten issue resolutions followed by a release. This tracker counts further distinct issues completed after that reset; the ten waves in `twenty-issue-release-progress.md` are earlier project history.

Baseline local and remote `main`: `83b00b598f48606eafe7266f03fcbab4c1b9c1a7`. At reset, #48 image scrambling was in progress and uncommitted. Accepted under this goal: **1 / 10**.

## Accepted

- [#48 image scrambling](https://github.com/kmizu/stablediffusion-nui/issues/48): commit [`b8a623fdad1d78317a10b37149192a59ccff3499`](https://github.com/kmizu/stablediffusion-nui/commit/b8a623fdad1d78317a10b37149192a59ccff3499) was pushed to `origin/main`; local HEAD and remote HEAD matched and the working tree was clean at verification. Issue closed 2026-10-30 15:17:54 UTC. See `encrypt-image-verification.md` for evidence and scope: SVD has no actual SVD-model test, metadata and temporary files may be readable, and 16-bit LA is unsupported. The mode is pixel scrambling, not secure encryption.

## In progress

- #85 is underway in isolated worktree `C:/Users/Mizushima/.codex/worktrees/d919/stablediffusion-nui`. No integration into main has occurred yet.
- #61 passed local acceptance in worktree `C:/Users/Mizushima/.codex/worktrees/4496/stablediffusion-nui` and is being integrated into main. Do not count it until commit, push and issue closure are verified; see `img2img-hires-fix-verification.md`.
- #46 is underway in isolated worktree `C:/Users/Mizushima/.codex/worktrees/embedding-inspector/stablediffusion-nui` at the same base. No integration into main has occurred yet.

## Release gate

Verify ten distinct resolutions and clean local/remote state, then build and test the distributable, publish it, and verify the remote tag and downloaded artifact hashes.
