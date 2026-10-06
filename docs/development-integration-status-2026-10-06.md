# Development integration validation, 2026-10-06

The source integration was published as `c1b53d23d5bf750659595d71ceb45bbc7cf6f505` on origin/main. The checks below describe that integration snapshot. Later uncommitted Parakeet/GUI performance changes and the README rewrite are documented separately in the [performance report](benchmarks/parakeet-performance-2026-10-06.md). Both ASR loading improvements and current CPU/GPU/GUI regressions are recorded in the [loading report](benchmarks/asr-loading-2026-10-06.md). These later changes have not been committed or pushed; publication, merge and Release updates remain on hold pending review.

The user authorized integrating all existing GUI/Arc/HF/vision/prefill and related source changes, the eight outstanding security repairs, and a normal push to the existing public origin/main. Baseline main and origin/main were c1878e5b98710e5bb413fff47215bd208380898f. Both additional clean feature worktrees already have their commits in main. No open PR or configured GitHub Actions workflow was found.

The source-only inventory excludes model weights, audio, binaries, caches, generated training/results data, personal Library identity files and private training data. A basic private-key/API-token scan found no matches; this is not a comprehensive security certification.

## Validation

- Production GUI Debug build: zero warnings/errors. Ten offline WPF state/layout/mock-transport checks passed. Standard Gui.Tests restore remains blocked by the known NuGet permission restriction and was not bypassed; its source was compiled using installed SDK references and existing xUnit assemblies.
- Isolated local HTTP security checks passed: missing authentication, browser Origin, non-JSON shutdown, malformed JSON, UNC/device/network paths, and four admitted bodies plus a rejected fifth request. No model, GPU or microphone was started by these checks.
- ASR CPU regressions: 32 passed. Actual file recognition, official CPU/Arc numerical comparisons and edited chat Send checks: [ASR validation](benchmarks/asr-2026-10-06.md).
- Broad Core CPU run with statically identified GPU-use classes excluded: 1500 passed, 109 skipped. GPU-only and mixed GPU/CPU classes excluded remain outside coverage.
- Benchmark tests: 46 passed. Selected changed Integration classes: 56 passed.
- Integration CPU run with statically identified Arc classes excluded: 494 passed, 6 skipped, no failures. The prior Mix8_16 checkpoint regression is fixed: standalone optimizer construction temporarily supplies the model precision policy when no ambient policy exists, preserving BF16 storage and restoring the caller scope. All 24 checkpoint dtype tests pass.

An initial unfiltered Integration run unexpectedly included an Arc training fixture and stalled. Newly launched testhost and its surviving Integration executable children were stopped only after verifying specific process paths/PIDs. A diagnostic timeout localized the stall to the QwenLoRA fixture. No broader GPU validation was subsequently performed. Existing user Release GUI/server processes were preserved.

Full solution --no-restore remains unavailable because the new Gui.Tests project's project.assets.json is absent. Production GUI, CLI, Core and Integration projects build independently. No user GUI was stopped for build locks.

The eight outstanding scan findings have corresponding repairs and CPU regression evidence; the original external security scan has not been rerun. See [security reconciliation](security-scan-current-2026-10-06.md). GPU development changes beyond previously approved ASR measurements have not been newly validated in this integration.
