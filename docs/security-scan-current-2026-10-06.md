# Security scan reconciliation, 2026-10-06

The source baseline was main c1878e5b; the scan image did not identify its scanned commit. Three parser issues were already addressed in the development working tree. The user subsequently authorized fixing the remaining eight. The old external scan has not been rerun; this reports code changes and focused regressions.

| Reported issue | Repair / evidence |
| --- | --- |
| Empty GGUF special token stalls tokenizer | Reject empty CONTROL/USER_DEFINED tokens before SplitSpecial; both token types tested. |
| Quadratic tokenization under model gate | Linked symbols and ranked adjacency queue replace repeated full scans; preserve selected-pair batch semantics. Bound text to 1 Mi characters, each piece to 64 KiB, merge work to 16 Mi units, output to model context, and honor cancellation. Existing Qwen tokenizer regressions and new limits pass. HTTP message count/aggregate characters checked before model gate. |
| Nested GGUF arrays exhaust parser stack | Configurable depth/count/remaining-byte/decoded-budget checks with bounded fixtures. |
| GGUF metadata lengths allocate before validation | Read limits prevalidate lengths, remaining bytes and cumulative allocation budget. |
| Unauthenticated loopback management | Per-launch random 256-bit token passed to GUI child via process environment, client Bearer header, constant-time comparison before parsing. Manual server prints its ephemeral token only when no launch token is supplied. Authentication rejection tested on actual isolated HTTP server. |
| Unbounded HTTP parsing/queue concurrency | Four nonqueued admission slots before JSON parse; fifth returns 429. Body max 40 MiB, read timeout 15s; connection/header limits. Actual four stalled bodies/fifth rejection tested. |
| Overlapping safetensors allocation amplification | Reject overlapping nonempty ranges and enforce decoded payload/count limits before reads. |
| BPE exponential byte expansion | Per-token 1 MiB, aggregate 128 MiB and merge-count caps before concatenation allocation; 21-step doubling regression rejected. |
| Outbound SMB via model paths | Reject UNC, device and root-relative network spellings before File.Exists; reject mapped network drive roots. Actual HTTP tests reject three UNC/device forms without touching their targets. |
| Parquet row-group allocations | Prevalidate selected-column metadata before text materialization: max 1M rows/values, 64 MiB compressed, 256 MiB uncompressed. Sequential and shuffled readers reject a 1,000,001-row fixture even with maxDocuments=1. |
| Browser simple POST unload/shutdown | Require authentication, local Host and remote loopback, reject Origin; require JSON content type for every POST, validate shutdown JSON before side effect. Actual HTTP tests prove rejected requests leave server healthy. |

Production GUI builds without warnings/errors; Core CPU suite 1500 passed / 109 skipped and Integration CPU suite 494 passed / 6 skipped. Security HTTP checks and ten offline GUI checks passed. The original external scanner and malicious decoded Parquet payload fuzzing were not run. Row-group metadata budgets bound ordinary materialization but do not claim to audit every allocation inside third-party decompressors. Standard Gui.Tests restore remains blocked by the known permission restriction; offline SDK compilation was used without bypassing that restriction. Existing user GUI/server processes were preserved.