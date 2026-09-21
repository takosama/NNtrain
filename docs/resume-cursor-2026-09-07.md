# Silent streaming resume diagnosis — 2026-09-07

The supplied log had already restored model weights, NekoMuon/AdamW state,
and global step 87,614. The next stage silently replayed the corpus and
document shuffle to discard 7,449,567 previously consumed documents. It does
not tokenize skipped documents or update the model, so GPU inactivity during
this stage is expected.

Read-only reproduction using the real configuration and manifest:

```powershell
dotnet benchmark-results/resume-cursor-2026-09-07/bin/NNtrain.Benchmarks.dll --verify-resume-cursor training.forgetmemorydrn-wiki-jp.json
```

Result: all **7,449,567** documents skipped in **164.5 seconds**; a next document
was available; exit code 0. The dataset contains 139 Parquet files totalling
37,237,718,426 bytes. This is a single wall-clock diagnostic run, not a cold-cache
benchmark or verification of the next CUDA optimizer update.

Fixes:

- Print cursor-replay stage before reading; every five seconds report count,
  percentage, elapsed time and estimated time remaining.
- Timer-based reporting continues while the reader is blocked on metadata,
  decompression or filling the shuffle buffer. Timer is stopped and joined
  before normal training output resumes.
- Preserve the original shuffle order, number of skipped documents and token
  buffer. No direct-seek shortcut was introduced: replay still takes time.
- Reject a corpus shorter than the saved cursor instead of silently ending
  the epoch.
- Add a model/GPU/tokenizer-free, read-only cursor diagnostic to Benchmarks.

Verification: 34 related integration tests passed, including eight new cases
for exact shuffled suffixes, buffer draining, corpus truncation, enumerator
cleanup and progress while MoveNext blocks. Final CLI Release build has zero
warnings/errors. Production JSON, tokenizer, model, optimizer, run marker and
metrics were not modified. Manifest remained 3,012 bytes, last-write
2026-09-07 15:31:34 JST, SHA256
`6CBB693E3757E4CE4054098B2770F28C40A10AAA0E0A04DA8E261FE208D3880F`.
