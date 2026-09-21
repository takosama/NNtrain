# DRN depth-scaled residual initialization (2026-09-05)

## Initialization hypothesis and change

Correction: the repeated-batch diagnostic below does not establish the cause
of the production FineWeb loss plateau, nor prove that initialization fixes it.
The old stalled checkpoint was unavailable for a direct comparison.

A DRN layer has two residual branches: the memory output projection and the
FFN second projection. Previously both used the global initialization scale
unchanged at every depth. Increasing 16 to 32 layers therefore increased
unscaled residual additions from 32 to 64 and, in `mix8_32`, doubled the
number of residual-stream BFP8 requantizations.

Fresh DRN models now initialize only those two output projections with

`residualScale = initializationScale / sqrt(2 * layers)`.

For 32 layers and `initializationScale = 0.02`, this is `0.0025`. Memory and
FFN input projections keep `0.02`; V2/V3 and standalone-layer initialization
remain unchanged. Runtime math, public APIs and checkpoint tensors are not
changed. Loading an existing checkpoint overwrites the fresh initialization,
so an old unscaled checkpoint intentionally remains bit-for-bit unchanged.

The CLI now prints the fresh-model residual output scale. Layer tracing is
available only through the benchmark command, so production training has no
per-layer retention, reduction or D2H overhead.

## GPU diagnostic

Command:

```powershell
dotnet run --project .\NNtrain.Benchmarks\NNtrain.Benchmarks.csproj `
  --configuration Release --no-build -- `
  --diagnose-drn-depth .\benchmark-results\drn-depth-scaled-20260905.json 200
```

Conditions: RTX 3070 Ti GPU0; batch 2, sequence 16, vocabulary 512, width 64,
hidden 192, K/V 16/16, dropout 0.1, initialization 0.02, fixed repeated tokens,
seed 1234. Ordinary Muon fixed NS5 every update at LR 0.0001 plus AdamW at
LR 0.0003, weight decay 0.01. Each variant ran 200 updates. This is a small
controlled diagnostic, not a FineWeb convergence or production throughput
claim.

| Variant | Initial loss | Final loss | Reduction |
| --- | ---: | ---: | ---: |
| L16 mix8_32, depth-scaled | 6.2602 | 1.5460 | 4.7142 |
| L32 mix8_32, old unscaled | 6.2422 | 1.6473 | 4.5949 |
| L32 mix8_32, depth-scaled | 6.2553 | **1.4088** | **4.8466** |
| L32 mix16_32, depth-scaled | 6.2548 | 1.4134 | 4.8415 |

The corrected 32-layer mix8 run ended 0.2385 loss below the old unscaled
32-layer run and 0.1373 below the corrected 16-layer run in this diagnostic.
Its result also closely matched corrected mix16_32 (difference 0.0046).

### Layer trace before training

| Variant | Layer 1 residual RMS | Layer 16 | Last layer | Layer 1 grad RMS | Last grad RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| L16 mix8 scaled | 0.01241 | 0.01455 | 0.01455 | 0.02795 | 0.02400 |
| L32 mix8 old unscaled | 0.01624 | 0.04791 | **0.06641** | 0.01998 | **0.00545** |
| L32 mix8 scaled | 0.01233 | 0.01346 | **0.01437** | 0.02834 | **0.02454** |
| L32 mix16 scaled | 0.01232 | 0.01346 | 0.01437 | 0.02829 | 0.02453 |

The old 32-layer residual RMS grew 4.09x from the first to last layer, while
the scaled run grew only 1.17x. Its last-layer activation-gradient RMS was
also 4.50x smaller than the scaled run. These measurements show a change in
residual/activation-gradient scale under this initialization; they do not alone
establish vanishing gradients or the cause of the production loss plateau.

For corrected 32-layer mix8, cumulative activation difference relative to
mix16 was 0.7% at layer 1 and 4.47% at layer 32. The corresponding old-unscaled
comparison is not a pure quantization measurement because its initialization
also differs. Full per-layer residual RMS, gradient RMS and corrected
mix8-vs-mix16 differences are in the raw JSON.

## Validation

- Release benchmark build: 0 warnings, 0 errors.
- Core DRN/V2/V3/model/conversion selection: 43 passed.
- Integration checkpoint/configuration selection: 148 passed.
- Existing checkpoints are not migrated or rescaled. A fresh checkpoint/run
  is required to receive the initialization fix; rescaling a trained model
  without matching optimizer-state transformation would be unsafe.

Raw result: `benchmark-results/drn-depth-scaled-20260905.json`.
