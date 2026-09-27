# Qwen3.5 LoRA training performance

Measured on 2026-09-27 with two Intel Arc B580 12 GB cards and
`Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf`. The base has 64 layers and width 5120.
The training configuration uses all 496 Attention/DeltaNet/MLP projections,
rank 8, alpha 16, Adam learning rate 0.0001, gradient clipping 1, and seed 1.
There are 58,363,904 trainable parameters. The base weights remain frozen and
encoded on the GPUs throughout forward and backward.

## Same four updates before and after

Both runs start from the same initialized adapter and use the same example
sequence (rows 1, 2, 3, 1 of the local 512-example conversation dataset). Times
cover the training update, excluding model loading and checkpoint saving.
The total token counts below include the final supervised token; the forward
pass therefore receives one fewer token.

| Update | Total / supervised tokens | Before (s) | After (s) |
|---:|---:|---:|---:|
| 1 | 26 / 12 | 13.9983771 | 3.6229951 |
| 2 | 47 / 14 | 24.0646614 | 5.2653642 |
| 3 | 22 / 10 | 11.6051315 | 2.9442985 |
| 4 | 26 / 12 | 13.5116634 | 3.2702896 |
| Total | | 63.1798334 | 15.1029474 |

This sample is **4.1833 times faster**, or 15.795 to 3.776 seconds per update.
It is not a measurement of total three-epoch training time or inference speed.
A repeat using the normal repository build reproduced approximately 15.10 s
for the four updates; the dataset's longest conversation (69 tokens) took
7.77 s as the fifth update without truncation.

## Changes selected by measurement

- Backward quantized projection reuses each decoded weight across 16 token rows.
  Tests preserve the reference partial-sum layout and accumulation order. Tiles
  of 4, 8, 16 and 32 rows were measured; 16 won on these GPUs.
- IQ2_S, IQ3_S and Q4_K forward projections reuse decoding across four rows,
  retaining the SG16 accumulation order.
- LoRA A projection and intermediate gradients use cooperative reductions.
- DeltaNet Q/K backward uses a cooperative 32-lane kernel at supported widths.
- Gradient norm accumulation uses eight splits and two device readbacks instead
  of one readback per adapter matrix.
- The vocabulary head processes only rows with supervised targets. Backward
  scatters their gradients into the complete causal trunk sequence.
- Each device may retain up to 512 MiB of reusable training buffers. Inference
  retains its previous kernel selection and buffer configuration.

Encoded weight residency stayed exactly 5,035,468,800 and 5,116,405,760 bytes on
the two GPUs. The remaining largest cost is IQ2_S backward projection: 7.096 s
of GPU kernel time across the four optimized updates, approximately 47% of
their wall time. Kernel totals and wall times cover different measurements.

## Correctness checks

The maximum loss difference across the four updates is 3.7997961e-6. Both
saved step-4 checkpoint headers are byte-identical and their SHA256 trailers
validate. A streaming comparison checked all 175,091,712 FP32 values across
496 matrices and all six arrays per matrix (A, B and Adam moments). All values
are finite and second moments are nonnegative. Aggregate maximum absolute
difference is 3.1680800e-5; relative L2 difference is 3.7332114e-6. Cooperative
reductions change floating-point summation order, so whole training states are
not bit-identical.

The normal repository build passed 131 Qwen3.5 core tests and 35 command/reporting
integration tests. Six legacy DRN CUDA-only cases were skipped on this Arc host.
Tests include quantized forward/transpose parity and tails, DeltaNet gradients,
response-head masking, optimizer/resume behavior, `lora` routing, durable loss
history and HTML opening before the first update.

```powershell
dotnet build NNtrain.Cli --configuration Release
dotnet test NNtrain.Core.Tests --configuration Release --filter "FullyQualifiedName~Qwen35"
dotnet test NNtrain.IntegrationTests --configuration Release --filter "FullyQualifiedName~QwenLora|FullyQualifiedName~TrainingMetricReporter|FullyQualifiedName~LossGraphMetricAdapter|FullyQualifiedName~LoraCommandTests"
```

Use the existing `lora --model <base.gguf> --config <training.json> [--resume]`
entry point. See [the training guide](qwen35-lora.md) for HTML output, checkpoint
compatibility and GGUF adapter export. Execution tuning is outside the numerical
training identity, so existing checkpoints retain their optimizer state.
