# Qwen3.5 layerwise prefill prototype, 2026-09-29

This isolated prototype processes short prompt chunks one layer at a time. Matrix projections, RMSNorm, feedforward activation, and residual additions span all rows in a chunk. The existing single-token DeltaNet and full-attention kernels still update cached state in token order. Cross-Arc hidden-state transfers occur once per chunk. The final prompt token remains on the existing logits path.

`InferencePrefillChunkTokens` defaults to zero in the model API; the GUI opts into 16. Prism models and LoRA training use the original serial prefill. LoRA generation can use either fused or separate adapter projections over multiple rows.

## Exact long-prompt probe

Qwen3.8 27B IQ2_M with the rintya QA20 LoRA on Arc 0+1. The same 1,475-token conversation and streaming request were used in every run. Each row is one cold-prompt measurement after model load; timing is request arrival through first streamed text. Every run emitted `The` first. The next 1,491-token prompt reused 1,475 tokens.

| Prefill rows | First text, seconds | Continued turn, seconds |
| ---: | ---: | ---: |
| 0, serial control in this checkout | 54.094 | 0.750 |
| 4 | 43.625 | 0.672 |
| 8 | 41.781 | 0.750 |
| 16 | 41.063 | 0.890 |
| 16, confirmation | 40.656 | 0.906 |
| 32 | 53.641 | 0.937 |

The two 16-row runs averaged 40.860 seconds, about 24% less than the same-checkout serial control. The previously saved main checkout measurement was 53.969 seconds. These are few samples; the 8-versus-16 difference is small enough to need more trials before treating it as a stable ranking. At 32 rows the benefit disappeared on this workload.

## Correctness and limits

Tiny hybrid-model tests compare the chunked path with serial prefill at 4, 8, 16, and 32-row boundaries. They exercise one and two Arc devices, nonzero LoRA adapter weights, fused and separate LoRA projection, prompt checkpoint reuse, generation, and exact next-token logits. All 16 cases passed.

The batch embedding kernel is part of the training-only OpenCL program, so this prototype still embeds and copies each token separately. DeltaNet and full attention still launch per token. The gain comes mainly from larger projection launches and fewer inter-device transfers. Cancellation is checked between chunks, which may defer cancellation by up to one chunk of prefill. Additional quantizations and longer contexts should be tested before generalizing the GUI default.
