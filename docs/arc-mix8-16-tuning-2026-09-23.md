# Arc `mix8_16` training tuning (2026-09-23)

## Measurement

All full-shape runs used `training.transformer.json`: two Arc B580 GPUs,
batch 16, accumulation 8, sequence 2048, 32 layers, width 512, 16 heads,
91,031,788 parameters, and 262,144 tokens per optimizer update. The Arc
Transformer probe uses fixed synthetic tokens, one warmup and two measured
updates; its reported throughput derives from the median update time. It
excludes tokenization, dataset and checkpoint I/O. Switch A/B pairs use the
same binary and configuration hashes; batch and quantization-block trials
intentionally change the configuration. Short runs distinguish changes of hundreds
of milliseconds, but changes around ten milliseconds are treated as noise.

| Candidate | A/B result | Decision |
| --- | ---: | --- |
| Optimizer: skip unnecessary clip kernels, share Nesterov scratch, reuse its squared-norm reduction | 14,151.5 → 14,179.3 tok/s; clip 18.5 → 13.5 ms and optimizer 192.0 → 161.6 ms/update | Enable |
| Pack attention probability and derivative as BF16 pairs for backward dQ and dK/dV | 14,179.3 → 14,373.3 tok/s | Enable for `mix8_16` T2048/D32 |
| Store decoded QKV activation as physical BF16 and feed ordered FP32 attention products | 14,174.6 → 14,362.5 tok/s; peak allocated memory per GPU reduced 96 MiB | Enable for `mix8_16` T2048/D32 |
| Combine the two attention paths | 14,369.5 → 14,548.4 tok/s; peak allocated memory per GPU reduced 96 MiB | Enable |
| Fuse packed microbatch-gradient accumulation | 14,373.3 → 14,381.0 tok/s when tested alone with packed attention | No meaningful whole-update gain; keep optional |
| Direct BF16 cross-GPU gradient reduction | 14,373.3 → 14,374.3 tok/s with packed attention | No meaningful whole-update gain; keep optional |
| Pipeline GPU 1 gradient read with GPU 0 upload | 14,373.3 → 14,374.1 tok/s with packed attention | No meaningful whole-update gain; keep optional |
| Batch 32, accumulation 4, same effective tokens | 14,153.5 → 13,663.8 tok/s | Keep batch 16, accumulation 8 |
| BFP8 quantization block 128 | 14,153.5 → 13,858.8 tok/s | Keep block 32 |

The final same-binary, same-configuration control/default comparison used one
warmup and four measured updates per run: **14,156.1 → 14,555.2 tok/s**, or
**18,518.0 → 18,010.3 ms/update** (+2.82% throughput). Both runs have matching
Core, Arc and benchmark binary hashes. The combined path allocates 4,555.9 MiB
on GPU 0 and 4,021.7 MiB on GPU 1 at peak, down from 4,651.9 and 4,117.7
MiB in its paired control. These are backend allocation counters, not total
driver VRAM reservations.

The physical BF16 QKV path preserves forward output and two accumulated
attention gradients bitwise in the focused Arc test. Its full-shape synthetic
loss and gradient norm also match the QKV-off path. The packed P/dS path
preserves forward output bitwise. Across two backward accumulations its
relative gradient RMS difference was 0.35–0.39%; the full-shape third loss
was 9.313022 without packing and 9.313025 with packing. The combined path
matched packed attention alone bitwise in its focused test and produced the
same full-shape loss and norm. This is a `mix8_16` numerical tradeoff and is
not enabled in other precision modes or unsupported attention shapes.

Artifacts: `benchmark-results/arc-mix8-16-tuning-{control,optimizer-only,attention,attention-fused-gradient,attention-direct-reduction,attention-pipeline}-20260923.json`,
`benchmark-results/arc-mix8-16-qkv-{control,bf16}-20260923.json`, and
`benchmark-results/arc-mix8-16-combined-{control,bf16-qkv}-20260923.json`,
and `benchmark-results/arc-mix8-16-final-{control,default}-20260923.json`.
