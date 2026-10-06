# Official CPU ASR comparison

These scripts are validation tools, not the GUI's runtime. Production ASR remains native C# with optional Intel Arc OpenCL dense operations.

The isolated workspace environment uses official Python 3.13.16 embedded x64, CPU PyTorch 2.14.1+cpu, Transformers 5.18.0 and NeMo 3.0.0. Python/Pip installation SHA256 and resolved package URLs are recorded under `benchmark-results/asr-20261006`. No PATH, launcher, file associations, registry, security settings or user Python directories were changed. `pip check` passed. The downloaded distributions total 264,651,368 bytes for 57 initial dependencies plus 115,070,973 bytes for 88 NeMo additions, excluding the approximately 10.9 MB Python and 1.8 MB pip distributions. Runtime, caches and downloaded archives are ignored by Git.

All scripts accept existing local model directories and PCM16 16 kHz mono files. They force offline Hugging Face access, use four CPU threads and never open a microphone, GPU, LLM server or external inference API. NeMo original archive weights use `torch.load(weights_only=True)` without extracting arbitrary paths. Converted safetensors are another supported input.

`official_nemotron.py` compares original FP32 or FP16-rounded weights with FP32 operations. It uses Japanese `ja-JP`, cached 25/32 mel-frame chunks, lookahead 3 and padded final decoding. `--encoder-only --round-fp16` exports two cached chunks from a 9,120-sample prefix. `official_parakeet.py` uses the official hybrid checkpoint's CTC branch; it does not benchmark TDT. `--round-fp16` retains generated FP32 signal frontend buffers to match C#, while neural and batch-norm weights remain FP16-rounded. `--prefix-samples 9120` exports short frontend/encoder comparison data.

Parakeet frontend parity targets the maintained NeMo 3.0.0 implementation: centered zero STFT padding, normalization over `floor(sample_count / hop)` valid frames, zero terminal frame and the corresponding valid encoder length. The checkpoint originated with an older NeMo release; no claim of exact parity with every historical NeMo frontend is made.

Comparison scripts operate on little-endian float32 exports. Core opt-in `OfficialAsrParityTests` export the matching native CPU or Arc intermediates. Set `NNTRAIN_ASR_EXPORT_FEATURES`, `NNTRAIN_ASR_EXPORT_ENCODER` or `NNTRAIN_ASR_EXPORT_PARAKEET` to `1` only for a deliberate validation run; model/audio/output paths are explicit environment variables in those tests. These checks are excluded from ordinary CPU tests by default.
