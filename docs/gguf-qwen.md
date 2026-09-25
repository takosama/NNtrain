# Qwen2 GGUF import (first stage)

PR #11 adds Qwen2/Qwen2.5 GGUF inspection, decoding, and Intel Arc inference.
The existing Transformer training and generation commands retain their own
configuration and precision settings.

## Generate

```powershell
dotnet run --project NNtrain.Cli -c Release -- qwen-gguf --model C:\models\qwen.gguf --prompt "Hello" --device 0 --max-new-tokens 16
```

The command uses greedy sampling and a single Arc GPU. Large projection matrices
remain in their original Q4_K/Q6_K representation in host memory and VRAM;
activations and the decoded embedding use Float32. The file must contain a
separate quantized `output.weight`. This command does not use `generate.json`
or its two-GPU selection. Its current generation path recomputes the prefix;
the GPT generation KV cache is not a Qwen KV cache.

## Supported scope

- GGUF versions 2 and 3 with `general.architecture = qwen2`.
- Dense import decodes F32, F16, BF16, Q4_K and Q6_K tensors.
- Native quantized inference requires Q4_K/Q6_K projection weights with complete
  256-element blocks per row. Tensor dimensions are checked before payload reads.
- RMSNorm, split-half RoPE, grouped-query attention and SwiGLU are implemented.
- The first GQA kernel supports at most 4096 tokens per forward call and rejects
  longer inputs explicitly. A larger GGUF context value does not remove this
  kernel limit. No-grad inference avoids the quadratic saved-probability buffer.
- Quantized Qwen training is unsupported. The dense model exposes LoRA building
  blocks; this does not add a Qwen LoRA CLI.

## Integration corrections

The merge includes Qwen kernel resource registration and OpenCL identifier fixes, tokenizer split-regex
repair, per-token temporary GPU buffer release, correct FP16 subnormal decoding
in quantized kernels, tensor-operation inventory entries, and failed-reader file
cleanup. Q4_K CPU decoding reuses bounded stack scratch across blocks.

Regression tests use synthetic GGUF vocabularies/tensors, CPU references and small
models on Arc. They do not establish tokenizer/logit parity or performance for a
downloaded pretrained Qwen checkpoint.
