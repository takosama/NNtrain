# Precision modes

Training JSON uses `precisionMode`. Its accepted values and contracts are:

| mode | parameter/activation storage | stable Float32 work |
|---|---|---|
| `float32` | IEEE binary32 | all operations |
| `bfloat16` | bfloat16 | GEMM accumulation, reductions and gradients where required by the CUDA kernels; AdamW moments remain BF16 |
| `mix16_32` | bfloat16 | GEMM accumulation, reductions, LayerNorm statistics, loss, gradients, optimizer state and master weights |

`TensorDType` remains the physical-storage API. Both `bfloat16` and
`mix16_32` map to two-byte `TensorDType.BFloat16` parameter/activation
storage; their model-level precision flag selects BF16 or FP32 optimizer
state. Raw `TensorDType.Float16` is retained as a low-level/legacy IEEE
binary16 path and defaults to the mixed execution contract when used directly.
New checkpoints record both the precision mode and its physical storage dtype
and reject inconsistent pairs.

`modelDType` is accepted only for old configuration files. Legacy
`modelDType: "float16"` maps to `precisionMode: "mix16_32"`. A configuration
must not specify both properties.

For two-GPU training, BF16 gradient buckets are enabled for the two 16-bit
modes. `float32` retains Float32 gradient communication so selecting it does
not silently reduce communication precision.
