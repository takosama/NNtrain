# Precision modes

Training JSON uses `precisionMode`. Its accepted values and contracts are:

| mode | parameter storage | retained training state | computation |
|---|---|---|---|
| `float32` | IEEE binary32 | Float32 | Float32 |
| `bfloat16` | BFloat16 | BFloat16 optimizer state | BFloat16 operands with Float32 accumulation and reductions where required |
| `mix16_32` | BFloat16 | Float32 gradients, master weights and optimizer state | BFloat16 GEMM with Float32 accumulation, normalization and loss |
| `bfp8` | signed Int8 with one tensor-wide Float32 scale | no master weight | backend-specific BFP8 path |
| `mix8_32` | signed Int8 with a Float32 scale per block | Float32 gradients, master weights and optimizer state | BFloat16 GEMM operands with Float32 accumulation, normalization and loss |
| `mix8_16` | signed Int8 with a Float32 scale per block | physically BFloat16 gradients, master weights and optimizer state | Int8 weight GEMM preferred, BFloat16 weight GEMM allowed when faster; BFP8/BFloat16 activations and non-weight kernels selected for speed |

`mix8_16` uses **BFloat16**, not IEEE Float16. Its default tensor storage
remains block BFP8, but its policy permits BFloat16 activation publication.
The Arc backend may choose Int8 or BFloat16 weight GEMM by measured speed;
Int8 is the preferred route when both perform similarly. Non-weight work may
use BFP8, BFloat16 or Float32 internal formats and may change the order of
Float32 operations to improve speed. Native Int32 DPAS partial accumulators
and Float32 reduction scratch are implementation details; retained gradients,
master weights and optimizer moments are BFloat16 values. These permissions
describe the numeric contract; each backend must explicitly implement a path
before selecting it. `mix8_16` currently targets Arc Transformer execution.

For the measured two-B580 Transformer shape, Arc selects BF16 weight GEMM
over the slower native INT8 Linear candidate, keeps fused Linear outputs in
BFP8, and publishes eligible other activations as BF16. Causal T2048/D32
backward uses a saved BF16 attention output and fused dP/dS with FP32 FMA
scratch. These are measured backend choices under the speed policy, rather
than requirements to preserve FP32 arithmetic order. See the
[candidate and final measurements](arc-mix8-16-speed-policy-2026-09-23.md).

Arc also combines normal/transposed gradient packing, publishes normalization
results directly as BF16, and caches rounded loss logits when device headroom
permits. Normalization uses parallel FP32 reductions with an anchored mean;
its rounding can differ from the older serial reduction. These choices follow
the non-weight speed policy and keep retained training state in BF16. See the
[non-attention profiles and comparisons](arc-nonattention-tuning-2026-09-23.md).

The measured T2048/D32 path also uses `native_exp` for softmax and its saved
replay, packs BF16 K/V and P/dS pairs in shared local memory, and loads K32
once in fused dP/dS. The packed tiles preserve their FP32 FMA order;
`native_exp` is an approximation verified on B580. Eligible width-512 residual
normalization fuses gamma/beta partials with dX, changing parameter-gradient
summation order. These paths are restricted to `mix8_16` and leave the retained
BF16 state and communication format unchanged. See the
[overall profiles, accuracy checks and A/B/A measurements](arc-overall-tuning-2026-09-24.md).

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

For CUDA two-GPU training, BF16 gradient buckets are enabled for `bfloat16`
and `mix16_32`. `float32` retains Float32 gradient communication so selecting
it does not silently reduce communication precision. Arc `mix8_16` uses
physically BFloat16 retained gradients on each device.
