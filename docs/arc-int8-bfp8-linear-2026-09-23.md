# Arc signed INT8 XMX BFP8 Linear candidate (2026-09-23)

`mix8_16` stores block32 BFP8 weights and activations as signed bytes with a
separate FP32 scale for each block. This is not native FP8 floating point. The
experimental `Mix8_16Int8Linear` option packs those bytes directly into SG16
DPAS panels and calls `intel_sub_group_i8_i8_matrix_mad_k32` for the forward
Linear projection. Every K32 INT32 partial dot is multiplied by its input and
weight scales before FP32 accumulation. Bias, ReLU, and block32 BFP8 output
publication follow the existing fused Linear boundary. Backward dX/dW retains
the BF16 matrix operand route. The hardware intrinsic and SG16 signature are
specified in the [Khronos Intel OpenCL subgroup matrix extension](https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_matrix_multiply_accumulate.html).

The route is opt-in and applies only to SG16 XMX, block32 BFP8 operands, a
block32 BFP8 result, K divisible by 32 and at most 2048, N divisible by 32,
and M at least 128. The INT8 kernels are compiled only when enabled. A CLI
probe can compare `--mix8-16-int8-linear on|off` in the same binary.

Focused Arc GPU tests: four passed. Against the established BF16 XMX route,
forward output relative RMS was 0.13–0.15% across three shapes, including a
row-tail and an 8192-row panel boundary. dX/dW matched exactly without ReLU;
with ReLU their relative RMS was 0.23% and 1.18% respectively. The `mix8_32`
mode did not dispatch to the INT8 kernel.

Full `training.transformer.json` synthetic two-B580 training, same Release
binary and config hash, one warmup plus two measured updates:

| Route | Median update | Throughput | GPU 0 peak backend allocation |
| --- | ---: | ---: | ---: |
| INT8 off | 18,017.92 ms | 14,549.1 tok/s | 4,555.9 MiB |
| INT8 on | 18,532.11 ms | 14,145.4 tok/s | 4,568.6 MiB |

The INT8 candidate was 2.85% slower and used 12.7 MiB more peak backend
allocation on GPU 0. On one measured update, the baseline BF16 fused forward
kernel plus A/B packing took about 1,105 ms of summed GPU kernel events per
GPU; INT8 forward plus packing took about 1,530 ms. These event sums do not
equal synchronized wall time, but explain the observed direction. The INT8
kernel uses a 32-column tile, while the measured BF16 fused route uses a
64-column tile. The default remains BF16 for this shape because it is faster.

Artifacts: `benchmark-results/arc-mix8-16-int8-linear-{off,on}-20260923.json`.
