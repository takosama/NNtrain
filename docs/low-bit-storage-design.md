# Low-bit storage design

`TensorDType` reserves Float8 E4M3FN, Float8 E5M2, Float4, Float2, and
`Ternary1Bit58` now. Reserving the names is deliberately not the same as
enabling them: the current Tensor runtime has real codecs and kernels only for
Float32 and Float16. Constructing a Tensor with a reserved dtype still fails
before allocating storage.

This document defines the boundary that later codecs must implement without
changing the Tensor, checkpoint, or module-state contracts again.

## Descriptor boundary

`TensorStorageDescriptor` combines a `TensorDType` with optional
`TensorStorageMetadata`:

| Encoding | Value payload | Sidecar metadata |
| --- | --- | --- |
| `Native` | One native element per logical value | none |
| `Packed` | Fixed-width codes packed into 8/16/32/64-bit units | none |
| `BlockQuantized` | Native-width codes | one positive scale per block, optional affine zero point |
| `PackedBlockQuantized` | Packed fixed-width codes | one positive scale per block, optional affine zero point |

`TensorPackingMetadata` specifies the physical code width, packing-unit width,
and bit order. `TensorQuantizationMetadata` specifies the block size, the
scales, and zero points where affine quantization requires them. Descriptor
validation checks shape-independent format rules as well as the exact count of
blocks for a tensor length. `GetPayloadByteLength`,
`GetAuxiliaryByteLength`, and `GetTotalByteLength` keep value bytes and
quantization sidecars explicit during allocation planning.

The descriptor only describes a format. `IsSupportedByCurrentRuntime` remains
false until a corresponding codec, forward kernel, backward policy, and
serialization implementation exist.

## Canonical planned layouts

| dtype | Initial physical representation | Typical metadata |
| --- | --- | --- |
| Float32 | 32-bit native IEEE value | none |
| Float16 | 16-bit native IEEE value | none |
| Float8 E4M3FN / E5M2 | one native byte per value | optional block scale |
| Float4 | 4-bit packed code | optional block scale/zero point |
| Float2 | 2-bit packed code | optional block scale/zero point |
| Ternary1Bit58 | 2-bit packed code for `{-1, 0, +1}` | optional ternary scale per block |

Ternary values contain `log2(3) = 1.5849625...` bits of information, but a
portable fixed-width first implementation uses a two-bit code. The descriptor
therefore records both the two physical bits and the `log2(3)` effective-bit
target. This prevents a misleading claim that a byte-aligned 2-bit payload is
already an entropy-coded 1.58-bit payload.

## Runtime and optimizer policy

Low-bit dtypes will be storage formats, not accumulator formats. The intended
path is:

1. Load/dequantize a block to Float32 SIMD lanes.
2. Perform forward arithmetic and reductions in Float32.
3. Accumulate gradients and optimizer state in Float32.
4. Update a Float32 master parameter where the parameter is trainable.
5. Re-encode the storage payload once per optimizer update.

This is the same master-weight boundary already used by Float16. It lets
future low-bit training preserve a stable optimizer contract while inference
can choose a separate quantized-only parameter path. `ushort`/`short` are
valid containers for packed bit fields; they are not substitutes for IEEE
Float16 arithmetic.

## Module state and SafeTensors

`ModuleParameterState.StorageMetadata` is optional and defaults to `null`.
`null` means raw native storage, so existing JSON states deserialize exactly as
before and current Float32/Float16 state captures retain their old shape and
value representation. A non-raw descriptor is not silently applied by the
current `Module` runtime.

The SafeTensors writer continues to emit only standard `F32` and `F16`
descriptors. It accepts omitted/raw metadata and rejects non-raw metadata with
a clear error rather than dropping scales, zero points, or packing rules. This
keeps current files byte-compatible and prevents lossy checkpoints while the
future codecs are unfinished. A low-bit SafeTensors implementation must add
both a standard dtype mapping (where one exists) and explicit, versioned
sidecar entries for the descriptor metadata and encoded bytes before enabling
save/load.

## Implementation order

1. Add each codec with exhaustive bit-pattern and scale/zero-point tests.
2. Add SIMD load/dequantize and pack/store kernels with scalar fallback.
3. Enable the dtype in Tensor construction only after every public operation
   has an explicit dispatch or rejection rule.
4. Add optimizer master-weight synchronization and numerical regression tests.
5. Add versioned checkpoint and SafeTensors payload support.
6. Benchmark separately from algorithmic changes, comparing F32, F16, and the
   new storage format in Debug and Release.

No step above changes existing Float32/Float16 semantics or treats a reserved
dtype as an implemented one.
