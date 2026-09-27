# Native IQ/Q5 dequantization fixtures

Generated on 2026-09-27. No NNtrain decoder is used to generate expected values.
No GPU inference, full-model expansion, model copy, or repository edits were performed.

## Files

- `iq-quant-blocks.json`: 139 independent 256-value blocks and their expected Float32 values.
- `generate.py`: standard-library-only Python generator using native llama.cpp exports through `ctypes`.
- `LICENSE.llama.cpp`: upstream MIT license, retained for source/table provenance.

## Independent reference

The generator calls the installed `ggml-base.dll` from LM Studio CPU backend
`llama.cpp-win-x86_64-avx2-2.41.0` directly:

| GGML type | Format | Encoded bytes | Native function | Cases |
|---|---|---:|---|---:|
| 13 | Q5_K | 176 | `dequantize_row_q5_K` | 25 |
| 21 | IQ3_S | 110 | `dequantize_row_iq3_s` | 45 |
| 22 | IQ2_S | 82 | `dequantize_row_iq2_s` | 69 |

Every case contains exactly 256 decoded values. Both encoded sizes and the
256-element block size are checked against exported `ggml_type_size` and
`ggml_blck_size` before any decode call.

The bundled executable reports `0.4.1-dev (build 1, commit b49650a)`, built by
MSVC 19.44.35228.0 for x64. Its reported commit is **not** asserted to be the
same as the official upstream source revision below. The native DLL SHA256 is:

```text
85555aa68330363c10004e1415021cdfc2399e835a4736b9e1b11d6cd5a4bfd4
```

Official source reference is pinned to
[`95887577ab5fead779581a7030a83c7752ff3234`](https://github.com/ggml-org/llama.cpp/tree/95887577ab5fead779581a7030a83c7752ff3234).
The [block layouts and IQ tables](https://github.com/ggml-org/llama.cpp/blob/95887577ab5fead779581a7030a83c7752ff3234/ggml/src/ggml-common.h)
and [dequantization functions](https://github.com/ggml-org/llama.cpp/blob/95887577ab5fead779581a7030a83c7752ff3234/ggml/src/ggml-quants.c)
are MIT licensed. The downloaded common-header SHA256 is
`0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c`.

For **every one of the 1024 IQ2_S and 512 IQ3_S grid entries**, the generator
also verifies that the native decoded result equals the corresponding pinned
upstream table values with the selected scale/sign patterns. Thus lookup-table
compatibility between the bundled native build and the pinned upstream source
is checked rather than assumed. Expected fixture values still come exclusively
from the native DLL; the table audit never supplies fixture expected values.

## Coverage

- 60 real model blocks from `Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf`.
- For each type: the first, middle, and last available tensors; their first,
  middle, and last rows; first and last blocks in those rows; plus six seeded
  random blocks. Q5_K has just one tensor (`output.weight`).
- 39 synthetic half/packed-bit cases: positive/negative zero, positive/negative
  minimum half subnormal, largest half subnormal, minimum half normal,
  positive/negative one, largest finite half, and packed `00`, `FF`, `55`, `AA`.
- 40 synthetic cases jointly cover all IQ table indices, all sign lanes, and
  scale nibble extremes. IQ2_S uses 32 blocks; IQ3_S uses eight.
- The all-`FF` Q5_K payload with `d=0.5`, `dmin=0.25` is additionally checked
  against the known value `0.5 * 63 * 31 - 0.25 * 63 = 960.75` at all 256 lanes.
- All expected values are finite. NaN/Infinity encodings are intentionally
  excluded because a finite model weight path is being validated.

Sampling seed is `27092026`. Only the selected tiny encoded blocks are read from
the model. The complete model is not hashed; its filename/size and the inventory
SHA256 are recorded, and every real sample carries its exact absolute byte offset.

## JSON schema

Top-level properties are `schemaVersion`, `description`, `reference`, `model`,
`randomSeed`, `counts`, and `cases`.

Each element of `cases` has:

```text
id: string, unique within the artifact
type: integer, one of 13 / 21 / 22
quantName: "Q5_K" / "IQ3_S" / "IQ2_S"
blockBytes: 176 / 110 / 82
elementCount: 256
encoded: base64 of one raw little-endian GGUF block
decoded: array of 256 JSON numbers, each exactly a native Float32 result
decodedFloat32LittleEndianSha256: SHA256 of the native output's 1024 raw bytes
source: real tensor/row/block/offset metadata, or synthetic generation details
```

Use `Convert.FromBase64String(encoded)` in C# and deserialize `decoded` directly
as `float[]`. JSON numbers were serialized from Python floats containing exactly
representable native Float32 values; this preserves their Float32 round trip.
Numerical comparisons may treat positive/negative zero as equivalent, while the
per-case byte hash can be used when preserving the exact zero sign is required.

## Reproduce

Run from the workspace root with the existing model, inventory, source header,
and installed native DLL:

```powershell
& 'C:\Users\takos\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' .\iq-quant-reference\generate.py
```

Inputs can be overridden with `--inventory`, `--dll`, `--source-common`, and
`--output`. The model path is read from the inventory. The generator does not
need an OpenCL device, NNtrain assemblies, or additional Python packages. It
overwrites only the explicitly selected output JSON.

Reproduction of the exact byte hashes requires the same native DLL and the same
model blocks. Other compatible llama.cpp DLL builds can be used for a fresh
reference run, but must be reported with their own binary hash/version.
