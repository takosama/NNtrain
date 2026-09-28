"""Regenerate small independent GGUF/linear references; no NNtrain code is imported.

python -m pip install gguf==0.19.0 numpy==2.3.5
python NNtrain.Core.Tests/Fixtures/QwenGgufReference/generate.py

Adapted from the user's gen (1).py, compare (1).py, and cl_check (2).py.
The official llama.cpp gguf-py writer/dequantizer provides all tensor references;
linear outputs use float64 products and sums, rather than the C# decoder/kernels.
"""

import base64
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import gguf
import numpy as np
from gguf import GGMLQuantizationType as T, quants


ROOT = Path(__file__).resolve().parent
SEED = 20260925
ROWS, INPUT_WIDTH, OUTPUT_WIDTH = 5, 512, 7
rng = np.random.default_rng(SEED)


def encoded(array, dtype):
    return base64.b64encode(np.asarray(array, dtype=dtype).tobytes()).decode("ascii")


def k_blocks(kind, subnormal):
    size = 144 if kind == T.Q4_K else 210
    blocks = OUTPUT_WIDTH * INPUT_WIDTH // 256
    data = rng.integers(0, 256, (blocks, size), dtype=np.uint8)
    # Explicit half bit patterns make minimum/maximum subnormals reproducible.
    half_bits = np.array([0x0001, 0x0002, 0x0100, 0x0200, 0x03FF], dtype="<u2")
    for offset in ([0, 2] if kind == T.Q4_K else [208]):
        scales = rng.uniform(1e-4, 2e-2, blocks).astype("<f2").view("<u2")
        if subnormal:
            scales[:] = np.resize(half_bits, blocks)
        else:
            scales[::3] = np.resize(half_bits, scales[::3].size)
        data[:, offset:offset + 2] = scales.view(np.uint8).reshape(-1, 2)
    return data.reshape(OUTPUT_WIDTH, -1)


def main():
    payloads = []
    for kind in (T.Q4_K, T.Q6_K):
        for subnormal in (False, True):
            name = f"{kind.name.lower()}_{'subnormal' if subnormal else 'mixed'}.weight"
            payloads.append((name, kind, k_blocks(kind, subnormal)))
    # Signed zero, denormals, normal boundaries and finite extremes.
    f32_bits = np.array([0, 0x80000000, 1, 0x80000001, 0x007FFFFF,
                         0x00800000, 0x3F800000, 0xBF800000, 0x7F7FFFFF], dtype="<u4")
    f16_bits = np.array([0, 0x8000, 1, 0x8001, 2, 0x8002, 0x0100, 0x8100,
                         0x03FF, 0x83FF, 0x0400, 0x8400, 0x3C00, 0xBC00,
                         0x7BFF, 0xFBFF], dtype="<u2")
    payloads.extend([
        ("f32_edges", T.F32, f32_bits.view("<f4")),
        ("f16_edges", T.F16, f16_bits.view("<f2")),
    ])

    path = ROOT / "reference.gguf"
    writer = gguf.GGUFWriter(path, "qwen2")
    writer.add_name("NNtrain independent small K-quant reference")
    for name, kind, payload in payloads:
        writer.add_tensor(name, payload, raw_dtype=kind)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    x = rng.normal(0, 0.1, (ROWS, INPUT_WIDTH)).astype("<f4")
    bias = rng.normal(0, 0.01, OUTPUT_WIDTH).astype("<f4")
    records = []
    # Read the file back with gguf-py as well: file layout and tensor shape are
    # independently established, not inferred from NNtrain's writer/reader.
    for tensor in gguf.GGUFReader(path).tensors:
        decoded = quants.dequantize(tensor.data, tensor.tensor_type).astype("<f4")
        record = {
            "name": tensor.name,
            "type": int(tensor.tensor_type),
            "shape": [int(n) for n in tensor.shape],
            "payloadBytes": int(tensor.n_bytes),
            "decodedFloat32Base64": encoded(decoded, "<f4"),
        }
        if tensor.tensor_type in (T.Q4_K, T.Q6_K):
            # Keep subnormal-only outputs sensitive to decoding underflow:
            # a large bias must not hide a zeroed subnormal contribution.
            used_bias = np.zeros_like(bias) if "subnormal" in tensor.name else bias
            weight = decoded.reshape(OUTPUT_WIDTH, INPUT_WIDTH).astype(np.float64)
            # Explicit float64 reduction, independent of BLAS implementation.
            result = np.sum(x.astype(np.float64)[:, None, :] * weight[None, :, :], axis=2)
            result += used_bias.astype(np.float64)
            record["biasFloat32Base64"] = encoded(used_bias, "<f4")
            record["linearFloat64Base64"] = encoded(result, "<f8")
        records.append(record)

    manifest = {
        "provenance": {
            "project": "https://github.com/ggml-org/llama.cpp/tree/master/gguf-py",
            "ggufVersion": version("gguf"),
            "numpyVersion": np.__version__,
            "seed": SEED,
            "userSourceSha256": {
                "gen (1).py": "bb6757f8ba692dc938d248430103349491507b772fa62461af90202fb7bc3b7a",
                "compare (1).py": "e98cb5c692adfb7a3eeabe054fb286690b64b58f26dc635972d893af86cff1f6",
                "cl_check (2).py": "9bc0a1b07524fe6af3b85fae3167bab4c4967989d6d6aee3c3ae186e4f9dd8ad",
            },
        },
        "ggufSha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "inputRows": ROWS,
        "inputWidth": INPUT_WIDTH,
        "outputWidth": OUTPUT_WIDTH,
        "inputFloat32Base64": encoded(x, "<f4"),
        "tensors": records,
    }
    (ROOT / "expected.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {path.name}: {path.stat().st_size} bytes, {len(records)} tensors")


if __name__ == "__main__":
    main()
