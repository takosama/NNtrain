"""Generate PQ2_0/PTQ1_0 fixtures using pinned official native CPU dequantizers.

Only GGUF headers and six encoded blocks per model are read. No GPU is used.
The Python scalar decoder below audits results; it never supplies expected values.
"""
import argparse
import base64
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess

COMMIT = "adfffbe41b2cabcd51fff326ab045662265062bb"
REPOSITORY = "https://github.com/PrismML-Eng/llama.cpp"
TAG = "prism-b10743-adfffbe"
ZIP_URL = f"{REPOSITORY}/releases/download/{TAG}/llama-{TAG}-bin-win-cpu-x64.zip"
ZIP_SHA256 = "d0b3016c9cc4bc1385de68be034adee570277ba952dd94292ba3888b7f18cc44"
DLL_SHA256 = "a1a54d932be6164be2d72278ce32891826f8f663e58661b620f1d3f6e4feb42f"
FORMATS = {
    142: ("PQ2_0", 34, 0, "dequantize_row_pq2_0"),
    143: ("PTQ1_0", 28, 26, "dequantize_row_ptq1_0"),
}
HALVES = [
    ("positive-zero", 0x0000), ("negative-zero", 0x8000),
    ("min-subnormal", 0x0001), ("negative-min-subnormal", 0x8001),
    ("max-subnormal", 0x03ff), ("negative-max-subnormal", 0x83ff),
    ("min-normal", 0x0400), ("negative-min-normal", 0x8400),
    ("below-one", 0x3bff), ("negative-below-one", 0xbbff),
    ("one", 0x3c00), ("negative-one", 0xbc00),
    ("max-finite", 0x7bff), ("negative-max-finite", 0xfbff),
]


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_exact(f, n):
    result = f.read(n)
    if len(result) != n:
        raise ValueError("Truncated GGUF")
    return result


def u32(f):
    return struct.unpack("<I", read_exact(f, 4))[0]


def u64(f):
    return struct.unpack("<Q", read_exact(f, 8))[0]


def gguf_string(f):
    return read_exact(f, u64(f)).decode("utf-8")


def skip_value(f, typ):
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4,
             7: 1, 10: 8, 11: 8, 12: 8}
    if typ in sizes:
        f.seek(sizes[typ], 1)
    elif typ == 8:
        f.seek(u64(f), 1)
    elif typ == 9:
        item_type, count = u32(f), u64(f)
        if item_type in sizes:
            f.seek(sizes[item_type] * count, 1)
        else:
            for _ in range(count):
                skip_value(f, item_type)
    else:
        raise ValueError(f"Unsupported GGUF metadata type: {typ}")


def directory(model):
    with model.open("rb") as f:
        magic, version, tensor_count, kv_count = struct.unpack("<4sIQQ", read_exact(f, 24))
        if magic != b"GGUF" or version != 3:
            raise ValueError("Expected little-endian GGUF v3")
        alignment = 32
        for _ in range(kv_count):
            name, typ = gguf_string(f), u32(f)
            if name == "general.alignment":
                if typ != 4:
                    raise ValueError("Expected UINT32 alignment")
                alignment = u32(f)
            else:
                skip_value(f, typ)
        tensors = []
        for _ in range(tensor_count):
            name, ndim = gguf_string(f), u32(f)
            shape = [u64(f) for _ in range(ndim)]
            tensors.append({"name": name, "shape": shape, "type": u32(f), "offset": u64(f)})
        data_offset = (f.tell() + alignment - 1) // alignment * alignment
        # Small directory hash identifies offsets without hashing multiple GB of weights.
        f.seek(0)
        header_sha = hashlib.sha256(read_exact(f, data_offset)).hexdigest()
    return tensors, data_offset, header_sha


def audit_scalar(typ, encoded):
    d = struct.unpack_from("<e", encoded, FORMATS[typ][2])[0]
    if typ == 142:
        return [(((encoded[2 + j // 4] >> (2 * (j % 4))) & 3) - 1) * d for j in range(128)]
    values = []
    for j in range(128):
        if j < 80:
            b, n = encoded[j % 16], j // 16
        elif j < 120:
            b, n = encoded[16 + (j - 80) % 8], (j - 80) // 8
        else:
            b, n = encoded[24 + (j - 120) % 2], (j - 120) // 2
        q = (b * (3 ** n)) & 255
        values.append((((q * 3) >> 8) - 1) * d)
    return values


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--dll", type=Path, default=here / "_native/cpu/ggml-base.dll")
    parser.add_argument("--pq-model", type=Path, required=True)
    parser.add_argument("--ptq-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=here / "prism-quant-blocks.json")
    args = parser.parse_args()
    dll_path = args.dll.resolve()
    if sha256(dll_path) != DLL_SHA256:
        raise ValueError("Native DLL differs from the pinned official CPU release")
    version = subprocess.run([str(dll_path.parent / "llama-cli.exe"), "--version"],
                             capture_output=True, text=True, check=True)
    version_text = (version.stdout + version.stderr).strip()
    if "commit adfffbe41" not in version_text:
        raise ValueError("Native version does not identify the pinned commit")
    dll_directory = os.add_dll_directory(str(dll_path.parent))
    dll = ctypes.CDLL(str(dll_path))
    dll.ggml_type_size.argtypes = [ctypes.c_int]
    dll.ggml_type_size.restype = ctypes.c_size_t
    dll.ggml_blck_size.argtypes = [ctypes.c_int]
    dll.ggml_blck_size.restype = ctypes.c_int64
    functions = {}
    for typ, (_, size, _, symbol) in FORMATS.items():
        assert dll.ggml_type_size(typ) == size
        assert dll.ggml_blck_size(typ) == 128
        fn = getattr(dll, symbol)
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
        fn.restype = None
        functions[typ] = fn
    cases = []

    def add(typ, label, encoded, details):
        name, size, _, _ = FORMATS[typ]
        assert len(encoded) == size
        native_input = ctypes.create_string_buffer(bytes(encoded), size)
        native_output = (ctypes.c_float * 128)()
        functions[typ](native_input, native_output, 128)
        decoded = list(native_output)
        assert all(math.isfinite(v) for v in decoded)
        decoded_bytes = ctypes.string_at(native_output, 128 * 4)
        audit_bytes = struct.pack("<128f", *audit_scalar(typ, encoded))
        assert decoded_bytes == audit_bytes, (label, "native/scalar bit mismatch")
        # JSON numeric parsing must also preserve the signed-zero expected bits.
        roundtrip = json.loads(json.dumps(decoded, allow_nan=False))
        assert struct.pack("<128f", *roundtrip) == decoded_bytes
        cases.append({
            "id": label, "type": typ, "quantName": name, "blockBytes": size,
            "elementCount": 128, "encoded": base64.b64encode(encoded).decode("ascii"),
            "decoded": decoded,
            "decodedFloat32LittleEndianSha256": hashlib.sha256(decoded_bytes).hexdigest(),
            "source": details,
        })

    # Four rotations cover every two-bit code in each of the 128 PQ2 lanes.
    pq_seen = [set() for _ in range(128)]
    for rotation in range(4):
        encoded = bytearray(34)
        struct.pack_into("<H", encoded, 0, 0x3c00)
        for lane in range(128):
            code = (lane + rotation) % 4
            encoded[2 + lane // 4] |= code << (2 * (lane % 4))
            pq_seen[lane].add(code)
        add(142, f"all-codes-pq2-{rotation}", encoded,
            {"kind": "synthetic", "rotation": rotation, "coverage": "all four codes at every lane"})
    assert all(v == set(range(4)) for v in pq_seen)

    # Latin coverage uses 256 blocks instead of duplicating 256 blocks per field.
    # Every one of the 24 qs and 2 qh byte positions gets every value 0..255.
    ptq_seen = [set() for _ in range(26)]
    for sweep in range(256):
        encoded = bytearray(28)
        for position in range(26):
            encoded[position] = (sweep + position * 37) & 255
            ptq_seen[position].add(encoded[position])
        struct.pack_into("<H", encoded, 26, 0x3c00)
        add(143, f"all-bytes-ptq1-{sweep:03d}", encoded,
            {"kind": "synthetic", "sweep": sweep, "payload": "(sweep + bytePosition * 37) mod 256"})
    assert all(v == set(range(256)) for v in ptq_seen)

    for typ, (_, size, scale_offset, _) in FORMATS.items():
        for label, half_bits in HALVES:
            encoded = bytearray((position * 37 + 83) & 255 for position in range(size))
            struct.pack_into("<H", encoded, scale_offset, half_bits)
            add(typ, f"half-{typ}-{label}", encoded,
                {"kind": "synthetic", "halfBits": f"0x{half_bits:04x}",
                 "payload": "(bytePosition * 37 + 83) mod 256"})

    models = []
    for typ, model in ((142, args.pq_model), (143, args.ptq_model)):
        tensors, data_offset, header_sha = directory(model)
        tensors = [t for t in tensors if t["type"] == typ]
        assert len(tensors) == 402, (model.name, len(tensors))
        by_name = {t["name"]: t for t in tensors}
        selection = [by_name["output.weight"], by_name["token_embd.weight"],
                     by_name["blk.0.attn_qkv.weight"], by_name["blk.0.ssm_out.weight"],
                     by_name["blk.0.ffn_down.weight"], tensors[-1]]
        size = FORMATS[typ][1]
        with model.open("rb") as f:
            for index, tensor in enumerate(selection):
                width, rows = tensor["shape"]
                assert width % 128 == 0
                row = (0, rows // 2, rows - 1)[index % 3]
                block = (0, width // 256, width // 128 - 1)[index % 3]
                offset = data_offset + tensor["offset"] + (row * (width // 128) + block) * size
                f.seek(offset)
                encoded = read_exact(f, size)
                add(typ, f"model-{typ}-{index:02d}", encoded,
                    {"kind": "model", "modelFilename": model.name, "tensor": tensor["name"],
                     "tensorShape": tensor["shape"], "row": row, "blockInRow": block,
                     "absoluteFileOffset": offset})
        models.append({"type": typ, "filename": model.name, "bytes": model.stat().st_size,
                       "dataOffset": data_offset, "headerAndPaddingSha256": header_sha,
                       "quantizedTensors": len(tensors), "sampledBlocks": len(selection),
                       "note": "Only header/directory and listed blocks read; full model not hashed or executed."})
    artifact = {
        "schemaVersion": 1,
        "description": "Encoded 128-value blocks decoded by the pinned official Prism native CPU C functions.",
        "reference": {
            "officialSourceRepository": REPOSITORY, "officialSourceCommit": COMMIT,
            "releaseTag": TAG, "releaseAssetUrl": ZIP_URL, "releaseAssetSha256": ZIP_SHA256,
            "nativeVersionOutput": version_text, "nativeDllName": dll_path.name,
            "nativeDllSha256": DLL_SHA256, "license": "MIT; see LICENSE",
            "symbols": [v[3] for v in FORMATS.values()],
            "sourceFiles": {
                "ggml/src/ggml-quants.c": "46dd353f30e7f1c42c7d82d88d91fad64dbe5d6e680b234de25355d9fb482d13",
                "ggml/src/ggml-common.h": "96539c9bd8e4544692bf242951180edaa2fd5775864fbc3e833f44111b96b934",
            },
            "nativeMatchesScalarAuditBitwise": True,
        },
        "coverage": {"pq2CodesPerLane": 4, "pq2Lanes": 128, "ptq1BytePositions": 26,
                     "ptq1ValuesPerPosition": 256, "finiteHalfCasesPerFormat": len(HALVES),
                     "signedZeroJsonRoundtripBitwise": True,
                     "excluded": ["infinity/NaN half scales", "whole-model execution", "Hadamard/permutation transforms"]},
        "models": models,
        "counts": {str(typ): sum(c["type"] == typ for c in cases) for typ in FORMATS},
        "cases": cases,
    }
    # Keep each 128-value decoded array on one line instead of 128 repetitive lines.
    formatted = json.dumps(artifact, ensure_ascii=False, allow_nan=False, indent=2)
    import re
    formatted = re.sub(r'("decoded": )\[\s*([^\]]+)\]',
                       lambda m: m[1] + "[" + " ".join(m[2].split()) + "]", formatted)
    args.output.write_text(formatted + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "bytes": args.output.stat().st_size,
                      "counts": artifact["counts"], "coverage": artifact["coverage"],
                      "sha256": sha256(args.output)}, indent=2))
    dll_directory.close()


if __name__ == "__main__":
    main()
