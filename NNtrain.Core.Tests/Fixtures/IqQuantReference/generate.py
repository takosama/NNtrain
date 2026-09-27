"""Create small native llama.cpp dequantization fixtures; never loads a model into RAM."""
import argparse
import base64
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import struct
import subprocess

SOURCE_COMMIT = "95887577ab5fead779581a7030a83c7752ff3234"
FORMATS = {13: ("Q5_K", 176, "dequantize_row_q5_K"),
           21: ("IQ3_S", 110, "dequantize_row_iq3_s"),
           22: ("IQ2_S", 82, "dequantize_row_iq2_s")}
SEED = 27092026


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    root = Path(__file__).resolve().parent.parent
    parser.add_argument("--inventory", type=Path, default=root / "qwen35-iq2-inventory.json")
    parser.add_argument("--dll", type=Path, default=Path.home() / ".lmstudio/extensions/backends/llama.cpp-win-x86_64-avx2-2.41.0/ggml-base.dll")
    parser.add_argument("--source-common", type=Path, default=root / "iq-quant-source/ggml-common.h")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "iq-quant-blocks.json")
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text(encoding="utf-8-sig"))
    model = Path(inventory["path"])
    assert model.stat().st_size == inventory["bytes"]
    source = args.source_common.read_text(encoding="utf-8")
    directory_handle = os.add_dll_directory(str(args.dll.parent))
    dll = ctypes.CDLL(str(args.dll))
    dll.ggml_type_size.argtypes = [ctypes.c_int]
    dll.ggml_type_size.restype = ctypes.c_size_t
    dll.ggml_blck_size.argtypes = [ctypes.c_int]
    dll.ggml_blck_size.restype = ctypes.c_int64
    functions = {}
    for typ, (_, size, symbol) in FORMATS.items():
        assert dll.ggml_type_size(typ) == size
        assert dll.ggml_blck_size(typ) == 256
        fn = getattr(dll, symbol)
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
        fn.restype = None
        functions[typ] = fn
    cases = []

    def add(typ, label, encoded, details):
        name, size, _ = FORMATS[typ]
        assert len(encoded) == size
        native_input = ctypes.create_string_buffer(bytes(encoded), size)
        native_output = (ctypes.c_float * 256)()
        functions[typ](native_input, native_output, 256)
        decoded = list(native_output)
        assert all(math.isfinite(v) for v in decoded)
        expected_bytes = ctypes.string_at(native_output, 256 * 4)
        case = {"id": label, "type": typ, "quantName": name, "blockBytes": size,
                "elementCount": 256, "encoded": base64.b64encode(encoded).decode("ascii"),
                "decoded": decoded,
                "decodedFloat32LittleEndianSha256": hashlib.sha256(expected_bytes).hexdigest(),
                "source": details}
        cases.append(case)
        return decoded

    # Actual model blocks from beginning/middle/end tensors and rows, plus seeded random samples.
    rng = random.Random(SEED)
    with model.open("rb") as f:
        for typ, (_, size, _) in FORMATS.items():
            tensors = [t for t in inventory["tensors"] if t["type"] == typ]
            selected = [tensors[i] for i in sorted({0, len(tensors) // 2, len(tensors) - 1})]
            locations = []
            for tensor in selected:
                cols, rows = tensor["shape"]
                assert cols % 256 == 0
                for row in sorted({0, rows // 2, rows - 1}):
                    for block in sorted({0, cols // 256 - 1}):
                        locations.append((tensor, row, block, "edge"))
            for _ in range(6):
                tensor = rng.choice(tensors)
                cols, rows = tensor["shape"]
                locations.append((tensor, rng.randrange(rows), rng.randrange(cols // 256), "random"))
            seen = set()
            for tensor, row, block, label in locations:
                key = (tensor["name"], row, block)
                if key in seen:
                    continue
                seen.add(key)
                cols, rows = tensor["shape"]
                offset = inventory["data_offset"] + tensor["offset"] + (row * (cols // 256) + block) * size
                f.seek(offset)
                encoded = f.read(size)
                add(typ, f"model-{typ}-{len(seen):02d}", encoded,
                    {"kind": "model", "selection": label, "tensor": tensor["name"],
                     "tensorShape": tensor["shape"], "row": row, "blockInRow": block,
                     "absoluteFileOffset": offset})

    # Finite half bit boundaries: signed zeros, min/largest subnormal, min normal,
    # +/-1, largest finite. Random payload exposes all packed fields and sign bits.
    half_cases = [("positive-zero", 0x0000), ("negative-zero", 0x8000),
                  ("min-subnormal", 0x0001), ("negative-min-subnormal", 0x8001),
                  ("max-subnormal", 0x03ff), ("min-normal", 0x0400),
                  ("one", 0x3c00), ("negative-one", 0xbc00), ("max-finite", 0x7bff)]
    for typ, (_, size, _) in FORMATS.items():
        for label, half_bits in half_cases:
            encoded = bytearray(rng.randbytes(size))
            struct.pack_into("<H", encoded, 0, half_bits)
            if typ == 13:
                struct.pack_into("<H", encoded, 2, half_bits)
            add(typ, f"half-{typ}-{label}", encoded,
                {"kind": "synthetic", "halfBits": f"0x{half_bits:04x}", "payload": "seeded-random"})
        for byte in (0x00, 0xff, 0x55, 0xaa):
            encoded = bytearray([byte] * size)
            struct.pack_into("<H", encoded, 0, 0x3800)  # d = 0.5
            if typ == 13:
                struct.pack_into("<H", encoded, 2, 0x3400)  # dmin = 0.25
            decoded = add(typ, f"pattern-{typ}-{byte:02x}", encoded,
                          {"kind": "synthetic", "payloadByte": byte})
            if typ == 13 and byte == 0xff:
                assert all(v == 960.75 for v in decoded)  # 0.5 * 63 * 31 - 0.25 * 63

    def read_grid(name, count, width):
        match = re.search(r"GGML_TABLE_BEGIN\([^,]+,\s*" + name + r",\s*" + str(count) + r"\)(.*?)GGML_TABLE_END", source, re.S)
        assert match
        values = [int(x, 16) for x in re.findall(r"0x([0-9a-fA-F]+)", match.group(1))]
        assert len(values) == count
        return [v.to_bytes(width, "little") for v in values]

    grids = {22: read_grid("iq2s_grid", 1024, 8), 21: read_grid("iq3s_grid", 512, 4)}
    coverage = {}
    for typ in (22, 21):
        _, size, _ = FORMATS[typ]
        vectors = 32 if typ == 22 else 64
        grid = grids[typ]
        covered = set()
        for start in range(0, len(grid), vectors):
            encoded = bytearray(size)
            struct.pack_into("<H", encoded, 0, 0x3c00)  # d = 1
            if typ == 22:
                # Layout: d[2], indices[32], signs[32], high-index bits[8], scales[8].
                for j in range(vectors):
                    index = start + j
                    covered.add(index)
                    encoded[2 + j] = index & 255
                    encoded[66 + j // 4] |= (index >> 8) << (2 * (j % 4))
                    encoded[34 + j] = (0x55, 0xaa, 0x00, 0xff)[j % 4]
                for j in range(8):
                    encoded[74 + j] = (j * 2) | ((15 - j * 2) << 4)
            else:
                # Layout: d[2], indices[64], high-index bits[8], signs[32], scales[4].
                for j in range(vectors):
                    index = start + j
                    covered.add(index)
                    encoded[2 + j] = index & 255
                    encoded[66 + j // 8] |= (index >> 8) << (j % 8)
                for j in range(32):
                    encoded[74 + j] = (0x55, 0xaa, 0x00, 0xff)[j % 4]
                encoded[106:110] = bytes((0x0f, 0x12, 0x78, 0xef))
            decoded = add(typ, f"all-grids-{typ}-{start:04d}", encoded,
                          {"kind": "synthetic", "firstGridIndex": start, "gridCount": vectors})
            # Independent table compatibility audit. Fixture values above always originate
            # from the preexisting native DLL, not this audit or NNtrain code.
            audited = []
            for j in range(vectors):
                if typ == 22:
                    s = encoded[74 + j // 4]
                    scale = (0.5 + ((s >> (4 * ((j % 4) // 2))) & 15)) * 0.25
                    signs = encoded[34 + j]
                    lane_shift = 0
                else:
                    s = encoded[106 + j // 16]
                    scale = 1 + 2 * ((s >> (4 * ((j % 16) // 8))) & 15)
                    signs = encoded[74 + j // 2]
                    lane_shift = (j % 2) * 4
                for lane, magnitude in enumerate(grid[start + j]):
                    sign = -1 if signs & (1 << (lane + lane_shift)) else 1
                    audited.append(float(scale * magnitude * sign))
            assert decoded == audited, (typ, start, "DLL lookup table differs from pinned official source")
        assert covered == set(range(len(grid)))
        coverage[str(typ)] = {"coveredGridIndices": len(covered), "totalGridIndices": len(grid),
                              "nativeMatchesPinnedOfficialTable": True}

    version_result = subprocess.run([str(args.dll.parent / "llama-server.exe"), "--version"],
                                    capture_output=True, text=True, check=True)
    native_version = (version_result.stdout + version_result.stderr).strip()
    artifact = {
        "schemaVersion": 1,
        "description": "One encoded 256-value block per case, decoded by existing native llama.cpp ggml-base.dll exports.",
        "reference": {
            "nativeDistribution": "LM Studio llama.cpp-win-x86_64-avx2 2.41.0",
            "nativeVersionOutput": native_version,
            "nativeDllName": args.dll.name,
            "nativeDllSha256": sha256(args.dll),
            "nativeCommitCaveat": "b49650a is the bundled executable's reported commit; it is not asserted to be the official source commit below.",
            "officialSourceRepository": "https://github.com/ggml-org/llama.cpp",
            "officialSourceCommit": SOURCE_COMMIT,
            "officialCommonHeaderSha256": sha256(args.source_common),
            "license": "MIT",
            "symbols": [v[2] for v in FORMATS.values()],
            "tableCoverage": coverage,
        },
        "model": {"filename": model.name, "bytes": model.stat().st_size,
                  "inventorySha256": sha256(args.inventory),
                  "dataOffset": inventory["data_offset"],
                  "note": "Only listed blocks were read. The complete model was neither loaded nor hashed."},
        "randomSeed": SEED,
        "counts": {str(typ): sum(c["type"] == typ for c in cases) for typ in FORMATS},
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "bytes": args.output.stat().st_size,
                      "counts": artifact["counts"], "tableCoverage": coverage}, indent=2))
    directory_handle.close()


if __name__ == "__main__":
    main()
