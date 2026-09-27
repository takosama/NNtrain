#!/usr/bin/env python3
"""Independently verify an exported LoRA with llama.cpp's official gguf reader.

Requires gguf-py and numpy for verification only. --gguf-package can point to a
local gguf-py source directory (the directory containing the gguf package).
"""

import argparse
import hashlib
import hmac
import json
from pathlib import Path
import struct
import sys


def verify(checkpoint, adapter, expected_step=None):
    from gguf.gguf_reader import GGUFReader

    reader = GGUFReader(adapter)
    with checkpoint.open("rb") as source:
        remaining = checkpoint.stat().st_size - 32
        if remaining < 12:
            raise ValueError("Truncated checkpoint")
        digest = hashlib.sha256()
        while remaining:
            data = source.read(min(1024 * 1024, remaining))
            if not data:
                raise ValueError("Truncated checkpoint")
            digest.update(data)
            remaining -= len(data)
        if not hmac.compare_digest(digest.digest(), source.read(32)):
            raise ValueError("Checkpoint checksum mismatch")
        source.seek(0)
        if source.read(8) != b"NNQ35LR1":
            raise ValueError("Unexpected checkpoint format")
        header_size = struct.unpack("<i", source.read(4))[0]
        if not 0 < header_size <= 1024 * 1024:
            raise ValueError("Invalid checkpoint header size")
        header = json.loads(source.read(header_size))
        if expected_step is not None and header["Step"] != expected_step:
            raise ValueError(f"Expected step {expected_step}, found {header['Step']}")
        rank = header["Options"]["Rank"]
        expected_metadata = {
            "general.type": "adapter", "general.architecture": "qwen35", "adapter.type": "lora",
            "adapter.lora.alpha": struct.unpack("<f", struct.pack("<f", header["Options"]["Alpha"]))[0],
            "nntrain.training.step": header["Step"],
            "nntrain.base_model.sha256": header["ModelSha256"].upper(),
            "nntrain.checkpoint.payload_sha256": digest.hexdigest().upper(),
        }
        if header.get("TrainingIdentity") is not None:
            expected_metadata["nntrain.training.identity"] = header["TrainingIdentity"]
        for key, expected in expected_metadata.items():
            field = reader.get_field(key)
            if field is None or field.contents() != expected:
                raise ValueError(f"Adapter metadata mismatch: {key}")
        by_name = {tensor.name: tensor for tensor in reader.tensors}
        if len(by_name) != len(reader.tensors) or len(by_name) != 2 * len(header["Entries"]):
            raise ValueError("Unexpected adapter tensor count")
        for entry in header["Entries"]:
            for suffix, shape in (("a", (entry["Input"], rank)), ("b", (rank, entry["Output"]))):
                name = entry["Name"] + ".lora_" + suffix
                tensor = by_name.get(name)
                if tensor is None or tuple(tensor.shape) != shape or int(tensor.tensor_type) != 0:
                    raise ValueError(f"Tensor directory mismatch: {name}")
                count = 4 * shape[0] * shape[1]
                if tensor.n_bytes != count or tensor.data.tobytes() != source.read(count):
                    raise ValueError(f"Tensor payload mismatch: {name}")
            source.seek(8 * rank * (entry["Input"] + entry["Output"]), 1)
        if source.tell() != checkpoint.stat().st_size - 32:
            raise ValueError("Checkpoint tensor count/size mismatch")
    with adapter.open("rb") as stream:
        adapter_digest = hashlib.file_digest(stream, "sha256").hexdigest().upper()
    return {"reader": "llama.cpp gguf-py", "tensor_count": len(reader.tensors),
            "all_shapes_match": True, "all_tensor_bytes_identical_to_checkpoint": True,
            "bytes": adapter.stat().st_size, "step": header["Step"], "rank": rank,
            "alpha": header["Options"]["Alpha"], "adapter_sha256": adapter_digest,
            "base_sha256": header["ModelSha256"].upper()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--expected-step", type=int)
    parser.add_argument("--gguf-package", type=Path)
    args = parser.parse_args()
    if args.gguf_package is not None:
        sys.path.insert(0, str(args.gguf_package.resolve()))
    try:
        print(json.dumps(verify(args.checkpoint, args.adapter, args.expected_step), indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, ImportError, struct.error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
