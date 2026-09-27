#!/usr/bin/env python3
"""Export an NNQ35LR1 training checkpoint as a llama.cpp GGUF LoRA adapter.

Uses only Python's standard library; never loads the base model or its weights
into RAM/GPU. The training checkpoint remains the resumable source of truth.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
import tempfile
from typing import BinaryIO

CHUNK = 1024 * 1024
ALIGNMENT = 32
TARGETS = {"attn_q", "attn_k", "attn_v", "attn_output", "attn_qkv",
           "attn_gate", "ssm_alpha", "ssm_beta", "ssm_out", "ffn_gate",
           "ffn_up", "ffn_down"}


def read_exact(stream: BinaryIO, count: int) -> bytes:
    value = stream.read(count)
    if len(value) != count:
        raise ValueError("Truncated input file")
    return value


def unpack(stream: BinaryIO, fmt: str):
    return struct.unpack("<" + fmt, read_exact(stream, struct.calcsize("<" + fmt)))[0]


def gguf_string(stream: BinaryIO) -> str:
    count = unpack(stream, "Q")
    if count > 64 * 1024 * 1024:
        raise ValueError("GGUF string exceeds the supported limit")
    return read_exact(stream, count).decode("utf-8", errors="strict")


def skip_gguf_value(stream: BinaryIO, value_type: int, depth: int = 0) -> None:
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if value_type in sizes:
        read_exact(stream, sizes[value_type])
    elif value_type == 8:
        gguf_string(stream)
    elif value_type == 9 and depth == 0:
        element_type, count = unpack(stream, "I"), unpack(stream, "Q")
        if count > 10_000_000:
            raise ValueError("GGUF metadata array exceeds the supported limit")
        for _ in range(count):
            skip_gguf_value(stream, element_type, depth + 1)
    else:
        raise ValueError(f"Unsupported GGUF metadata type {value_type}")


def base_directory(stream: BinaryIO) -> dict[str, tuple[int, ...]]:
    stream.seek(0)
    if read_exact(stream, 4) != b"GGUF" or unpack(stream, "I") not in (2, 3):
        raise ValueError("Base is not a little-endian GGUF v2/v3 file")
    tensor_count, metadata_count = unpack(stream, "Q"), unpack(stream, "Q")
    if tensor_count > 100_000 or metadata_count > 100_000:
        raise ValueError("GGUF directory exceeds the supported limit")
    architecture = None
    keys = set()
    for _ in range(metadata_count):
        key, value_type = gguf_string(stream), unpack(stream, "I")
        if key in keys:
            raise ValueError(f"Duplicate GGUF metadata key: {key}")
        keys.add(key)
        if key == "general.architecture":
            if value_type != 8:
                raise ValueError("GGUF architecture must be a string")
            architecture = gguf_string(stream)
        else:
            skip_gguf_value(stream, value_type)
    if architecture != "qwen35":
        raise ValueError(f"Expected qwen35 base, got {architecture!r}")
    tensors = {}
    for _ in range(tensor_count):
        name, ndim = gguf_string(stream), unpack(stream, "I")
        if ndim < 1 or ndim > 4 or name in tensors:
            raise ValueError("Invalid or duplicate base tensor directory entry")
        shape = tuple(unpack(stream, "Q") for _ in range(ndim))
        if any(d == 0 for d in shape):
            raise ValueError("Base tensor dimensions must be positive")
        read_exact(stream, 12)  # storage type and offset; weights are not decoded
        tensors[name] = shape
    return tensors


def hash_prefix(stream: BinaryIO, count: int) -> bytes:
    stream.seek(0)
    digest = hashlib.sha256()
    while count:
        data = read_exact(stream, min(CHUNK, count))
        digest.update(data)
        count -= len(data)
    return digest.digest()


def positive_int(value, name: str, maximum: int = 2**31 - 1) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"Invalid {name}")
    return value


def load_checkpoint(stream: BinaryIO, base_sha256: str, base_tensors: dict,
                    expected_step: int | None) -> tuple[dict, list[dict], str]:
    size = os.fstat(stream.fileno()).st_size
    if size < 44:
        raise ValueError("Truncated LoRA checkpoint")
    payload_end = size - 32
    digest = hash_prefix(stream, payload_end)
    if not hmac.compare_digest(digest, read_exact(stream, 32)):
        raise ValueError("LoRA checkpoint checksum mismatch")
    stream.seek(0)
    if read_exact(stream, 8) != b"NNQ35LR1":
        raise ValueError("Not an NNtrain Qwen3.5 LoRA checkpoint")
    length = unpack(stream, "i")
    if not 1 <= length <= min(CHUNK, payload_end - stream.tell()):
        raise ValueError("Invalid checkpoint header length")
    header = json.loads(read_exact(stream, length))
    if not isinstance(header, dict) or header.get("Version") != 1:
        raise ValueError("Unsupported checkpoint version")
    if str(header.get("ModelSha256", "")).upper() != base_sha256:
        raise ValueError("LoRA checkpoint base SHA-256 mismatch")
    step = header.get("Step")
    if type(step) is not int or not 0 <= step <= 2**31 - 1:
        raise ValueError("Invalid checkpoint step")
    if expected_step is not None and step != expected_step:
        raise ValueError(f"Expected completed step {expected_step}, found {step}")
    options = header.get("Options")
    if not isinstance(options, dict):
        raise ValueError("Missing LoRA options")
    rank = positive_int(options.get("Rank"), "rank", 256)
    alpha = options.get("Alpha")
    if type(alpha) not in (int, float) or not math.isfinite(alpha) or not 0 < alpha <= 1024:
        raise ValueError("Invalid LoRA alpha")
    targets = options.get("Targets")
    if (not isinstance(targets, list) or any(type(x) is not str for x in targets)
            or len(set(targets)) != len(targets) or not set(targets) <= TARGETS):
        raise ValueError("Invalid LoRA target options")
    layers = options.get("Layers")
    if layers is not None and (not isinstance(layers, list) or not layers
            or any(type(x) is not int or x < 0 for x in layers)
            or len(set(layers)) != len(layers)):
        raise ValueError("Invalid LoRA layer options")
    if type(options.get("IncludeOutput")) is not bool:
        raise ValueError("Invalid IncludeOutput option")
    expected = []
    for name in base_tensors:
        match = re.fullmatch(r"blk\.(\d+)\.([a-z_]+)\.weight", name)
        if match and match[2] in targets and (layers is None or int(match[1]) in layers):
            expected.append(name)
    if options["IncludeOutput"]:
        expected.append("output.weight")
    expected.sort()
    entries = header.get("Entries")
    if (not isinstance(entries, list) or not entries or len(entries) > 100_000
            or any(not isinstance(entry, dict) for entry in entries)
            or [entry.get("Name") for entry in entries] != expected):
        raise ValueError("LoRA target directory does not match the selected base targets")
    records = []
    position = stream.tell()
    for entry in entries:
        name = entry["Name"]
        input_width = positive_int(entry.get("Input"), "input dimension")
        output_width = positive_int(entry.get("Output"), "output dimension")
        # NNtrain can adapt the output head even when it is tied to embeddings.
        shape = base_tensors.get(name)
        if name == "output.weight" and shape is None:
            shape = base_tensors.get("token_embd.weight")
            if shape is not None:
                raise ValueError("GGUF export of a tied output adapter is unsupported")
        if shape != (input_width, output_width):
            raise ValueError(f"LoRA matrix shape mismatch: {name}")
        a_bytes, b_bytes = 4 * input_width * rank, 4 * output_width * rank
        if position + 3 * (a_bytes + b_bytes) > payload_end:
            raise ValueError(f"Truncated LoRA tensor: {name}")
        records.extend([
            {"name": name + ".lora_a", "shape": (input_width, rank), "source": position, "size": a_bytes},
            {"name": name + ".lora_b", "shape": (rank, output_width), "source": position + a_bytes, "size": b_bytes},
        ])
        position += 3 * (a_bytes + b_bytes)  # skip Adam mA, mB, vA, vB
    if position != payload_end:
        raise ValueError("Unexpected trailing checkpoint tensor data")
    return header, records, digest.hex().upper()


def write_string(stream: BinaryIO, value: str) -> None:
    data = value.encode("utf-8")
    stream.write(struct.pack("<Q", len(data)))
    stream.write(data)


def pad(stream: BinaryIO) -> None:
    stream.write(b"\0" * (-stream.tell() % ALIGNMENT))


def write_gguf(source: BinaryIO, dest: BinaryIO, header: dict, records: list[dict],
               base_name: str, output_name: str, checkpoint_digest: str) -> None:
    metadata = {
        "general.type": (8, "adapter"),
        "general.architecture": (8, "qwen35"),
        "general.name": (8, output_name),
        "general.file_type": (4, 0),  # ALL_F32
        "general.alignment": (4, ALIGNMENT),
        "adapter.type": (8, "lora"),
        "adapter.lora.alpha": (6, header["Options"]["Alpha"]),
        "nntrain.base_model.name": (8, base_name),
        "nntrain.base_model.sha256": (8, header["ModelSha256"].upper()),
        "nntrain.training.step": (4, header["Step"]),
        "nntrain.checkpoint.payload_sha256": (8, checkpoint_digest),
    }
    if header.get("TrainingIdentity") is not None:
        if not isinstance(header["TrainingIdentity"], str):
            raise ValueError("Invalid training identity")
        metadata["nntrain.training.identity"] = (8, header["TrainingIdentity"])
    dest.write(struct.pack("<4sIQQ", b"GGUF", 3, len(records), len(metadata)))
    for key, (value_type, value) in metadata.items():
        write_string(dest, key)
        dest.write(struct.pack("<I", value_type))
        if value_type == 8:
            write_string(dest, value)
        else:
            dest.write(struct.pack("<f" if value_type == 6 else "<I", value))
    offset = 0
    for record in records:
        write_string(dest, record["name"])
        dest.write(struct.pack("<IQQIQ", 2, *record["shape"], 0, offset))
        offset += record["size"]
        offset += -offset % ALIGNMENT
    pad(dest)
    for record in records:
        source.seek(record["source"])
        remaining = record["size"]
        while remaining:
            data = read_exact(source, min(CHUNK, remaining))
            values = array.array("f")
            values.frombytes(data)
            if sys.byteorder != "little":
                values.byteswap()
            if not all(map(math.isfinite, values)):
                raise ValueError(f"Non-finite LoRA parameters: {record['name']}")
            dest.write(data)
            remaining -= len(data)
        pad(dest)


def export(checkpoint: Path, model: Path, output: Path, *, expected_step: int | None = None,
           overwrite: bool = False) -> dict:
    checkpoint, model, output = checkpoint.resolve(), model.resolve(), output.resolve()
    if checkpoint == model or output in (checkpoint, model):
        raise ValueError("Checkpoint, base model and output must be distinct files")
    if output.exists():
        if output.samefile(checkpoint) or output.samefile(model):
            raise ValueError("Output must not overwrite an input file or its hard link")
        if not overwrite:
            raise ValueError("Output already exists; choose another path or pass --force")
    if output.suffix.lower() != ".gguf":
        raise ValueError("Output filename must end in .gguf")
    if expected_step is not None and expected_step < 0:
        raise ValueError("Expected step must be nonnegative")
    with model.open("rb") as base, checkpoint.open("rb") as source:
        base_sha256 = hash_prefix(base, os.fstat(base.fileno()).st_size).hex().upper()
        tensors = base_directory(base)
        header, records, digest = load_checkpoint(source, base_sha256, tensors, expected_step)
        output.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=output.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as dest:
                write_gguf(source, dest, header, records, model.name, output.stem, digest)
                dest.flush()
                os.fsync(dest.fileno())
            if overwrite:
                os.replace(temporary, output)
            else:
                # Atomic publication without a race that could replace a new file.
                os.link(temporary, output)
                temporary.unlink()
        finally:
            temporary.unlink(missing_ok=True)
    return {"output": str(output), "bytes": output.stat().st_size, "step": header["Step"],
            "tensors": len(records), "rank": header["Options"]["Rank"],
            "alpha": header["Options"]["Alpha"], "base_sha256": base_sha256,
            "format": "GGUF v3 F32 LoRA adapter"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, help="Require this completed update count before exporting")
    parser.add_argument("--force", action="store_true", help="Atomically replace an existing output adapter")
    args = parser.parse_args()
    try:
        result = export(args.checkpoint, args.model, args.output,
                        expected_step=args.expected_step, overwrite=args.force)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
