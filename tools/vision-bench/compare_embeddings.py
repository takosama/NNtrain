"""Compare complete real-mmproj embeddings emitted by VisionBench."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("reference")
parser.add_argument("candidate")
parser.add_argument("--directory", type=Path,
                    default=Path("benchmark-results/vision-speed-20261005"))
parser.add_argument("--reference-directory", type=Path)
parser.add_argument("--max-relative-error", type=float, default=0.0)
arguments = parser.parse_args()
records = []
for size in (256, 768):
    reference = (arguments.reference_directory or arguments.directory) / f"{arguments.reference}-{size}.f32"
    candidate = arguments.directory / f"{arguments.candidate}-{size}.f32"
    a, b = (np.fromfile(path, dtype="<f4") for path in (reference, candidate))
    if a.size == 0 or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f"Invalid embedding pair: {reference}, {candidate}")
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    difference = b64 - a64
    denominator = np.linalg.norm(a64)
    relative = float(np.linalg.norm(difference) / denominator) if denominator else float(np.linalg.norm(difference))
    records.append({
        "size": size,
        "elements": int(a.size),
        "bitwise_equal": bool(np.array_equal(a.view(np.uint32), b.view(np.uint32))),
        "max_abs_error": float(np.abs(difference).max()),
        "relative_l2": relative,
        "cosine": float(np.dot(a64, b64) / (denominator * np.linalg.norm(b64))),
        "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "accepted": relative <= arguments.max_relative_error,
    })
text = json.dumps({"reference": arguments.reference, "candidate": arguments.candidate,
                   "max_relative_error": arguments.max_relative_error, "results": records}, indent=2)
(arguments.directory / f"{arguments.reference}-vs-{arguments.candidate}.json").write_text(text, encoding="utf-8")
print(text)
raise SystemExit(0 if all(record["accepted"] for record in records) else 1)
