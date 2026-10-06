"""Compare completed, paired Qwen LoRA SpeedProbe reports.

This script performs no training. It rejects pairs with different model, data,
token IDs, optimizer setup, device layout, or unapproved execution options.
It does not infer quality from a synthetic performance fixture.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path


def require_same(label: str, left: object, right: object) -> None:
    if left != right:
        raise ValueError(f"{label} differs between baseline and candidate")


def require_finite_positive(label: str, value: object) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return number


def compare_pair(
    length: int, baseline_path: Path, candidate_path: Path, allowed_options: set[str]
) -> dict[str, object]:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8-sig"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8-sig"))
    for name, report in (("baseline", baseline), ("candidate", candidate)):
        if report.get("Status") != "completed":
            raise ValueError(f"{name} report is not completed")
        if report.get("RequestedTokens") != length:
            raise ValueError(f"{name} is not a {length}-token fixture")
        if not report.get("Samples"):
            raise ValueError(f"{name} has no completed updates")

    for key in (
        "PromptMode", "RequestedTokens", "Model", "Data", "Resume",
        "SelectedExamples", "LoraOptions", "InitialStep", "LoraTargets",
    ):
        require_same(key, baseline.get(key), candidate.get(key))
    for key in ("Framework", "OperatingSystem", "Architecture"):
        require_same(f"Runtime.{key}", baseline["Runtime"].get(key), candidate["Runtime"].get(key))
    left_lanes = baseline.get("Lanes", [])
    right_lanes = candidate.get("Lanes", [])
    require_same("device layout", [lane["Device"] for lane in left_lanes],
                 [lane["Device"] for lane in right_lanes])

    # Older frozen probes predate these options. Their absence is equivalent
    # to the declared disabled default, not an unexplained setting change.
    legacy_defaults = {
        "TrainingStreamedAttentionTileRows": 0,
        "TrainingHostCheckpointBufferHandoff": False,
        "TrainingHostCheckpointForwardBufferHandoff": False,
        "TrainingHostCheckpointBackwardBufferHandoff": False,
        "TrainingHostCheckpointForwardCopyHandoff": False,
        "TrainingHostCheckpointAsyncRead": False,
        "TrainingHybridCheckpointMiBPerDevice": 0,
        "TrainingIQ2RollingTranspose": False,
        "TrainingFusedAttentionRows": False,
        "TrainingFusedAttentionOutput": False,
        "TrainingOnlineAttention": False,
    }
    left_options = {**legacy_defaults, **(baseline.get("Options") or {})}
    right_options = {**legacy_defaults, **(candidate.get("Options") or {})}
    actual_diffs = {
        key: [left_options.get(key), right_options.get(key)]
        for key in left_options.keys() | right_options.keys()
        if left_options.get(key) != right_options.get(key)
    }
    unapproved = sorted(actual_diffs.keys() - allowed_options)
    if unapproved:
        raise ValueError("unapproved execution option differences: " + ", ".join(unapproved))
    if bool(baseline.get("KernelTimingsEnabled")) or bool(candidate.get("KernelTimingsEnabled")):
        raise ValueError("disable kernel timing collection for speed comparisons")

    base_samples = baseline["Samples"]
    cand_samples = candidate["Samples"]
    if len(base_samples) != len(cand_samples):
        raise ValueError("update count differs")
    rows: list[dict[str, object]] = []
    for index, (base, cand) in enumerate(zip(base_samples, cand_samples, strict=True), start=1):
        for key in (
            "Example", "PhysicalLine", "ExampleSourceSha256", "TokenIds",
            "Tokens", "ResponseStartIndex", "SupervisedTokens", "Step",
        ):
            require_same(f"sample {index}.{key}", base.get(key), cand.get(key))
        if base["Tokens"] != length or cand["Tokens"] != length:
            raise ValueError(f"sample {index} has the wrong token count")
        base_seconds = require_finite_positive(f"baseline sample {index} seconds", base["Seconds"])
        cand_seconds = require_finite_positive(f"candidate sample {index} seconds", cand["Seconds"])
        base_loss = float(base["Loss"])
        cand_loss = float(cand["Loss"])
        base_norm = float(base["GradientNorm"])
        cand_norm = float(cand["GradientNorm"])
        if not all(map(math.isfinite, (base_loss, cand_loss, base_norm, cand_norm))):
            raise ValueError(f"sample {index} has nonfinite loss or gradient norm")
        rows.append({
            "step": base["Step"],
            "baselineSeconds": base_seconds,
            "candidateSeconds": cand_seconds,
            "inputTokens": length - 1,
            "baselineInputTokensPerSecond": (length - 1) / base_seconds,
            "candidateInputTokensPerSecond": (length - 1) / cand_seconds,
            "lossAbsDelta": abs(base_loss - cand_loss),
            "gradientNormAbsDelta": abs(base_norm - cand_norm),
        })

    base_times = [float(row["baselineSeconds"]) for row in rows]
    cand_times = [float(row["candidateSeconds"]) for row in rows]
    base_mean = statistics.mean(base_times)
    cand_mean = statistics.mean(cand_times)
    def backend_peaks(report: dict[str, object]) -> list[int]:
        samples = report["Samples"]
        return [int(device["PeakAllocatedBytes"])
                for device in samples[-1].get("GpuAfter", [])]

    return {
        "tokens": length,
        "baselineReport": str(baseline_path.resolve()),
        "candidateReport": str(candidate_path.resolve()),
        "baselineBinariesSha256": baseline.get("BinarySha256"),
        "candidateBinariesSha256": candidate.get("BinarySha256"),
        "approvedOptionDifferences": actual_diffs,
        "updates": len(rows),
        "baselineMeanSeconds": base_mean,
        "candidateMeanSeconds": cand_mean,
        "durationReductionPercent": 100 * (base_mean - cand_mean) / base_mean,
        "throughputSpeedup": base_mean / cand_mean,
        "baselineInputTokensPerSecond": (length - 1) / base_mean,
        "candidateInputTokensPerSecond": (length - 1) / cand_mean,
        "maxLossAbsDelta": max(float(row["lossAbsDelta"]) for row in rows),
        "maxGradientNormAbsDelta": max(float(row["gradientNormAbsDelta"]) for row in rows),
        "baselineBackendPeakAllocatedBytesByDevice": backend_peaks(baseline),
        "candidateBackendPeakAllocatedBytesByDevice": backend_peaks(candidate),
        "samples": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", nargs=3, action="append", metavar=("TOKENS", "BASELINE", "CANDIDATE"),
                        required=True, help="repeat for 4096 and 8192")
    parser.add_argument("--allow-option-diff", action="append", default=[], metavar="NAME")
    parser.add_argument("--max-loss-abs-delta", type=float, default=None)
    parser.add_argument("--max-grad-abs-delta", type=float, default=None)
    parser.add_argument("--output", type=Path, help="new JSON file; existing files are preserved")
    args = parser.parse_args()
    try:
        pairs = [compare_pair(int(tokens), Path(base), Path(cand), set(args.allow_option_diff))
                 for tokens, base, cand in args.pair]
        if len({pair["tokens"] for pair in pairs}) != len(pairs):
            raise ValueError("duplicate token lengths")
        for pair in pairs:
            if args.max_loss_abs_delta is not None and pair["maxLossAbsDelta"] > args.max_loss_abs_delta:
                raise ValueError(f"{pair['tokens']}: loss delta exceeds tolerance")
            if args.max_grad_abs_delta is not None and pair["maxGradientNormAbsDelta"] > args.max_grad_abs_delta:
                raise ValueError(f"{pair['tokens']}: gradient delta exceeds tolerance")
        result = {"schemaVersion": 1, "pairs": pairs,
                  "note": "Synthetic single-update timing is exploratory; it does not establish epoch-level speed or heldout quality. Backend allocation peaks are not physical VRAM measurements."}
        output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.open("x", encoding="utf-8").write(output)
            print(args.output.resolve())
        else:
            sys.stdout.write(output)
        return 0
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"comparison failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
