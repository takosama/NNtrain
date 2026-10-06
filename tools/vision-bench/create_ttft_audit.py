"""Summarize serialized, real-model image benchmarks and numerical gates."""
import hashlib
import json
from pathlib import Path
from statistics import median
import sys
import xml.etree.ElementTree as ET

directory = Path("benchmark-results/vision-ttft-v2-20261005")
candidate_label = sys.argv[1] if len(sys.argv) > 1 else "gguf-bslm-final"
prepared_label = sys.argv[2] if len(sys.argv) > 2 else "gguf-bslm-prepared-final"
quality_suffix = sys.argv[3] if len(sys.argv) > 3 else "gguf-bslm"


def read(name):
    return json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))


def runs(record):
    return next(result["generation"] for result in record["results"] if "generation" in result)


def summarize(record):
    measurements = runs(record)
    return {
        "first_text_ms": [row["first_text_ms"] for row in measurements],
        "first_text_median_ms": median(row["first_text_ms"] for row in measurements),
        "post_first_token_median_ms": median(
            (row["elapsed_ms"] - row["first_text_ms"]) / (row["stats"]["CompletionTokens"] - 1)
            for row in measurements
        ),
        "model_load_ms": next(result["model_load_ms"] for result in record["results"] if "generation" in result),
        "image_prepare_ms": [row["image_prepare_ms"] for row in measurements],
        "attachment_prepare_ms": [row.get("attachment_prepare_ms") for row in measurements],
        "reused_prompt_tokens": [row["stats"]["ReusedPromptTokens"] for row in measurements],
    }


baseline, candidate, prepared = read("legacy-formal"), read(candidate_label), read(prepared_label)
baseline_runs, candidate_runs, prepared_runs = runs(baseline), runs(candidate), runs(prepared)
assert len(baseline_runs) == len(candidate_runs) == len(prepared_runs) == 3
for original, actual, primed in zip(baseline_runs, candidate_runs, prepared_runs):
    assert original["answer"] == actual["answer"] == primed["answer"]
    assert original["stats"]["PromptTokens"] == actual["stats"]["PromptTokens"] == primed["stats"]["PromptTokens"] == 599
    assert actual["stats"]["ReusedPromptTokens"] == 0
    assert primed["stats"]["ReusedPromptTokens"] == 580
    assert actual["stats"]["StopReason"] == primed["stats"]["StopReason"] == 0

gates = {}
for name in (f"real-model-parity-full-pipeline-{quality_suffix}", f"real-model-lora-parity-full-pipeline-{quality_suffix}"):
    gate = read(name)
    assert gate["accepted"] and all(row["tokens_equal"] for row in gate["comparisons"])
    gates[name] = gate

old, new, ready = summarize(baseline), summarize(candidate), summarize(prepared)
ratio = old["first_text_median_ms"] / new["first_text_median_ms"]
assert ratio >= 5, f"Required 5x TTFT improvement not reached: {ratio}"
paths = [Path("NNtrain.Gui/InferenceSession.cs"), Path("NNtrain.Core/Modules/Qwen35QuantizedModel.cs"),
         Path("NNtrain.Arc/Kernels/qwen35_prefill_xmx.cl"), Path("NNtrain.Arc/Kernels/qwen35_delta_fused.cl"),
         Path("NNtrain.Arc/Kernels/qwen35_vision_attention_xmx.cl"), Path("NNtrain.Core/Modules/Qwen35ExecutionOptions.cs"),
         Path("NNtrain.Core/Modules/Qwen35QuantizedModel.Multimodal.cs"), Path("NNtrain.Core/Modules/Qwen35QuantizedModel.Prefill.cs"),
         Path("NNtrain.Arc/ArcExecutionLane.ExternalMemory.cs"), Path("NNtrain.Gui/MultimodalPrompt.cs"),
         Path("NNtrain.Arc/ArcExecutionLane.cs"), Path("NNtrain.Arc/ArcExecutionOptions.cs"),
         Path("tools/vision-bench/Program.cs"), Path("tools/vision-bench/RealPrefillParity.cs")]
test_artifacts = {}
for name in ("prefill-xmx-gguf-bslm", "arc-external-reservation-20261005", "delta-rows-20261005",
             "delta-subgroup-rms-20261005", "vision-attention-xmx-20261005"):
    path = Path("benchmark-results/test-results") / f"{name}.trx"
    counters = ET.parse(path).find(".//{*}Counters").attrib
    assert int(counters["failed"]) == 0 and int(counters["passed"]) == int(counters["total"])
    test_artifacts[str(path)] = {"passed": int(counters["passed"]), "failed": int(counters["failed"])}
published = Path("NNtrain.Gui/bin/Release/net10.0-windows/win-x64/publish/NNtrain.Gui.exe")
assert published.is_file() and published.stat().st_mtime >= max(path.stat().st_mtime for path in paths)
audit = {
    "completed": True,
    "hardware": "Intel Arc B580 12GB x2",
    "model": "models/Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf",
    "mmproj": "models/huggingface/unsloth/Qwen3.5-27B-GGUF/mmproj-F16.gguf",
    "image_pixels": [768, 768], "prompt_tokens": 599,
    "temperature": 0, "top_k": 1, "thinking": False,
    "candidate_artifact": candidate_label, "prepared_artifact": prepared_label,
    "selected_quality_suffix": quality_suffix,
    "base_model_load_included_in_ttft": False,
    "new_image_preparation_included_in_raw_ttft": True,
    "attachment_preparation_included_in_prepared_send_ttft": False,
    "external_inference_server": False,
    "baseline": old, "candidate": new, "prepared_send": ready,
    "raw_ttft_speedup": ratio, "raw_ttft_fraction": 1 / ratio,
    "post_first_token_fraction": new["post_first_token_median_ms"] / old["post_first_token_median_ms"],
    "target_one_fifth_achieved": ratio >= 5,
    "quality": gates,
    "test_artifacts": test_artifacts,
    "additional_console_tests": {"mixed_prefix_lora_eos_passed": 7, "failed": 0, "trx_saved": False},
    "numerics": "Near-FP32 XMX prefill and vision; original GGUF GEMV and exact Delta state order. Not globally bitwise FP32.",
    "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
    "published_gui": {"path": str(published), "bytes": published.stat().st_size,
                      "sha256": hashlib.sha256(published.read_bytes()).hexdigest()},
}
(directory / "completion-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"speedup": ratio, "raw_median_ms": new["first_text_median_ms"],
                  "prepared_send_median_ms": ready["first_text_median_ms"]}, ensure_ascii=False))
