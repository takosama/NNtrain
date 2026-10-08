#!/usr/bin/env python3
"""Validate production IQ2_S benchmark evidence and emit a Japanese report.

Uses Python's standard library only; never loads a model or submits GPU work.
Example:
  python summarize.py final-forward/results.json final-transpose/results.json \
    --model-results final-model/model-results.json --output final-report \
    --test-trx tests/iq2-tiled-scaled-correctness.trx

The report uses host wall medians. When a case has several tiled variants it
shows the fastest measured candidate, explicitly without choosing a production
default. All variants, including unselected candidates, must pass validation.
"""

from __future__ import annotations

import argparse
import array
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import xml.etree.ElementTree as ET


class EvidenceError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def finite(value, label, *, positive=False):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), f"{label}: finite number required")
    require(value > 0 if positive else value >= 0, f"{label}: invalid sign")
    return float(value)


def integer(value, label, minimum=0):
    require(isinstance(value, int) and not isinstance(value, bool)
            and value >= minimum, f"{label}: integer >= {minimum} required")
    return value


def near(actual, expected, label, absolute=1e-9):
    finite(actual, label)
    require(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=absolute),
            f"{label}: stored {actual!r}, recomputed {expected!r}")


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def digest(value, label):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value),
            f"{label}: SHA-256 required")
    return value.lower()


def read_json(path):
    def reject_constant(value):
        raise EvidenceError(f"{path}: nonfinite JSON literal {value}")

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"{path}: duplicate JSON key {key}")
            result[key] = value
        return result

    document = json.loads(path.read_text(encoding="utf-8-sig"),
                          parse_constant=reject_constant, object_pairs_hook=unique_keys)
    require(document.get("schemaVersion") == 1, f"{path}: unsupported schema")
    require(document.get("complete") is True and document.get("error") is None,
            f"{path}: benchmark did not complete successfully")
    require(document.get("completedUtc"), f"{path}: completion timestamp missing")
    return document


def statistics_check(samples, metric, stored, count, label):
    require(len(samples) == count, f"{label}: wrong sample count")
    require([sample.get("round") for sample in samples] == list(range(count)),
            f"{label}: sample rounds must be consecutive from zero")
    values = [finite(sample.get(metric), f"{label}[{i}]", positive=True)
              for i, sample in enumerate(samples)]
    actual = {"count": count, "median": statistics.median(values),
              "mean": statistics.fmean(values),
              "standardDeviation": statistics.pstdev(values),
              "minimum": min(values), "maximum": max(values)}
    require(stored.get("count") == count, f"{label}: summary count mismatch")
    for key, value in actual.items():
        near(stored.get(key), value, f"{label}.{key}")
    return actual


def kernel_check(samples, metric, label):
    for index, sample in enumerate(samples):
        kernels = sample.get("kernelMilliseconds")
        require(isinstance(kernels, dict) and kernels, f"{label}[{index}]: no kernel timings")
        times = [finite(value, f"{label}[{index}].{key}") for key, value in kernels.items()]
        # Cumulative profile subtraction can introduce very small cancellation.
        near(sample[metric], math.fsum(times), f"{label}[{index}].kernel sum", 1e-6)


def validation_check(value, label, l2_limit, requested_points, *, full=True):
    require(isinstance(value, dict), f"{label}: missing validation")
    require(value.get("finite") is True, f"{label}: full output is not finite")
    require(value.get("guardsIntact") is True, f"{label}: output guards failed")
    require(integer(value.get("cpuPoints"), f"{label}.cpuPoints", 1) == requested_points,
            f"{label}: unexpected CPU reference point count")
    for key in ("maxAbsoluteErrorVsDouble", "relativeL2VsDouble", "worstToleranceRatio"):
        finite(value.get(key), f"{label}.{key}")
    require(value["worstToleranceRatio"] <= 1.0, f"{label}: CPU double tolerance failed")
    if full:
        require(value.get("fullOutputCompared") is True, f"{label}: full output comparison missing")
        finite(value.get("maxAbsoluteErrorVsOriginal"), f"{label}.maxAbsoluteErrorVsOriginal")
        finite(value.get("relativeL2VsOriginal"), f"{label}.relativeL2VsOriginal")
        require(value["relativeL2VsOriginal"] <= l2_limit, f"{label}: full-output L2 gate failed")
        require(isinstance(value.get("bitwiseEqualToOriginal"), bool), f"{label}: bitwise result missing")
    return value


def hash_manifest_check(values, label):
    require(isinstance(values, dict) and values, f"{label}: missing hash manifest")
    return {key: digest(value, f"{label}.{key}") for key, value in values.items()}


def source_evidence(path, document, *, model=False):
    measured_key = "measuredAssemblySha256" if model else "measuredFileSha256"
    measured = hash_manifest_check(document.get(measured_key), f"{path}.{measured_key}")
    for assembly in ("NNtrain.Arc.dll", "NNtrain.Core.dll"):
        require(assembly in measured, f"{path}: {assembly} hash missing")
    sources = {} if model else hash_manifest_check(document.get("sourceFileSha256"), f"{path}.sourceFileSha256")
    return {"path": str(path), "sha256": sha256(path), "startedUtc": document.get("startedUtc"),
            "completedUtc": document["completedUtc"], "measuredFileSha256": measured,
            "sourceFileSha256": sources, "model": document.get("model"),
            "device": document.get("deviceInformation" if model else "device"),
            "commandArguments": document.get("commandArguments"),
            "timing": document.get("timing"), "validation": document.get("validation"),
            "operation": document.get("operation"), "scope": document.get("scope")}


def validate_micro(path, args):
    document = read_json(path)
    direction = document.get("direction")
    require(direction in ("forward", "transpose"), f"{path}: production direction missing")
    require("FP32" in document.get("operation", "") and "IQ2S" in document["operation"],
            f"{path}: expected production FP32 activation / IQ2_S weight operation")
    count = integer(document["timing"].get("samples"), f"{path}.samples", args.minimum_samples)
    integer(document["timing"].get("warmup"), f"{path}.warmup", args.minimum_warmup)
    l2_limit = finite(document["validation"].get("allOutputL2Limit"), f"{path}.allOutputL2Limit", positive=True)
    require(l2_limit <= 5e-5, f"{path}: projection L2 gate was loosened")
    requested_points = integer(document["validation"].get("cpuPointsRequested"), f"{path}.cpuPointsRequested", 4)
    baseline_names = (["original-inference-gguf-bslm", "original-training-fp32-rows4"]
                      if direction == "forward" else ["original-transpose-vec8-rows8-reduce"])
    cases = document.get("cases")
    require(isinstance(cases, list) and cases, f"{path}: no cases")
    result_cases, comparisons = [], []
    for case_index, case in enumerate(cases):
        label = f"{path}.cases[{case_index}]"
        rows = integer(case.get("rows"), label + ".rows", 1)
        k = integer(case.get("inputWidth"), label + ".inputWidth", 1)
        n = integer(case.get("outputWidth"), label + ".outputWidth", 1)
        source = case.get("source", {})
        digest(source.get("payloadSha256"), label + ".source.payloadSha256")
        actual = source.get("kind") == "Complete unmodified GGUF IQ2_S tensor"
        require(actual or args.allow_synthetic, label + ": synthetic input requires --allow-synthetic")
        if actual:
            require(source.get("type") == 22 and source.get("shape") == [k, n],
                    label + ": expected IQ2_S GGUF shape [K,N]")
            require(source.get("tensor") == case.get("name"), label + ": tensor name mismatch")
            integer(source.get("bytes"), label + ".source.bytes", 1)
            require(document.get("model"), label + ": GGUF model provenance missing")
        activation = case.get("activation" if direction == "forward" else "upstreamGradient", {})
        require(activation.get("type") == "FP32", label + ": activation type must be FP32")
        require(activation.get("additionalQuantization") is not True, label + ": activation was quantized")
        integer(activation.get("seed"), label + ".activation.seed")
        digest(activation.get("sha256"), label + ".activation.sha256")
        points = min(requested_points, rows * (n if direction == "forward" else k))
        validation_check(case.get("originalValidation"), label + ".originalValidation", l2_limit, points, full=False)
        variants = case.get("variants")
        require(isinstance(variants, list) and variants, label + ": variants missing")
        names = [variant.get("name") for variant in variants]
        require(len(names) == len(set(names)), label + ": duplicate variant names")
        require(all(name in names for name in baseline_names), label + ": required original baselines missing")
        tiled_candidates = [variant for variant in variants if variant.get("name", "").startswith("tiled-")]
        direct_candidates = [variant for variant in variants if variant.get("name", "").startswith("inference-bslm-")]
        require(tiled_candidates, label + ": no tiled training/backward candidate")
        for variant in variants:
            variant_label = label + "." + variant["name"]
            for phase in ("validation", "postTimingValidation"):
                validation_check(variant.get(phase), variant_label + "." + phase, l2_limit, points)
                if variant["name"].startswith("inference-bslm-"):
                    require(variant[phase].get("bitwiseEqualToOriginal") is True,
                            variant_label + ": inference BSLM must be bitwise equal before and after timing")
            if variant.get("scalarFallbackComparison") is not None:
                validation_check(variant["scalarFallbackComparison"], variant_label + ".scalarFallbackComparison", l2_limit, points)
            samples = variant.get("samples", [])
            for metric in ("gpuMilliseconds", "wallMilliseconds"):
                statistics_check(samples, metric, variant.get(metric, {}), count, variant_label + "." + metric)
            kernel_check(samples, "gpuMilliseconds", variant_label)
        selected_candidates = {}
        stored_case = {"sourceResult": str(path), "direction": direction, "name": case["name"],
                       "rows": rows, "inputWidth": k, "outputWidth": n, "source": source,
                       "activation": activation, "bias": case.get("bias"), "original": case.get("original"),
                       "originalValidation": case["originalValidation"], "variants": variants,
                       "selectedCandidates": selected_candidates,
                       "cpuPoints": points, "projectionL2Limit": l2_limit,
                       "fallback": document["validation"].get("fallback", False)}
        result_cases.append(stored_case)
        for baseline_name in baseline_names:
            candidates = (direct_candidates or tiled_candidates) if baseline_name == "original-inference-gguf-bslm" else tiled_candidates
            selected = min(candidates, key=lambda variant: variant["wallMilliseconds"]["median"])
            selection = "fastest-measured-candidate" if len(candidates) > 1 else "only-measured-candidate"
            selected_candidates[baseline_name] = selected["name"]
            for candidate in candidates:
                baseline = next(item for item in variants if item["name"] == baseline_name)
                comparisons.append({"sourceResult": str(path), "direction": direction, "name": case["name"],
                                    "rows": rows, "inputWidth": k, "outputWidth": n,
                                    "baseline": baseline_name, "candidate": candidate["name"],
                                    "selectedForReport": candidate is selected, "candidateSelection": selection,
                                    "hostWallMedianSpeedup": baseline["wallMilliseconds"]["median"] / candidate["wallMilliseconds"]["median"],
                                    "gpuKernelMedianSpeedup": baseline["gpuMilliseconds"]["median"] / candidate["gpuMilliseconds"]["median"],
                                    "baselineWallMilliseconds": baseline["wallMilliseconds"],
                                    "candidateWallMilliseconds": candidate["wallMilliseconds"],
                                    "baselineGpuMilliseconds": baseline["gpuMilliseconds"],
                                    "candidateGpuMilliseconds": candidate["gpuMilliseconds"]})
    return source_evidence(path, document), result_cases, comparisons


def logits_load(result_path, metadata):
    original_path = Path(metadata["path"])
    path = original_path if original_path.is_file() else result_path.parent / original_path.name
    require(path.is_file(), f"{result_path}: logits file missing: {original_path}")
    count = integer(metadata.get("count"), str(path) + ".count", 1)
    require(path.stat().st_size == count * 4, f"{path}: unexpected logit byte count")
    require(sha256(path) == digest(metadata.get("sha256"), str(path) + ".sha256"), f"{path}: logits SHA mismatch")
    values = array.array("f")
    values.frombytes(path.read_bytes())
    if sys.byteorder != "little":
        values.byteswap()
    require(all(math.isfinite(value) for value in values), f"{path}: nonfinite logits")
    return values


def logits_compare(original, candidate):
    require(len(original) == len(candidate) and original, "Logit dimensions differ")
    difference = math.fsum((a - b) ** 2 for a, b in zip(original, candidate))
    norm = math.fsum(value * value for value in original)
    old_top = max(range(len(original)), key=original.__getitem__)
    new_top = max(range(len(candidate)), key=candidate.__getitem__)
    return {"count": len(original), "maxAbsoluteError": max(abs(a - b) for a, b in zip(original, candidate)),
            "relativeL2": math.sqrt(difference / max(norm, 1e-100)),
            "originalTop1": old_top, "candidateTop1": new_top, "top1Equal": old_top == new_top}


def validate_model(path, *, profile_evidence=False):
    document = read_json(path)
    count = integer(document["timing"].get("sampleCount"), str(path) + ".sampleCount", 1 if profile_evidence else 3)
    integer(document["timing"].get("warmup"), str(path) + ".warmup", 0 if profile_evidence else 1)
    profiled = document["timing"].get("profileKernels", True)
    require(isinstance(profiled, bool), f"{path}: profileKernels must be boolean")
    require(not profile_evidence or profiled, f"{path}: separate profile evidence must enable profiling")
    if not profiled:
        require(document["timing"].get("requiresSeparateProfileEvidence") is True, f"{path}: no-profile evidence requirement missing")
    digest(document["model"].get("sha256"), str(path) + ".model.sha256")
    gate = finite(document["numericGate"].get("l2Limit"), str(path) + ".l2Limit", positive=True)
    require(gate <= 0.001 and document["numericGate"].get("top1MustMatch") is True,
            f"{path}: model numeric gate was loosened")
    options = document.get("options", {})
    original_options, candidate_options = options.get("original", {}), options.get("candidate", {})
    require(original_options and candidate_options, f"{path}: model options missing")
    switches = [key for key in set(original_options) | set(candidate_options)
                if original_options.get(key) != candidate_options.get(key)]
    candidate_kernels = document.get("candidateKernels", {})
    attention_enabled = bool(candidate_kernels.get("attention"))
    expected_switches = {"iq2tiledforward", "inferencebatchtextattention"} if attention_enabled else {"iq2tiledforward"}
    require({key.lower() for key in switches} == expected_switches
            and all(original_options[key] is False and candidate_options[key] is True for key in switches),
            f"{path}: model sessions differ beyond declared IQ2/attention switches")
    for values in (original_options, candidate_options):
        collect = next((value for key, value in values.items() if key.lower() == "collectkerneltimings"), None)
        require(collect is profiled, f"{path}: options and timing disagree about profiling")
    threshold = integer(candidate_kernels.get("projectionMinimumRows", 1024), str(path) + ".projectionMinimumRows", 128)
    projection = candidate_kernels.get("projection")
    if projection is not None:
        require(projection == "q35l_prefill_xmx_iq2_s_gguf_bslm_k32r" and threshold == 512,
                f"{path}: unknown declared production IQ2 kernel/policy")
    def projection_exercised(kernels):
        return kernels.get(projection, 0) > 0 if projection else any(
            name.startswith("q35s_forward_") and value > 0 for name, value in kernels.items())
    def attention_exercised(kernels):
        return kernels.get("q35a_scores_rows", 0) > 0 and kernels.get("q35a_attend_rows", 0) > 0
    passes = document.get("passes", [])
    require(passes, f"{path}: no model samples")
    declared_summaries = {}
    for item in document.get("summary", []):
        length = integer(item.get("prefixTokens"), str(path) + ".summary.prefixTokens", 128)
        require(length not in declared_summaries, f"{path}: duplicate model summary prefix")
        expected = item.get("candidateKernelExpected", True)
        require(isinstance(expected, bool), f"{path}: candidateKernelExpected must be boolean")
        if projection is not None or not expected:
            chunk = integer(document["timing"].get("configuredPrefixChunkTokens"), str(path) + ".configuredPrefixChunkTokens", 128)
            require(expected == (min(length, chunk) >= threshold), f"{path}: IQ2 coverage expectation disagrees with chunk size")
        if attention_enabled:
            require(item.get("candidateAttentionExpected") is True, f"{path}: batched attention coverage expectation missing")
        declared_summaries[length] = item
    indexed = {}
    for record in passes:
        key = (integer(record.get("prefixTokens"), str(path) + ".prefixTokens", 128), record.get("tiled"))
        require(isinstance(key[1], bool) and key not in indexed, f"{path}: duplicate/invalid model pass")
        indexed[key] = record
        statistics_check(record["samples"], "wallMilliseconds", record.get("wallMilliseconds", {}), count, f"{path}.{key}.wallMilliseconds")
        if profiled:
            statistics_check(record["samples"], "gpuKernelMilliseconds", record.get("gpuKernelMilliseconds", {}), count, f"{path}.{key}.gpuKernelMilliseconds")
            kernel_check(record["samples"], "gpuKernelMilliseconds", f"{path}.{key}")
        else:
            require(record.get("gpuKernelMilliseconds") is None, f"{path}: unprofiled GPU statistics must be null")
        for sample in record["samples"]:
            require(key[0] in declared_summaries, f"{path}.{key}: model summary missing")
            expected = declared_summaries[key[0]].get("candidateKernelExpected", True)
            if profiled:
                require(projection_exercised(sample["kernelMilliseconds"]) == (key[1] and expected),
                        f"{path}.{key}: measured IQ2 route disagrees with policy")
                if attention_enabled:
                    require(attention_exercised(sample["kernelMilliseconds"]) == key[1], f"{path}.{key}: incorrect attention route")
            else:
                require(sample.get("gpuKernelMilliseconds") is None and sample.get("kernelMilliseconds") == {},
                        f"{path}: unprofiled kernel measurements must be absent")
            phase_keys = ("resetMilliseconds", "primePrefixMilliseconds", "lastTokenMilliseconds")
            if any(phase in sample for phase in phase_keys):
                durations = [finite(sample.get(phase), f"{path}.{key}.{phase}") for phase in phase_keys]
                near(sample["wallMilliseconds"], math.fsum(durations), f"{path}.{key}.phase sum", 1e-6)
    require(set(declared_summaries) == {key[0] for key in indexed}, f"{path}: summary and model pass coverage differ")
    rows = []
    for length in sorted({key[0] for key in indexed}):
        require((length, False) in indexed and (length, True) in indexed, f"{path}: unpaired model prefix {length}")
        old, new = indexed[length, False], indexed[length, True]
        files = document.get("logitFiles", {})
        old_values = logits_load(path, files[f"original-prefix{length}.logits.f32"])
        new_values = logits_load(path, files[f"tiled-prefix{length}.logits.f32"])
        comparison = logits_compare(old_values, new_values)
        require(comparison["relativeL2"] <= gate and comparison["top1Equal"], f"{path}: full model logit gate failed")
        stored = new.get("logitComparison", {})
        for metric in ("relativeL2", "maxAbsoluteError"):
            near(stored.get(metric), comparison[metric], f"{path}.{length}.{metric}")
        for metric in ("count", "originalTop1", "candidateTop1", "top1Equal"):
            require(stored.get(metric) == comparison[metric], f"{path}.{length}.{metric}: logit comparison mismatch")
        row = {"prefixTokens": length, "totalPromptTokens": length + 1,
               "hostWallMedianSpeedup": old["wallMilliseconds"]["median"] / new["wallMilliseconds"]["median"],
               "gpuKernelMedianSpeedup": old["gpuKernelMilliseconds"]["median"] / new["gpuKernelMilliseconds"]["median"] if profiled else None,
               "baselineWallMilliseconds": old["wallMilliseconds"], "candidateWallMilliseconds": new["wallMilliseconds"],
               "baselineGpuMilliseconds": old["gpuKernelMilliseconds"], "candidateGpuMilliseconds": new["gpuKernelMilliseconds"],
               "logitComparison": comparison}
        stored_summary = declared_summaries[length]
        expected = stored_summary.get("candidateKernelExpected", True)
        require(stored_summary.get("candidateKernelExercised") is (expected if profiled else None), f"{path}: model summary IQ2 route mismatch")
        if attention_enabled:
            require(stored_summary.get("candidateAttentionExercised") is (True if profiled else None), f"{path}: model summary attention route mismatch")
        row["candidateKernelExpected"] = expected
        row["candidateKernelExercised"] = expected if profiled else None
        row["candidateAttentionExpected"] = attention_enabled
        row["candidateAttentionExercised"] = attention_enabled if profiled else None
        near(stored_summary.get("wallMedianSpeedup"), row["hostWallMedianSpeedup"], f"{path}.summary.wallSpeedup")
        if profiled:
            near(stored_summary.get("gpuKernelMedianSpeedup"), row["gpuKernelMedianSpeedup"], f"{path}.summary.gpuSpeedup")
        else:
            require(stored_summary.get("gpuKernelMedianSpeedup") is None, f"{path}: unprofiled GPU speedup must be null")
        row["phaseMedians"] = {"original": {}, "candidate": {}}
        for label, record in (("original", old), ("candidate", new)):
            for phase in ("resetMilliseconds", "primePrefixMilliseconds", "lastTokenMilliseconds"):
                if all(phase in sample for sample in record["samples"]):
                    row["phaseMedians"][label][phase] = statistics.median(sample[phase] for sample in record["samples"])
        rows.append(row)
    return {"evidence": source_evidence(path, document, model=True), "comparisons": rows,
            "options": options, "prompt": document.get("prompt"), "numericGate": document["numericGate"],
            "candidateKernels": candidate_kernels, "profileKernels": profiled,
            "logitFiles": document["logitFiles"], "passes": passes}


def validate_profile_pair(model, profile):
    require(profile is not None, "No-profile model results require --model-profile with separately measured route evidence")
    require(profile["profileKernels"], "Separate route evidence must enable kernel profiling")
    require(model["evidence"]["model"]["sha256"] == profile["evidence"]["model"]["sha256"], "Model SHA differs between latency/profile runs")
    require(model["evidence"]["device"] == profile["evidence"]["device"], "Model devices differ between latency/profile runs")
    require(model["candidateKernels"] == profile["candidateKernels"], "Candidate policy differs between latency/profile runs")
    for session in ("original", "candidate"):
        def without_profiling(values):
            return {key: value for key, value in values.items() if key.lower() != "collectkerneltimings"}
        require(without_profiling(model["options"][session]) == without_profiling(profile["options"][session]),
                f"Model {session} configuration differs beyond profiling")
    require(model["prompt"].get("seedText") == profile["prompt"].get("seedText"), "Model prompt seeds differ")
    for length, ids in profile["prompt"]["tokenIds"].items():
        if length in model["prompt"]["tokenIds"]:
            require(model["prompt"]["tokenIds"][length] == ids, "Profile prompt tokens differ from latency run")
    if any(row["candidateKernelExpected"] for row in model["comparisons"]):
        require(any(row["candidateKernelExercised"] is True for row in profile["comparisons"]), "Profile does not demonstrate selected IQ2 kernel")
    if any(row["candidateAttentionExpected"] for row in model["comparisons"]):
        require(any(row["candidateAttentionExercised"] is True for row in profile["comparisons"]), "Profile does not demonstrate selected attention kernels")


def validate_trx(path):
    root = ET.parse(path).getroot()
    results = list(root.iterfind(".//{*}UnitTestResult"))
    counters = root.find(".//{*}Counters")
    require(results and counters is not None, f"{path}: no TRX test evidence")
    require(all(result.get("outcome") == "Passed" for result in results), f"{path}: nonpassing test result")
    count = len(results)
    require(int(counters.get("total", -1)) == count and int(counters.get("passed", -1)) == count,
            f"{path}: test counter mismatch")
    names = [item.get("testName", "") for item in results]
    amplitudes = []
    for name in names:
        match = re.search(r"TinyNormalRows.*amplitude:\s*([\d.eE+-]+)", name)
        if match:
            amplitudes.append(float(match.group(1)))
    return {"path": str(path), "sha256": sha256(path), "passed": count, "testNames": names,
            "tinyNormalAmplitudes": sorted(amplitudes),
            "scope": "Separate unit-test evidence; these extreme amplitudes are not microbenchmark timing inputs."}


def esc(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def link(path, label=None):
    return f"[{esc(label or Path(path).name)}](<{Path(path).as_posix()}>)"


def table(headers, rows):
    return ["| " + " | ".join(map(esc, headers)) + " |",
            "| " + " | ".join("---" for _ in headers) + " |"] + [
                "| " + " | ".join(map(esc, row)) + " |" for row in rows]


def format_stats(stats):
    return f"{stats['mean']:.3f} ± {stats['standardDeviation']:.3f} ({stats['minimum']:.3f}–{stats['maximum']:.3f})"


def production_route(row, policy):
    if row["baseline"] != "original-inference-gguf-bslm":
        return "16×32新経路の対象" if row["rows"] >= 128 else "旧経路を維持"
    if policy == "rowmajor-bslm":
        return "row-major BSLMの対象" if row["rows"] >= 512 else "旧IQ2経路を維持"
    eligible = row["rows"] >= 128 and ((row["inputWidth"] >= 8192 and row["rows"] >= 1024)
                                      or (row["outputWidth"] >= 16384 and row["rows"] >= 4096))
    return "16×32新経路の対象" if eligible else "旧経路を維持"


def median_text(stats):
    return "—" if stats is None else f"{stats['median']:.3f}"


def model_route(row):
    projection = "IQ2 row-major" if row["candidateKernelExpected"] else "旧IQ2"
    attention = " + batched text attention" if row["candidateAttentionExpected"] else ""
    evidence = "（別profileで確認）" if row["candidateKernelExercised"] is None else "（profile確認済み）"
    return projection + attention + evidence


def devices_text(value):
    devices = value if isinstance(value, list) else [value]
    return "; ".join(f"GPU {item.get('index', '?')}: {item.get('name', '?')}"
                     for item in devices if isinstance(item, dict))


def render(summary, output):
    selected = [row for row in summary["comparisons"] if row["selectedForReport"]]
    lines = ["# IQ2_S 本番演算経路の比較", "", "## 測定の範囲", "",
             "FP32の入力とGGUFのIQ2_S重みを使う投影演算を比較した。入力をIQ2_Sへ量子化する測定ではない。",
             "主指標はhost wall時間の中央値。各経路に必要なpack、作業バッファの確保・再利用、GPU同期を含む。",
             "初期入力転送、モデル読込、コンパイル、検証用の出力読戻しは投影演算の時間から除外した。逆伝播のGPU内コピーはhost wallに含み、GPU列のカーネル時間合計には含まれない。",
             "倍率は旧時間÷新時間で、1倍を超えれば高速化。GPU列はイベント時間の合計であり、host wallとは別の指標。", "",
             "候補が複数あるケースは、host wall中央値が最小だった候補を各行に明記した。これは測定内の候補選択であり、製品の既定値を決定したという意味ではない。", "",
             "以下の投影演算の倍率を、LoRA学習全体やモデル全体の倍率として扱わない。", ""]
    if summary["productionPolicy"] == "rowmajor-bslm":
        lines += ["## 組込み時の経路選択", "",
                  "推論は既存BSLMの演算順を維持し、入力high/residualの配置をrow-majorに変更する。学習は別の16×32経路を使用する。以下は各候補を直接呼んだマイクロベンチ。", "",
                  "- 推論：M≥512で `q35l_prefill_xmx_iq2_s_gguf_bslm_k32r` とrow-major packを使用。M<512では旧IQ2経路を維持。比較した候補は計測前後の全出力で旧BSLMとbit一致を必須とした。",
                  "- 学習forward・逆伝播dX：M≥128で16×32新経路の対象。",
                  "- 全モデルの候補はbatched text attentionも有効化。256 tokenでもattentionは変更されるため、そのモデル速度差をIQ2単独の効果とは解釈しない。",
                  "- メモリ・デバイス対応・resident panel等による既存経路への分岐は残る。表は形状条件を示す。",
                  "- Mは実際に投影する行数。GUIのprefix chunkが1024なら、8192 tokenのpromptでも各chunkのMは最大1024となる。", ""]
    elif summary["productionPolicy"] == "arc-b580-hybrid":
        lines += ["## 組込み時の経路選択", "",
                  "選択された本番方針は16×32の新経路と既存経路の併用。以下の表は新経路を直接呼んだマイクロベンチで、全形状を新経路に置き換えた製品の速度を示すものではない。", "",
                  "- 学習forward・逆伝播dX：M≥128で新経路の対象。",
                  "- 既存の推論GGUF BSLM：M≥128かつ、(K≥8192かつM≥1024) または (N≥16384かつM≥4096) の形状で新経路の対象。他の形状では旧推論経路を維持。",
                  "- 残メモリ・デバイス対応・resident panel等の条件による既存経路への分岐は残る。表の対象判定は形状条件だけを示す。",
                  "- Mは実際に投影する行数。GUIのprefix chunkが1024なら、4096 tokenのpromptでも各chunkのMは最大1024となる。", ""]
    groups = [("推論 forward：旧GGUF BSLMとの比較", "original-inference-gguf-bslm"),
              ("学習 forward：旧FP32 rows4との比較", "original-training-fp32-rows4"),
              ("逆伝播 dX：旧vec8 / rows8 / reduceとの比較", "original-transpose-vec8-rows8-reduce")]
    for title, baseline in groups:
        lines += ["## " + title, ""]
        rows = [row for row in selected if row["baseline"] == baseline]
        if not rows:
            lines += ["測定データなし。", ""]
            continue
        headers = ["テンソル / M×K→N", "新候補", "旧wall ms", "新wall ms", "wall倍率", "旧GPU ms", "新GPU ms", "新wall 平均±σ (最小–最大) ms"]
        cells = [
            [f"{r['name']} / {r['rows']}×{r['inputWidth']}→{r['outputWidth']}", r["candidate"],
             f"{r['baselineWallMilliseconds']['median']:.3f}", f"{r['candidateWallMilliseconds']['median']:.3f}",
             f"{r['hostWallMedianSpeedup']:.3f}×", f"{r['baselineGpuMilliseconds']['median']:.3f}",
             f"{r['candidateGpuMilliseconds']['median']:.3f}", format_stats(r["candidateWallMilliseconds"])] for r in rows]
        if summary["productionPolicy"] != "unspecified":
            headers.append("本番の形状判定")
            for row, cell in zip(rows, cells):
                cell.append(production_route(row, summary["productionPolicy"]))
        lines += table(headers, cells)
        lines += ["", "旧経路を含む全候補の平均・母標準偏差・最小・最大と全サンプルはsummary.jsonに保存。", ""]
    lines += ["## 全モデルでの確認", ""]
    model = summary.get("model")
    if model is None:
        lines += ["全モデルの測定データは未指定。モデル全体の高速化率は未確認。", ""]
    else:
        lines += ["GUIの推論設定に合わせ、LoRAなし・画像なしで、旧モデルと新モデルを順にロードした。各回Reset後にprefixを処理し、次の1 tokenの全語彙logitsを取得。",
                  "prefixチェックポイントの保存とlogits読戻しをhost wallに含む。モデルロード・tokenizeは除外。旧→新の順で測定しているため、時間変化の影響は残る。実際の画面上のTTFTを直接測った値ではない。", ""]
        lines += [f"モデル測定：{devices_text(model['evidence']['device'])}。投影マイクロベンチとGPU構成が異なる場合、それぞれの旧/新同条件の比較として扱う。",
                  "この全モデル比較の変更範囲はIQ2 row-major BSLMとbatched text attention。投影表のIQ2単独の倍率とは別の測定。", ""]
        if not model["profileKernels"]:
            lines += ["主測定ではカーネルprofilingを無効にしてhost wallを測定した。GPU合計時間は未測定のため「—」。同じバイナリ・モデル・設定の別profile測定で使用カーネルと数値一致を確認した。", ""]
        lines += table(["prefix token (+次1)", "旧wall ms", "新wall ms", "wall倍率", "旧GPU合計 ms", "新GPU合計 ms", "logits相対L2", "Top1一致", "実行経路"], [
            [r["prefixTokens"], f"{r['baselineWallMilliseconds']['median']:.3f}", f"{r['candidateWallMilliseconds']['median']:.3f}",
             f"{r['hostWallMedianSpeedup']:.3f}×", median_text(r['baselineGpuMilliseconds']),
             median_text(r['candidateGpuMilliseconds']), f"{r['logitComparison']['relativeL2']:.3e}", "はい",
             model_route(r)] for r in model["comparisons"]])
        lines += ["", "全logitsファイルのSHA-256・長さ・有限値を再検査し、誤差とTop1を再計算した。複数GPUのイベント合計は実経過時間とは異なる。",
                  "256 tokenでIQ2が旧経路でもattentionは新経路になる。速度差にはattentionの効果と別セッション間の変動を含む。", ""]
        phase_rows = []
        for row in model["comparisons"]:
            for session, phases in row["phaseMedians"].items():
                if phases:
                    phase_rows.append([row["prefixTokens"], session] + [f"{phases[key]:.3f}" for key in
                                      ("resetMilliseconds", "primePrefixMilliseconds", "lastTokenMilliseconds")])
        if phase_rows:
            lines += ["### Host wallの区間別中央値", "",
                      "追加同期を入れずに連続するhost区間を記録。各回の3区間の和はwallと一致するが、各区間の中央値の和はwall中央値と必ずしも一致しない。", ""]
            lines += table(["prefix token", "経路", "Reset ms", "PrimePrefix ms", "LastToken ms"], phase_rows)
            lines += [""]
        profile = summary.get("modelProfile")
        if profile:
            lines += ["### 別profileによる経路監査", "",
                      "この監査は使用カーネルと全logitsの確認用。サンプル数が少ない場合もあり、速度倍率の主測定には使用しない。", ""]
            lines += table(["prefix token", "IQ2候補実行", "attention候補実行", "logits相対L2", "Top1一致"], [
                [row["prefixTokens"], "はい" if row["candidateKernelExercised"] else "旧IQ2を維持",
                 "はい" if row["candidateAttentionExercised"] else "いいえ", f"{row['logitComparison']['relativeL2']:.3e}", "はい"]
                for row in profile["comparisons"]])
            lines += [""]
    lines += ["## 数値検証", "",
              "計測前後の全出力について有限値・ガード領域・旧経路との相対L2を検査し、独立したCPU double内積との誤差をサンプル点で確認した。以下は新候補の前後2回の最大値。", ""]
    error_rows = []
    for case in summary["cases"]:
      for selected_name in sorted(set(case["selectedCandidates"].values())):
        variant = next(item for item in case["variants"] if item["name"] == selected_name)
        values = [variant["validation"], variant["postTimingValidation"]]
        error_rows.append([f"{case['direction']} / {case['name']} / M={case['rows']} / {selected_name}", case["cpuPoints"],
                           f"{max(v['maxAbsoluteErrorVsDouble'] for v in values):.3e}",
                           f"{max(v['relativeL2VsDouble'] for v in values):.3e}",
                           f"{max(v['relativeL2VsOriginal'] for v in values):.3e}",
                           f"{max(v['worstToleranceRatio'] for v in values):.3f}",
                           "はい" if all(v["bitwiseEqualToOriginal"] for v in values) else "いいえ"])
    lines += table(["ケース", "CPU点数", "CPU最大絶対誤差", "CPU相対L2", "旧参照との全出力相対L2", "CPU許容比 ≤1", "旧参照とbit一致"], error_rows)
    lines += ["", "forwardの全出力参照は旧推論BSLM。学習rows4も同じ参照およびCPU doubleで検証した。逆伝播の参照は旧transpose経路。bit一致と許容誤差内の一致を区別して記録した。", "",
              "## 入力・再現条件", ""]
    lines += table(["方向 / テンソル", "M", "GGUF形状 [K,N]", "FP32 seed", "入力範囲・種類", "重みpayload SHA-256"], [
        [f"{case['direction']} / {case['name']}", case["rows"], case["source"].get("shape", "合成データ"),
         case["activation"]["seed"],
         ("±100000を含む範囲外試験" if case["fallback"] else "決定的乱数・通常範囲 [-2,2)") + (" / 合成重み" if "shape" not in case["source"] else " / 実GGUF重み"),
         case["source"]["payloadSha256"]] for case in summary["cases"]])
    lines += ["", "入力SHA-256、重みoffset/byte数、bias、全CLI引数、デバイス・ドライバー、サンプル数・warmup回数はsummary.jsonのcases/evidenceに保存。", ""]
    for evidence in summary["evidence"]:
        timing = evidence["timing"]
        lines += [f"- {link(evidence['path'])}：{devices_text(evidence['device'])}。warmup {timing['warmup']}回、{timing['samples']}サンプル。round robinで奇数roundは逆順。"]
    if model:
        lines += [f"- {link(model['evidence']['path'])}：モデルwarmup {model['evidence']['timing']['warmup']}回、{model['evidence']['timing']['sampleCount']}サンプル。"]
    if summary.get("modelProfile"):
        item = summary["modelProfile"]["evidence"]
        lines += [f"- {link(item['path'])}：別profile監査。warmup {item['timing']['warmup']}回、{item['timing']['sampleCount']}サンプル。"]
    lines += ["", "## 極小値・範囲外の単体試験", ""]
    if not summary["tests"]:
        lines += ["TRX未指定。1e-20などの範囲試験について、このレポートには検証済みの証拠を添付していない。", ""]
    for test in summary["tests"]:
        lines += [f"- {link(test['path'])}：{test['passed']}件すべて成功。"]
        if test["tinyNormalAmplitudes"]:
            values = ", ".join(f"{value:.2e}" for value in test["tinyNormalAmplitudes"])
            lines += [f"  tiny-normal両方向の試験振幅：{values}。これは上記の速度測定とは別の単体試験。"]
    lines += ["", "## 来歴とファイル", "",
              "入力JSONのcomplete、前後の数値検証、全サンプルの統計値、記録されたカーネル時間合計を再検査済み。共通の実装バイナリおよびソースのSHA-256は入力間で一致。失敗した測定は含めない。", ""]
    all_evidence = summary["evidence"] + ([model["evidence"]] if model else []) + ([summary["modelProfile"]["evidence"]] if summary.get("modelProfile") else [])
    lines += table(["証拠JSON", "JSON SHA-256"], [[link(item["path"]), item["sha256"]] for item in all_evidence])
    lines += [""] + table(["バイナリ / ソース", "SHA-256"], [[name, value] for name, value in summary["commonImplementationSha256"].items()])
    lines += ["", f"全数値・全候補・全サンプル・入力/ソースSHA一覧：{link(output / 'summary.json')}", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, nargs="+", help="Completed forward and/or transpose results.json files")
    parser.add_argument("--model-results", type=Path, help="Optional completed model-results.json with adjacent logits")
    parser.add_argument("--model-profile", type=Path, help="Separate profiled model route evidence, required for no-profile model results")
    parser.add_argument("--output", type=Path, required=True, help="Destination directory for report.ja.md and summary.json")
    parser.add_argument("--test-trx", type=Path, action="append", default=[], help="Passing unit-test TRX (repeatable)")
    parser.add_argument("--minimum-samples", type=int, default=9, help="Default 9; lower explicitly for a preliminary report")
    parser.add_argument("--minimum-warmup", type=int, default=3, help="Default 3; lower explicitly for a preliminary report")
    parser.add_argument("--allow-synthetic", action="store_true", help="Permit synthetic weight cases, clearly labelled")
    parser.add_argument("--expect-rows", default="", help="Optional exact M coverage per tensor/direction, e.g. 1024,4096,8192")
    parser.add_argument("--expect-directions", default="", help="Optional exact direction coverage, e.g. forward,transpose")
    parser.add_argument("--expect-tensors", default="", help="Optional exact tensor suffix coverage per direction, e.g. attn_gate,ffn_gate,ffn_down")
    parser.add_argument("--production-policy", choices=("unspecified", "arc-b580-hybrid", "rowmajor-bslm"), default="unspecified",
                        help="Explicit production dispatch annotation; never inferred from winning candidates")
    args = parser.parse_args(argv)
    require(args.minimum_samples >= 1 and args.minimum_warmup >= 0, "Invalid minimum samples/warmup")
    paths = [path.resolve(strict=True) for path in args.results]
    require(len(paths) == len(set(paths)), "Input JSON paths must be distinct")
    evidence, cases, comparisons = [], [], []
    for path in paths:
        item, rows, cmp = validate_micro(path, args)
        evidence.append(item)
        cases.extend(rows)
        comparisons.extend(cmp)
    case_keys = [(case["direction"], case["name"], case["rows"]) for case in cases]
    require(len(case_keys) == len(set(case_keys)), "Duplicate direction/tensor/M across JSON inputs")
    actual_directions = {case["direction"] for case in cases}
    if args.expect_directions:
        require(actual_directions == set(args.expect_directions.split(',')), "Incomplete direction coverage")
    if args.expect_tensors:
        expected_tensors = set(args.expect_tensors.split(','))
        require(all(expected_tensors), "--expect-tensors contains an empty suffix")
        for direction in actual_directions:
            actual_names = {case["name"] for case in cases if case["direction"] == direction}
            matches = {suffix: [name for name in actual_names if name == suffix or name.endswith('.' + suffix + '.weight')]
                       for suffix in expected_tensors}
            require(all(len(names) == 1 for names in matches.values())
                    and {name for names in matches.values() for name in names} == actual_names,
                    f"Incomplete/unexpected tensor coverage for {direction}: {sorted(actual_names)}")
    if args.expect_rows:
        expected_rows = set(map(int, args.expect_rows.split(",")))
        require(expected_rows and min(expected_rows) > 0, "--expect-rows must be positive")
        for direction, name in {(case["direction"], case["name"]) for case in cases}:
            actual_rows = {case["rows"] for case in cases if case["direction"] == direction and case["name"] == name}
            require(actual_rows == expected_rows, f"Incomplete M coverage for {direction}/{name}: {actual_rows}")
    model = validate_model(args.model_results.resolve(strict=True)) if args.model_results else None
    require(not args.model_profile or model is not None, "--model-profile requires --model-results")
    profile = validate_model(args.model_profile.resolve(strict=True), profile_evidence=True) if args.model_profile else None
    if model and (not model["profileKernels"] or profile):
        validate_profile_pair(model, profile)
    if model and args.expect_rows:
        require({row["prefixTokens"] for row in model["comparisons"]} == expected_rows,
                "Full model prefix coverage differs from --expect-rows")
    if args.production_policy == "arc-b580-hybrid":
        require(all(re.search(r"(?:^|-)16x32(?:-|$)", name) for case in cases for name in case["selectedCandidates"].values()),
                "Hybrid policy is 16x32, but a reported best candidate uses a different tile; report that experiment separately")
    elif args.production_policy == "rowmajor-bslm":
        for case in cases:
            for baseline, name in case["selectedCandidates"].items():
                if baseline == "original-inference-gguf-bslm":
                    require(name == "inference-bslm-k32r", "Selected inference candidate differs from rowmajor-bslm production policy")
                else:
                    require(re.search(r"(?:^|-)16x32(?:-|$)", name), "Selected learning candidate differs from 16x32 production policy")
    # Refuse to silently combine different implementation versions. The reporting
    # script itself does not need to match because it is not timed GPU code.
    implementations = {}
    for item in evidence + ([model["evidence"]] if model else []) + ([profile["evidence"]] if profile else []):
        for category in ("measuredFileSha256", "sourceFileSha256"):
            for name, value in item[category].items():
                key = name.replace("\\", "/")
                require(key not in implementations or implementations[key] == value,
                        f"Mixed implementation builds for {name}; report runs separately")
                implementations[key] = value
    tests = [validate_trx(path.resolve(strict=True)) for path in args.test_trx]
    summary = {"schemaVersion": 1, "validatedComplete": True,
               "generatedUtc": dt.datetime.now(dt.timezone.utc).isoformat(),
               "method": "Host wall median baseline/candidate ratio. Fastest measured candidate per case; no production default inferred.",
               "minimumSamplesRequired": args.minimum_samples, "minimumWarmupRequired": args.minimum_warmup,
               "expectedRows": args.expect_rows, "expectedDirections": args.expect_directions,
               "expectedTensors": args.expect_tensors,
               "productionPolicy": args.production_policy,
               "evidence": evidence, "cases": cases, "comparisons": comparisons, "model": model, "modelProfile": profile, "tests": tests,
               "commonImplementationSha256": {key: value for key, value in implementations.items()
                                              if key in ("NNtrain.Arc.dll", "NNtrain.Core.dll")
                                              or key.endswith(("qwen35_iq2_tiles.cl", "qwen35_prefill_xmx.cl", "qwen35_attention.cl",
                                                               "ArcExecutionLane.Iq2.cs", "Qwen35ExecutionOptions.cs",
                                                               "Qwen35QuantizedModel.cs", "Qwen35QuantizedModel.Prefill.cs"))}}
    output = args.output.resolve()
    for target in (output / "summary.json", output / "report.ja.md"):
        require(target not in paths and target != (args.model_results.resolve() if args.model_results else None),
                "Refusing to overwrite input evidence")
        require(target != (args.model_profile.resolve() if args.model_profile else None), "Refusing to overwrite profile evidence")
    report = render(summary, output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "report.ja.md").write_text(report, encoding="utf-8")
    print(f"Validated {len(cases)} projection cases; {len(comparisons)} comparisons; model={'yes' if model else 'no'}")
    print(output / "report.ja.md")
    print(output / "summary.json")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (EvidenceError, KeyError, TypeError, OSError, json.JSONDecodeError, ET.ParseError) as error:
        print(f"Invalid benchmark evidence: {error}", file=sys.stderr)
        sys.exit(2)
