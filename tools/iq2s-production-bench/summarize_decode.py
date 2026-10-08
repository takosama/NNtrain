#!/usr/bin/env python3
"""Validate Qwen35GenerationProbe schema-2 artifacts and report decode throughput.

CPU only, Python standard library. Example:
  python summarize_decode.py original.json --candidate rms=rms.json \
    --candidate combined=combined.json --require-logits --output decode-report

Requires >=4 runs by default; run 0 is excluded from performance statistics but
retained in the report and all generated-ID/logit consistency checks.
"""

from __future__ import annotations

import argparse
import array
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import xml.etree.ElementTree as ET


PLANNED_OPTION_CHANGES = frozenset(("InferenceFusedResidualRms", "InferenceCachedDeltaDecode",
                                  "InferenceResidentIq2Panels", "InferenceXmxGgufBslmPrefill"))


class EvidenceError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def number(value, label, *, positive=False):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (value > 0 if positive else value >= 0),
            f"{label}: finite {'positive' if positive else 'nonnegative'} number required")
    return float(value)


def integer(value, label, minimum=0):
    require(isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
            f"{label}: integer >= {minimum} required")
    return value


def near(value, expected, label):
    require(math.isclose(number(value, label), expected, rel_tol=1e-9, abs_tol=1e-7),
            f"{label}: stored {value}, recomputed {expected}")


def digest(value, label):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value), f"{label}: SHA-256 required")
    return value.lower()


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def parse_json(text, label):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"{label}: duplicate JSON key {key}")
            result[key] = value
        return result
    def invalid(value):
        raise EvidenceError(f"{label}: invalid JSON constant {value}")
    return json.loads(text, object_pairs_hook=unique, parse_constant=invalid)


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def stats(values):
    require(values, "No samples to summarize")
    return {"count": len(values), "median": statistics.median(values),
            "mean": statistics.fmean(values), "populationStandardDeviation": statistics.pstdev(values),
            "minimum": min(values), "maximum": max(values), "p95": percentile(values, .95)}


def ids_check(values, label, vocabulary):
    require(isinstance(values, list) and values, f"{label}: token IDs required")
    require(all(isinstance(value, int) and not isinstance(value, bool) and 0 <= value < vocabulary
                for value in values), f"{label}: invalid token ID")
    return values


def logits_check(path, document, required):
    metadata = document.get("Logits")
    if metadata is None:
        require(not required, f"{path}: --require-logits needs full logit snapshots")
        return None
    logits_path = Path(str(path) + ".logits.f32")
    require(logits_path.is_file(), f"{path}: adjacent logits file missing: {logits_path}")
    actual_hash = sha256(logits_path)
    require(actual_hash == digest(metadata.get("Sha256"), str(path) + ".Logits.Sha256"), f"{path}: logits SHA mismatch")
    require(metadata.get("Format") == "little-endian float32, row-major [snapshot, vocabulary]",
            f"{path}: unsupported logits format")
    snapshots = metadata.get("Snapshots")
    require(isinstance(snapshots, list) and snapshots, f"{path}: logits snapshots missing")
    vocabulary = document["Shape"]["VocabularySize"]
    generated = document["Samples"][0]["GeneratedTokenIds"]
    require(len(snapshots) <= len(generated), f"{path}: logits must follow the measured continuation")
    require(logits_path.stat().st_size == len(snapshots) * vocabulary * 4, f"{path}: wrong logits file length")
    with logits_path.open("rb") as stream:
        for index, snapshot in enumerate(snapshots):
            require(snapshot.get("Step") == index and snapshot.get("VocabularySize") == vocabulary
                    and snapshot.get("ByteOffset") == index * vocabulary * 4, f"{path}: invalid snapshot shape/order")
            expected_input = document["PromptTokenIds"][-1] if index == 0 else generated[index - 1]
            require(snapshot.get("InputTokenId") == expected_input, f"{path}: snapshot teacher input differs from measured continuation")
            values = array.array("f")
            values.frombytes(stream.read(vocabulary * 4))
            if sys.byteorder != "little":
                values.byteswap()
            require(len(values) == vocabulary and all(math.isfinite(value) for value in values),
                    f"{path}: nonfinite/truncated full logits")
            top = max(range(vocabulary), key=values.__getitem__)
            require(snapshot.get("GreedyTokenId") == top, f"{path}: logit argmax metadata differs")
            if document["Sampling"]["Method"] == "GPU greedy argmax":
                require(top == generated[index], f"{path}: logit greedy IDs differ from measured output")
    return {"path": str(logits_path), "sha256": actual_hash, "bytes": logits_path.stat().st_size,
            "snapshotCount": len(snapshots), "vocabularySize": vocabulary,
            "allFinite": True, "matchesMeasuredContinuation": True, "metadata": metadata}


def validate(path, label, args):
    document = parse_json(path.read_text(encoding="utf-8-sig"), str(path))
    require(document.get("SchemaVersion") == 2 and document.get("CompletedUtc") and document.get("StartedUtc"),
            f"{path}: completed Qwen35GenerationProbe schema 2 required")
    require(document.get("Adapter") is None and document.get("Options", {}).get("LoraTraining") is False,
            f"{path}: this report requires base-model inference without LoRA")
    model = document.get("Model", {})
    digest(model.get("Sha256"), str(path) + ".Model.Sha256")
    integer(model.get("Bytes"), str(path) + ".Model.Bytes", 1)
    binaries = document.get("BinarySha256", {})
    require(isinstance(binaries, dict) and all(key in binaries for key in ("NNtrain.Core", "NNtrain.Arc", "NNtrain.Benchmarks")),
            f"{path}: binary provenance missing")
    for key, value in binaries.items():
        digest(value, str(path) + ".BinarySha256." + key)
    devices = document.get("Devices")
    require(isinstance(devices, list) and devices and len({item["Index"] for item in devices}) == len(devices),
            f"{path}: distinct devices required")
    sampling = document.get("Sampling", {})
    require(sampling.get("EosTokenId") is None, f"{path}: expected fixed output length without EOS stop")
    method = sampling.get("Method")
    temperature = number(sampling.get("Temperature"), str(path) + ".Sampling.Temperature")
    if method == "GPU greedy argmax":
        require(temperature == 0 or sampling.get("TopK") == 1, f"{path}: inconsistent greedy sampling")
    else:
        require(method == "Seeded CPU top-k/top-p" and temperature > 0,
                f"{path}: unsupported sampling policy")
        integer(sampling.get("TopK"), str(path) + ".Sampling.TopK", 2)
        integer(sampling.get("RandomSeed"), str(path) + ".Sampling.RandomSeed")
        require(0 < number(sampling.get("TopP"), str(path) + ".Sampling.TopP") <= 1,
                f"{path}: invalid nucleus sampling mass")
    vocabulary = integer(document.get("Shape", {}).get("VocabularySize"), str(path) + ".vocabulary", 1)
    prompt_ids = ids_check(document.get("PromptTokenIds"), str(path) + ".PromptTokenIds", vocabulary)
    requested = integer(document.get("RequestedNewTokens"), str(path) + ".RequestedNewTokens", 2)
    count = integer(document.get("MeasuredRuns"), str(path) + ".MeasuredRuns", args.minimum_runs)
    samples = document.get("Samples", [])
    require(len(samples) == count and [sample.get("RunIndex") for sample in samples] == list(range(count)),
            f"{path}: missing/nonconsecutive runs")
    require(count > args.discard_first, f"{path}: all runs were discarded")
    require(document.get("RunsHaveIdenticalGeneratedTokenIds") is True, f"{path}: generated IDs vary between runs")
    recalculated = []
    expected_generated = None
    for sample in samples:
        run = sample["RunIndex"]
        where = f"{path}.Samples[{run}]"
        generated = ids_check(sample.get("GeneratedTokenIds"), where + ".GeneratedTokenIds", vocabulary)
        require(len(generated) == requested and sample.get("AllTokenIds") == prompt_ids + generated,
                where + ": generated count/prompt mismatch")
        expected_generated = generated if expected_generated is None else expected_generated
        require(generated == expected_generated and sample.get("MatchesFirstRunTokenIds") is True,
                where + ": generated sequence differs from first run")
        tokens = sample.get("Tokens", [])
        require(len(tokens) == requested, where + ": missing token timestamps")
        times, intervals = [], []
        for index, token in enumerate(tokens):
            require(token.get("GeneratedIndex") == index and token.get("TokenId") == generated[index],
                    where + ": callback token ID/index mismatch")
            elapsed = number(token.get("ElapsedMilliseconds"), where + f".Tokens[{index}].ElapsedMilliseconds", positive=True)
            if index == 0:
                require(token.get("SincePreviousTokenMilliseconds") is None, where + ": first-token interval must be null")
            else:
                interval = elapsed - times[-1]
                require(interval > 0, where + ": token timestamps must strictly increase")
                near(token.get("SincePreviousTokenMilliseconds"), interval, where + ": token interval")
                intervals.append(interval)
            times.append(elapsed)
        duration = times[-1] - times[0]
        throughput = (requested - 1) * 1000.0 / duration
        near(sample.get("PrefillThroughFirstTokenMilliseconds"), times[0], where + ": first-token time")
        near(sample.get("DecodeMilliseconds"), duration, where + ": decode time")
        near(sample.get("DecodeTokensPerSecond"), throughput, where + ": decode throughput")
        require(number(sample.get("TotalMilliseconds"), where + ": total time", positive=True) >= times[-1],
                where + ": total time precedes last callback")
        recalculated.append({"runIndex": run, "included": run >= args.discard_first,
                             "firstTokenMilliseconds": times[0], "decodeMilliseconds": duration,
                             "decodedIntervals": requested - 1, "tokensPerSecond": throughput,
                             "tokenIntervalMilliseconds": intervals, "intervalStatistics": stats(intervals)})
    included = [run for run in recalculated if run["included"]]
    return {"label": label, "path": str(path), "sha256": sha256(path), "raw": document,
            "recalculatedRuns": recalculated, "includedRunCount": len(included),
            "tokensPerSecond": stats([run["tokensPerSecond"] for run in included]),
            "decodeMilliseconds": stats([run["decodeMilliseconds"] for run in included]),
            "tokenIntervalMilliseconds": stats([value for run in included for value in run["tokenIntervalMilliseconds"]]),
            "generatedTokenIds": expected_generated, "logits": logits_check(path, document, args.require_logits)}


def compare_bytes(first, second):
    with Path(first).open("rb") as a, Path(second).open("rb") as b:
        while True:
            left, right = a.read(1024 * 1024), b.read(1024 * 1024)
            if left != right:
                return False
            if not left:
                return True


def compare(baseline, candidate, args):
    old, new = baseline["raw"], candidate["raw"]
    label = candidate["label"]
    for key in ("Devices", "Shape", "Runtime", "PromptTokenIds", "RequestedNewTokens", "Sampling", "BinarySha256"):
        require(old[key] == new[key], f"{label}: {key} differs from baseline")
    require(old["Model"]["Sha256"].lower() == new["Model"]["Sha256"].lower()
            and old["Model"]["Bytes"] == new["Model"]["Bytes"], f"{label}: model SHA/size differs")
    require(old["Prompt"] == new["Prompt"], f"{label}: prompt text differs")
    require(baseline["generatedTokenIds"] == candidate["generatedTokenIds"], f"{label}: generated IDs differ from baseline")
    for key in ("CollectKernelTimings", "DetailedProfiling"):
        require(old["Options"].get(key) == new["Options"].get(key), f"{label}: profiling settings differ")
    changed = {key: {"original": old["Options"].get(key), "candidate": new["Options"].get(key)}
               for key in sorted(set(old["Options"]) | set(new["Options"]))
               if old["Options"].get(key) != new["Options"].get(key)}
    allowed_changes = PLANNED_OPTION_CHANGES | set(getattr(args, "allow_option_change", []))
    require(not (set(changed) - allowed_changes),
            f"{label}: unplanned option changes {sorted(set(changed) - allowed_changes)}; declare intentional extra switches with --allow-option-change")
    old_logits, new_logits = baseline["logits"], candidate["logits"]
    equal_logits = None
    if old_logits is not None and new_logits is not None:
        require(old_logits["snapshotCount"] == new_logits["snapshotCount"]
                and old_logits["vocabularySize"] == new_logits["vocabularySize"], f"{label}: logits shapes differ")
        equal_logits = old_logits["sha256"] == new_logits["sha256"] and compare_bytes(old_logits["path"], new_logits["path"])
        require(equal_logits, f"{label}: complete logits bytes differ")
    require(not args.require_logits or equal_logits is True, f"{label}: required logits comparison unavailable")
    return {"candidate": label, "medianTokensPerSecondRatio": candidate["tokensPerSecond"]["median"] / baseline["tokensPerSecond"]["median"],
            "meanTokensPerSecondRatio": candidate["tokensPerSecond"]["mean"] / baseline["tokensPerSecond"]["mean"],
            "allGeneratedTokenIdsEqual": True, "fullLogitBytesEqual": equal_logits,
            "changedOptions": changed}


def delta_appendix(path):
    root = ET.parse(path).getroot()
    results = list(root.iterfind(".//{*}UnitTestResult"))
    require(results and all(item.get("outcome") == "Passed" for item in results), f"{path}: TRX includes nonpassing tests")
    experiments = []
    for node in root.iter():
        if node.tag.rsplit("}", 1)[-1] not in ("StdOut", "StdErr"):
            continue
        for line in (node.text or "").splitlines():
            start = line.find('{"benchmark": "DeltaStepFused')
            if start < 0:
                start = line.find('{"benchmark":"DeltaStepFused')
            if start >= 0:
                value = parse_json(line[start:], str(path))
                require(value.get("complete") is True and value.get("outputsAndStatesBitwiseEqual") is True,
                        f"{path}: incomplete or unequal DeltaNet experiment")
                count = integer(value.get("samples"), str(path) + ".Delta.samples", 1)
                integer(value.get("warmup"), str(path) + ".Delta.warmup")
                integer(value.get("updatesPerSample"), str(path) + ".Delta.updatesPerSample", 1)
                raw = value.get("rawSamples", [])
                require(len(raw) == 2 * count, f"{path}: incomplete DeltaNet samples")
                for cached, label in ((False, "original"), (True, "candidate")):
                    rows = sorted((item for item in raw if item.get("Cached") is cached), key=lambda item: item["Round"])
                    require([item["Round"] for item in rows] == list(range(count)), f"{path}: invalid DeltaNet rounds")
                    metrics = [("WallMilliseconds", label + "WallPerUpdate")]
                    if value["profile"]:
                        metrics.append(("GpuMilliseconds", label + "GpuPerUpdate"))
                        for sample in rows:
                            total = math.fsum(number(duration, str(path) + ".Delta.kernel")
                                              for duration in sample["KernelMilliseconds"].values())
                            near(sample["GpuMilliseconds"], total, str(path) + ".Delta.kernelSum")
                    else:
                        require(value.get(label + "GpuPerUpdate") is None
                                and all(item.get("GpuMilliseconds") is None and item.get("KernelMilliseconds") == {} for item in rows),
                                f"{path}: unprofiled DeltaNet GPU timings must be absent")
                    for metric, stored_key in metrics:
                        actual = stats([number(item[metric], str(path) + ".Delta." + metric, positive=True) for item in rows])
                        stored = value[stored_key]
                        require(stored.get("count") == count, f"{path}: DeltaNet statistic count mismatch")
                        for key in ("median", "mean", "minimum", "maximum", "populationStandardDeviation"):
                            stored_name = "standardDeviation" if key == "populationStandardDeviation" else key
                            near(stored.get(stored_name), actual[key], str(path) + ".Delta." + stored_name)
                near(value.get("hostWallMedianSpeedup"), value["originalWallPerUpdate"]["median"]
                     / value["candidateWallPerUpdate"]["median"], str(path) + ".Delta.speedup")
                experiments.append(value)
    require(experiments, f"{path}: no optional DeltaNet benchmark JSON found")
    return {"path": str(path), "sha256": sha256(path), "experiments": experiments,
            "scope": "Separate one-layer DeltaNet microbenchmark; not full-model decode throughput or production selection."}


def escape(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def link(path, relative_to):
    target = Path(os.path.relpath(path, relative_to)).as_posix()
    return f"[{Path(path).name}](<{target}>)"


def table(headers, rows):
    return ["| " + " | ".join(map(escape, headers)) + " |", "| " + " | ".join("---" for _ in headers) + " |"] + [
        "| " + " | ".join(map(escape, row)) + " |" for row in rows]


def render(summary, output):
    records = [summary["baseline"]] + summary["candidates"]
    raw = records[0]["raw"]
    device_text = "; ".join(f"GPU {item['Index']}: {item['Name']}" for item in raw["Devices"])
    sampling = raw["Sampling"]
    sampling_text = ("greedy生成" if sampling["Method"] == "GPU greedy argmax" else
                     f"seed固定sampling（temperature={sampling['Temperature']}、top-p={sampling['TopP']}、top-k={sampling['TopK']}、seed={sampling['RandomSeed']}）")
    lines = ["# Qwen GGUF 生成速度の比較", "", "## 測定範囲", "",
             f"同じベースモデル・{len(raw['Devices'])} GPU（{device_text}）・{len(raw['PromptTokenIds'])} tokenの同じprompt・{sampling_text}・LoRAなしで比較。各runで{raw['RequestedNewTokens']} tokenを生成。EOSによる早期終了はしない。",
             "最初の出力tokenから最後の出力tokenまでをdecode時間とし、`(生成数−1)×1000 / decode ms`でtok/sを再計算した。prompt処理と最初のtoken生成、モデルロード、logits保存はこの指標に含めない。",
             f"先頭{summary['discardFirstRuns']} runを速度集計から除外。除外runも生成ID・時刻・logits整合性の検証対象。実行順は各JSONの時刻に記録されており、別プロセス/別セッション間の変動を含む。",
             "全モデルのdecode測定であり、既存のprefillやIQ2演算単体の倍率とは別の値。", "",
             f"カーネルtiming収集：{raw['Options'].get('CollectKernelTimings')}、詳細profiling：{raw['Options'].get('DetailedProfiling')}。旧/新で同じ設定であることを検査。", "", "## 生成速度", ""]
    ratios = {item["candidate"]: item["medianTokensPerSecondRatio"] for item in summary["comparisons"]}
    lines += table(["経路", "採用run数", "中央値 tok/s", "平均 tok/s", "母標準偏差", "最小–最大 tok/s", "中央値の旧比", "token間隔中央値 ms", "token間隔p95 ms"], [
        [item["label"], item["includedRunCount"], f"{item['tokensPerSecond']['median']:.3f}",
         f"{item['tokensPerSecond']['mean']:.3f}", f"{item['tokensPerSecond']['populationStandardDeviation']:.3f}",
         f"{item['tokensPerSecond']['minimum']:.3f}–{item['tokensPerSecond']['maximum']:.3f}",
         f"{ratios[item['label']]:.3f}×" if item["label"] in ratios else "基準",
         f"{item['tokenIntervalMilliseconds']['median']:.3f}", f"{item['tokenIntervalMilliseconds']['p95']:.3f}"] for item in records])
    lines += ["", "tok/sの統計単位はrun。token間隔は採用runの全区間をまとめ、p95は昇順の位置 `(n−1)×0.95` を線形補間して計算。倍率1超が高速化を示す。少数runの記述統計であり、統計的有意差や普遍的な高速化を示すものではない。", "",
              "## 一致検証と設定差", ""]
    for item in summary["comparisons"]:
        lines += [f"### {escape(item['candidate'])}", "", "全run・全生成token ID：旧と完全一致。"]
        if item["fullLogitBytesEqual"]:
            record = next(value for value in records if value["label"] == item["candidate"])
            lines += [f"全語彙logits：{record['logits']['snapshotCount']} snapshotの全バイトが旧と一致。SHA-256、サイズ、有限値、入力continuation、argmax IDを再検査。"]
        else:
            lines += ["全語彙logitsの比較は未実施。生成ID一致のみを検証。"]
        lines += [""] + table(["設定", "旧", "候補"], [[key, value["original"], value["candidate"]]
                                                       for key, value in item["changedOptions"].items()]) + [""]
    if summary["deltaExperiments"]:
        lines += ["## 参考：単一DeltaNet更新の実験", "",
                  "以下は別の単体測定。上の全モデルtok/sには加算・乗算しない。遅い候補は高速化候補として採用しない。", ""]
        for artifact in summary["deltaExperiments"]:
            lines += [f"出典：{link(artifact['path'], output)}", ""]
            lines += table(["profiling", "旧wall ms/更新", "候補wall ms/更新", "旧/候補倍率", "出力・状態"], [
                [item["profile"], f"{item['originalWallPerUpdate']['median']:.6f}",
                 f"{item['candidateWallPerUpdate']['median']:.6f}", f"{item['hostWallMedianSpeedup']:.3f}×", "bit一致"]
                for item in artifact["experiments"]]) + [""]
    lines += ["## 証拠と再現条件", "", f"モデルSHA-256：`{raw['Model']['Sha256']}`", ""]
    lines += table(["経路", "生JSON", "SHA-256", "開始UTC", "終了UTC"], [
        [item["label"], link(item["path"], output), item["sha256"], item["raw"]["StartedUtc"], item["raw"]["CompletedUtc"]] for item in records])
    lines += [""] + table(["測定バイナリ", "SHA-256"], raw["BinarySha256"].items())
    lines += ["", "元JSONとlogitsファイルは変更していない。全rawデータ、再計算した各run・token間隔、設定差、完全一致の結果は以下に保存。",
              link(output / "decode-summary.json", output), ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--baseline-name", default="original")
    parser.add_argument("--candidate", action="append", required=True, metavar="NAME=JSON")
    parser.add_argument("--require-logits", action="store_true")
    parser.add_argument("--allow-option-change", action="append", default=[], metavar="OPTION",
                        help="Explicitly allow an additional option difference beyond the planned RMS, cached Delta, resident IQ2 and GGUF BSLM switches")
    parser.add_argument("--minimum-runs", type=int, default=4)
    parser.add_argument("--discard-first", type=int, default=1, help="Number of initial runs excluded from statistics; default run 0 only")
    parser.add_argument("--delta-trx", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require(args.minimum_runs >= 2 and 0 <= args.discard_first < args.minimum_runs,
            "Require >=2 runs and a discard count below --minimum-runs")
    named = [(args.baseline_name, args.baseline.resolve(strict=True))]
    for value in args.candidate:
        label, separator, filename = value.partition("=")
        require(separator and label.strip() and filename, "--candidate expects NAME=JSON")
        named.append((label.strip(), Path(filename).resolve(strict=True)))
    require(len({label for label, _ in named}) == len(named), "Duplicate baseline/candidate labels")
    require(len({path for _, path in named}) == len(named), "A result JSON cannot be used twice")
    records = [validate(path, label, args) for label, path in named]
    comparisons = [compare(records[0], record, args) for record in records[1:]]
    summary = {"schemaVersion": 1, "validatedComplete": True,
               "generatedUtc": dt.datetime.now(dt.timezone.utc).isoformat(),
               "scope": "Actual full-model decode between first and last callbacks; prefill/first output excluded. Sampling method and seed are recorded in the raw results.",
               "minimumRunsRequired": args.minimum_runs, "discardFirstRuns": args.discard_first,
               "allowedOptionChanges": sorted(PLANNED_OPTION_CHANGES | set(args.allow_option_change)),
               "requireLogitBytesEqual": args.require_logits,
               "baseline": records[0], "candidates": records[1:], "comparisons": comparisons,
               "deltaExperiments": [delta_appendix(path.resolve(strict=True)) for path in args.delta_trx]}
    output = args.output.resolve()
    protected = {path for _, path in named} | {Path(record["logits"]["path"]) for record in records if record["logits"]}
    protected |= {path.resolve() for path in args.delta_trx}
    for target in (output / "report.decode.ja.md", output / "decode-summary.json"):
        require(target not in protected, "Refusing to overwrite raw evidence")
    report = render(summary, output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.decode.ja.md").write_text(report, encoding="utf-8")
    (output / "decode-summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Validated {len(records)} generation probes; all runs and generated IDs match.")
    print(output / "report.decode.ja.md")
    print(output / "decode-summary.json")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (EvidenceError, KeyError, TypeError, OSError, json.JSONDecodeError, ET.ParseError) as error:
        print(f"Invalid decode evidence: {error}", file=sys.stderr)
        sys.exit(2)
