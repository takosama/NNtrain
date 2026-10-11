"""Verify saved evidence and assemble the Japanese release report."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET

out = Path(__file__).resolve().parent
root = out.parents[1]
def read(path):
    return json.loads(path.read_text(encoding="utf-8"))
def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

rows = []
summaries = {}
for mode, title in [("greedy", "通常生成（greedy）"), ("sampling", "GUI相当サンプリング"), ("lora", "実LoRA適用・greedy")]:
    command = [sys.executable, str(root / "tools/iq2s-production-bench/summarize_decode.py"),
        str(out / f"final-{mode}-original.json"), "--candidate", f"packed={out / f'final-{mode}-packed.json'}",
        "--allow-option-change", "InferencePackedIq2Decode", "--require-logits", "--output", str(out / f"{mode}-report")]
    if mode == "lora":
        command.append("--allow-matching-adapter")
    subprocess.run(command, check=True)
    s = read(out / f"{mode}-report/decode-summary.json")
    assert s["validatedComplete"] and s["comparisons"][0]["fullLogitBytesEqual"]
    a, b = s["baseline"], s["candidates"][0]
    assert a["raw"]["AfterLoad"]["LiveDeviceBytes"] == b["raw"]["AfterLoad"]["LiveDeviceBytes"]
    sa, sb = a["tokensPerSecond"], b["tokensPerSecond"]
    ratio = s["comparisons"][0]["medianTokensPerSecondRatio"]
    assert ratio > 1, f"No speedup for {mode}"
    rows.append(f"| {title} | {sa['median']:.3f} | {sb['median']:.3f} | +{(ratio-1)*100:.2f}% | {sa['mean']:.3f} ± {sa['populationStandardDeviation']:.3f} | {sb['mean']:.3f} ± {sb['populationStandardDeviation']:.3f} |")
    summaries[mode] = s

prefill = read(out / "prefill-long/model-results.json")
assert prefill["complete"] and prefill["numericGate"]["bitwiseRequired"]
pr = []
for length in (4096, 8192):
    a = next(p for p in prefill["passes"] if not p["tiled"] and p["prefixTokens"] == length)
    b = next(p for p in prefill["passes"] if p["tiled"] and p["prefixTokens"] == length)
    assert sha(out / f"prefill-long/original-prefix{length}.logits.f32") == sha(out / f"prefill-long/tiled-prefix{length}.logits.f32")
    pr.append(f"| {length:,} | {a['wallMilliseconds']['median']/1000:.3f}秒 | {b['wallMilliseconds']['median']/1000:.3f}秒 | 完全一致 |")

micro = read(out / "micro-release/results.json")
assert micro["complete"]
mr = []
for case in micro["cases"]:
    assert case["packing"]["allRawBytesRecovered"] and case["packing"]["allPackedBytesMatchIndependentCpu"]
    a = next(v for v in case["variants"] if v["name"] == "original-pair")
    b = next(v for v in case["variants"] if v["name"] == "packed82-pair")
    assert b["validation"]["bitwiseEqualToOriginal"] and b["postTimingValidation"]["bitwiseEqualToOriginal"]
    mr.append(f"| {case['name']} | {case['inputWidth']}×{case['outputWidth']} | {a['wallMilliseconds']['median']:.6f} | {b['wallMilliseconds']['median']:.6f} | {b['versusProductionPairWallSpeedup']:.3f}× |")

trx = ET.parse(out / "tests/packed-iq2-release.trx")
counters = next(e for e in trx.iter() if e.tag.endswith("}Counters"))
assert counters.attrib["failed"] == "0" and counters.attrib["passed"] == counters.attrib["total"]
assert counters.attrib.get("notExecuted", "0") == "0"

changed = subprocess.check_output(["git", "diff", "--cached", "--name-only", "-z"], cwd=root).decode().split("\0")
sources = [p for p in changed if p and not p.startswith("benchmark-results/")]
provenance = {
    "sourceBaseCommit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip(),
    "sourceScope": "Only these staged source changes were exported; unrelated working tree changes excluded.",
    "sourceFiles": {p: {"gitBlob": subprocess.check_output(["git", "rev-parse", ":"+p], cwd=root).decode().strip(),
        "exportedSha256": sha(out / "staged-build" / p)} for p in sources},
    "measuredReleaseBinaries": summaries["greedy"]["baseline"]["raw"]["BinarySha256"],
    "prefillBinaries": prefill["measuredAssemblySha256"],
    "tests": counters.attrib,
}
(out / "source-provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")

raw = summaries["greedy"]["baseline"]["raw"]
report = f'''# IQ2_S 圧縮配置・展開・格納の高速化

## 結果

対象: Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf、Intel Arc B580 12GB × 2。
前回採用したprefill・分割出力ヘッドを両条件で有効にし、今回の差は `InferencePackedIq2Decode` だけ。
GUIの標準推論で新経路を有効化した。Core APIでは明示オプションとして提供する。

| 条件 | 従来 tok/s | 新経路 tok/s | 改善 | 従来 平均±母SD | 新経路 平均±母SD |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

各条件は16 tokenの同じpromptから128 tokenを4回生成し、先頭1回を除いた3回で集計。
decode速度は最初から最後の生成tokenまでの127区間で計算し、モデル読込・prefillを除外。
サンプリングはtemperature 0.6 / top-p 0.95 / top-k 20 / seed 5429。
LoRA条件は既存の `lora_rintya_qa142_Qwen3.8-27B-Uncensored-noMTP-IQ2_M_2epoch.gguf` を使用。
全4回の生成IDと、各条件16 snapshot × 全248,320語彙のFP32 logitsが従来と全バイト一致。
全128ステップのlogitsを保存したという意味ではない。GPUプロファイルを無効にして計測した。

## 圧縮・パッキング・展開・格納

- **82 bytes / 256 weightsを維持**。再量子化も重み精度の変更も行わない。
- grid下位indexとsignをushortへ、上位indexとscaleを共有byteへ再配置する。
- モデル読込時にCPUで一度だけ変換し、GPUへ1コピーだけ転送・常駐させる。
- decodeは整列したushortと共有byteから展開し、FP32 FMA/reduction順を維持する。
- rows2/4/8/16、BSLM/K32rの長文prefill、fused LoRAも新配置から直接計算する。
- 学習、embedding、出力ヘッド、他の量子化形式、未対応デバイス、非既定の非BSLM XMXは従来経路。
- 生成ごとのデバイス/形式判定は読込時に確定し、hot pathから除いた。

モデル読込直後のlive GPU allocationは新旧ともGPU0 `{raw['AfterLoad']['LiveDeviceBytes'][0]:,}` bytes、
GPU1 `{raw['AfterLoad']['LiveDeviceBytes'][1]:,}` bytes。今回の配置変更による追加常駐VRAMは0 bytes。
CPU変換はロード時だけのコスト。ロード時間はJSONに分離記録し、キャッシュ・コンパイル条件の影響があるためdecode倍率に含めない。

## 行列単体

| 実モデルの重み | K×N | 従来 ms | 新経路 ms | 倍率 |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(mr)}

1 token投影、3 warmup、9 sample、各sample16 dispatch。順序を交互に反転。
全出力・末尾guard・独立CPU基準点を検証。元GGUF全バイトの復元とGPU/CPU packing一致も検証。
別候補の128KiB符号lookup tableは実測で遅く、production経路には採用していない。

## 長文入力

| prefix token数 | 従来 中央値 | 新経路 中央値 | 全語彙logits |
| --- | ---: | ---: | --- |
{chr(10).join(pr)}

各1 warmup + 3 sample。reset、prefix処理、次の1 tokenの全logits取得を含む。
8,192 tokenでは約0.12%遅く、長文prefillの高速化は今回確認できなかった。
長文データはGPUカーネルが同一の導入検証ビルドで採取し、最終版ではCPU側のdecode経路選択を読込時に移した。
計測バイナリSHAは各JSONとprovenanceに記録。512 tokenの別profileでpacked K32rの実行も確認した。

## 検証と証拠

- 公開対象だけをGit indexから独立ディレクトリへ取り出し、GUI・benchmarkをReleaseビルド。
- 最終GPU/CPUテスト: **{counters.attrib['passed']} passed / 0 failed / 0 skipped**。
- 全65,536種類のhalf bit表現、packing往復、odd tail、WG32/64/128、native-half設定、LoRA、prefill範囲外fallbackを検証。
- 小型実モデルでrouting、学習のloss/gradient一致、非BSLM fallbackのVRAM peak一致を検証。
- 元GGUF SHA-256: `{raw['Model']['Sha256']}`。

[通常生成](greedy-report/report.decode.ja.md) / [サンプリング](sampling-report/report.decode.ja.md) / [LoRA](lora-report/report.decode.ja.md)

[行列測定](micro-release/results.json) / [長文測定](prefill-long/model-results.json) / [経路profile](prefill-profile/model-results.json)

[テスト結果](tests/packed-iq2-release.trx) / [ソース・バイナリ対応](source-provenance.json) / [再現手順](../../tools/iq2s-packed-decode-bench/README.md)

今回の倍率は上記モデル・GPU・ドライバ・入力・設定での実測。学習全体の速度や他デバイスの倍率には外挿しない。
GGUF、LoRA重み、実行バイナリ、logitsバイナリはコミット対象に含めず、ソースと検証メタデータ・統計を保存する。
'''
(out / "report.ja.md").write_text(report, encoding="utf-8")
print(out / "report.ja.md")
