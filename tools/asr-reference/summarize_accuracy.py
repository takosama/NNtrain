import json, unicodedata
from pathlib import Path
base = Path('benchmark-results/asr-20261006')
def read(name):
    return json.loads((base/name).read_text(encoding='utf-8-sig'))
def normalize(text):
    return ''.join(c for c in text if unicodedata.category(c)[0] in ('L','N'))
def distance(a,b):
    row = list(range(len(b)+1))
    for i,x in enumerate(a,1):
        next_row=[i]
        for j,y in enumerate(b,1):
            next_row.append(min(next_row[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=next_row
    return row[-1]
rows=[]
for family in ('nemotron','parakeet'):
    for index in range(3):
        reference_path = base/(f'fleurs-ja-validation-{index}-reference.txt' if index else 'fleurs-ja-validation-0.reference.txt')
        reference = reference_path.read_text(encoding='utf-8-sig')
        filename = f'parakeet-real-arc-japanese-final-{index}.json' if family=='parakeet' else (
            f'nemotron-real-arc-japanese-{index}.json' if index else 'real-arc-japanese-final.json')
        native=read(filename)
        official=read(f'official-{family}-rounded-fp16-{index}.json')
        original=read(f'official-{family}-fp32-{index}.json')
        text=official['text'][0] if isinstance(official['text'],list) else official['text']
        original_text=original['text'][0] if isinstance(original['text'],list) else original['text']
        target=normalize(reference)
        rows.append(dict(model=family,index=index,audioSeconds=native['audioSeconds'],
            csharpArcText=native['transcript'],officialRoundedFp16Text=text,reference=reference,
            officialOriginalFp32Text=original_text,
            normalizedTranscriptsMatch=normalize(native['transcript'])==normalize(text),
            csharpArcCer=distance(target,normalize(native['transcript']))/len(target),
            officialCer=distance(target,normalize(text))/len(target),
            officialOriginalFp32Cer=distance(target,normalize(original_text))/len(target),
            fp16ChangesNormalizedText=normalize(text)!=normalize(original_text),
            csharpArcSeconds=native['recognitionSeconds'],officialCpuSeconds=official['recognitionSeconds'],
            csharpDedicatedPeak=native['osDedicatedGpuPeakBytes']))
result=dict(recordings='three distinct public Google FLEURS ja_jp validation recordings, indices 0/1/2',
    normalization='Unicode letters and numbers retained, punctuation/spacing ignored; edit distance divided by reference count',
    scope='Small sample validation, not a general accuracy benchmark; official CPU uses four threads',rows=rows)
(base/'official-asr-accuracy-summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
for row in rows:
    print(json.dumps({k:v for k,v in row.items() if k not in ('csharpArcText','officialRoundedFp16Text','officialOriginalFp32Text','reference')}))
