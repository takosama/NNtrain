import json
from pathlib import Path
import numpy as np
base = Path('benchmark-results/asr-20261006')
def compare(a, b):
    error = np.abs(a-b)
    return dict(maxAbsoluteError=float(error.max()), meanAbsoluteError=float(error.mean()),
                relativeL2=float(np.linalg.norm(a-b)/np.linalg.norm(b)))
official = np.fromfile(base/'official-parakeet-prefix.encoder.f32', dtype='<f4').reshape(8,1024)
cpu = np.fromfile(base/'csharp-parakeet-encoder-cpu.f32', dtype='<f4').reshape(8,1024)
features = np.fromfile(base/'official-parakeet-prefix.features.f32', dtype='<f4')
native = np.fromfile(base/'csharp-parakeet-prefix.features.f32', dtype='<f4')
result = dict(frontend=compare(native, features), officialVsCsharpCpu=compare(cpu,official),
              scope='0.57 s input; official NeMo 3.0.0; FP16 parameters, FP32 operations')
arc_path = base/'csharp-parakeet-encoder-arc.f32'
if arc_path.exists():
    arc = np.fromfile(arc_path,dtype='<f4').reshape(8,1024)
    result['officialVsCsharpArc'] = compare(arc, official)
    result['csharpCpuVsArc'] = compare(arc, cpu)
(base/'official-parakeet-encoder-comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
