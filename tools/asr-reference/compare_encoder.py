import json
from pathlib import Path
import numpy as np
base = Path('benchmark-results/asr-20261006')
official = np.fromfile(base/'official-nemotron-encoder.encoder.f32', dtype='<f4').reshape(8, 1024)
cpu = np.fromfile(base/'csharp-nemotron-encoder-cpu.f32', dtype='<f4').reshape(8, 1024)
arc = np.fromfile(base/'csharp-nemotron-encoder-arc.f32', dtype='<f4').reshape(8, 1024)
def compare(a, b):
    error = np.abs(a-b)
    return dict(maxAbsoluteError=float(error.max()), meanAbsoluteError=float(error.mean()),
                relativeL2=float(np.linalg.norm(a-b)/np.linalg.norm(b)),
                cosineSimilarity=float(np.vdot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))))
result = dict(shape=[8,1024], officialVsCsharpCpu=compare(cpu,official),
              officialVsCsharpArc=compare(arc,official), csharpCpuVsArc=compare(arc,cpu),
              scope='Two cached chunks on 0.57 s prefix; FP16 parameters, FP32 operations')
(base/'official-nemotron-encoder-comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result))
