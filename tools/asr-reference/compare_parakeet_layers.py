import json
from pathlib import Path
import numpy as np
base = Path('benchmark-results/asr-20261006')
rows = []
for name in ['pre_encode']+[f'layer-{i}' for i in range(24)]:
    official = np.fromfile(base/'parakeet-trace-official'/(name+'.f32'), dtype='<f4')
    native = np.fromfile(base/'parakeet-trace-arc'/(name+'.f32'), dtype='<f4')
    error = np.abs(official-native)
    rows.append(dict(layer=name, maxError=float(error.max()), meanError=float(error.mean())))
(base/'parakeet-layer-comparison.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
print(json.dumps(rows))
