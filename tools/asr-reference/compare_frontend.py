import json
from pathlib import Path
import numpy as np
base = Path('benchmark-results/asr-20261006')
official = np.fromfile(base/'official-nemotron-fp32-0.features.f32', dtype='<f4').reshape(128, -1)
native = np.fromfile(base/'csharp-nemotron.features.f32', dtype='<f4').reshape(128, -1)
delta = np.abs(official[:, :native.shape[1]] - native)
result = dict(officialShape=list(official.shape), csharpShape=list(native.shape),
              comparison='valid floor(N/hop) frames; official extra centered terminal frame excluded',
              maxAbsoluteError=float(delta.max()), meanAbsoluteError=float(delta.mean()),
              percentile99AbsoluteError=float(np.quantile(delta, .99)))
(base/'official-nemotron-frontend-comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result))
