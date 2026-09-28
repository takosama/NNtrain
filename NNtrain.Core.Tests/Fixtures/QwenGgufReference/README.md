# Independent GGUF reference

These fixtures adapt the user's `gen (1).py`, `compare (1).py`, and `cl_check (2).py`
review scripts. Their SHA-256 hashes are recorded in `expected.json`.

`reference.gguf` is written and read by the official llama.cpp
[gguf-py package](https://github.com/ggml-org/llama.cpp/tree/master/gguf-py)
(PyPI `gguf==0.19.0`). All expected decoded values come from its
`quants.dequantize`, independently of NNtrain. Linear expected values are computed
from those weights using explicit NumPy float64 products and sums.

The 10 KiB file contains:

- Two 7 × 512 Q4_K matrices and two Q6_K matrices: random packed scale/quant bits,
  mixed normal/subnormal scales, and matrices whose scales are all subnormal.
- F32 and F16 edge values, including signed zero, smallest/largest subnormals,
  minimum normals, and finite extremes.
- No pretrained weights or full model; the GGUF is intentionally a tensor fixture.

The tests check GGUF shape/type/payload orientation, bit-exact CPU decoding, and
linear results on Arc for the original, cooperative, and subgroup-enabled paths
at 1, 3, and 5 input rows. Subgroup dispatch can fall back on unsupported hardware.
The numerical bound is `max(abs(reference)) * 5e-6 + 1e-9`, allowing FP32 reduction
order differences. Subnormal-only cases have zero bias so lost subnormal weights
cannot be hidden by a larger bias. These tests do not claim full-model logit or
tokenizer equivalence to a Hugging Face checkpoint.

## Regenerate

From the repository root with Python 3 and pinned packages:

```text
python -m pip install gguf==0.19.0 numpy==2.3.5
python NNtrain.Core.Tests/Fixtures/QwenGgufReference/generate.py
```

The generator is deterministic (seed 20260925); it updates `reference.gguf` and
`expected.json` beside itself. Expected bytes are little-endian float32 or float64
encoded as Base64, to preserve signed zero and exact bit patterns. Package versions
and the GGUF file hash are recorded in the manifest. Python is needed only when
regenerating: routine .NET tests consume the checked-in files.
