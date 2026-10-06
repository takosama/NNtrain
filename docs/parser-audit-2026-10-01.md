# Local GGUF / SafeTensors parser fixes

Current dirty checkout based on c1878e5; no commit, push, upload, GPU work,
learning-process restart or checkpoint modification. Existing CPU/GPU/default-on
work was retained. Protected CPU source hashes match the earlier snapshot.

## GGUF

`NNtrain.Core/Serialization/GgufReader.cs` checks table counts, remaining encoded
bytes, array element counts, string byte lengths, nesting depth and a shared
decoded-allocation budget before allocating the corresponding variable-size
objects. Arithmetic is checked; unknown empty-array element types are rejected.
Nested arrays remain supported within the configured depth.

New public `GgufReadLimits` can be supplied to `new GgufReader(path, limits)`.
Defaults: array 4,194,304 elements; string 16 MiB encoded bytes; metadata 100,000
entries; tensors 1,000,000; nesting depth 32; allocation-accounting budget 512 MiB.
Depth may be configured from 0 through 128. Accounting includes immediate array
slots/values, table entries, strings and temporary UTF8 bytes; it is a conservative
parser policy, not a measured GC heap cap. Tensor payloads are outside this
metadata budget. These limits may reject unusually large legitimate metadata;
callers can supply explicit limits. No model tensor mapping behavior was changed.

## SafeTensors

`NNtrain.Core/Torch/safetensors.cs` uses the descriptor validator in
`SafeTensorModelStream.cs` before allocating any float payload arrays. All
non-empty ranges are sorted and checked for overlap, including identical,
partial and contained ranges in either header order. Adjacent and zero-length
ranges remain valid. Zero dimensions denote empty tensors; element-count
overflow becomes InvalidDataException. Indexed model loading reuses the decoded
entries rather than decoding the file twice.

`load_keys` reads prefix/header only and validates descriptors without decoding
payloads. Internal `SafeTensorReadLimits` is injectable into Stream load/key
tests and streamed model restore. Defaults: header 64 MiB, tensors 100,000,
rank 64, total logical float32-expanded payload 16 GiB. This is a total logical
size limit shared with streaming, not its peak staging allocation. The public
facade uses the defaults; no public override was added. Large legitimate models
above that policy require a deliberate internal policy change.

Streamed restore validates mapping and every mapped payload's finite/target
Float16-range constraints before the first model write. This requires a second
payload read with bounded staging (at most the existing 16 MiB block). It avoids
partial model updates from later invalid descriptors or later invalid numbers;
it is not transaction rollback for device failures, I/O failures during the
write pass, or mutations outside the normal FileShare.Read protection.

## Validation

57 focused CPU tests passed, zero failed/skipped, including existing GGUF startup,
eager SafeTensors codecs and streamed model tests. New fixtures are small: no
OOM-sized payload or GPU fixture was executed. New tests cover nested/empty GGUF,
count/string limits and truncation, unsigned maximum counts, cumulative budgets,
SafeTensors overlap/containment/header-order reversal, empty/adjacent ranges,
F32/F16/BF16, overflow, malformed later descriptors, tiny rank/header/expansion
limits, no payload reads before invalid-header rejection, and unchanged model
values after a later NaN.

Command:

```powershell
dotnet test NNtrain.Core.Tests/NNtrain.Core.Tests.csproj -c Release --no-restore --filter 'FullyQualifiedName~GgufReader|FullyQualifiedName~GgufStartupRead|FullyQualifiedName~SafeTensor' --logger 'trx;LogFileName=parser-audit.trx' --results-directory benchmark-results/parser-audit-20261001
dotnet build NNtrain.Cli/NNtrain.Cli.csproj -c Release --no-restore -v minimal
```

Evidence: `benchmark-results/parser-audit-20261001/parser-audit.trx` and
`cpu-preservation.json`. Changed parser/test files:

- `NNtrain.Core/Serialization/GgufReader.cs`
- `NNtrain.Core/Serialization/GgufReadLimits.cs` (new)
- `NNtrain.Core/Torch/safetensors.cs`
- `NNtrain.Core/Torch/SafeTensorModelStream.cs`
- `NNtrain.Core/Torch/SafeTensorReadLimits.cs` (new)
- `NNtrain.Core.Tests/GgufReaderLimitTests.cs` (new)
- `NNtrain.Core.Tests/SafeTensorDescriptorValidationTests.cs` (new)

This work does not establish GPU performance or long-term LoRA quality and does
not alter the previously requested default-on settings.
