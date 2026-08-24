# Data design

## MNIST file locations

Phase 6-1 removes file-location policy from `Mnist`. Its constructor requires
the image IDX path and label IDX path explicitly; the class only reads and
decodes those files and does not search the working directory or provide hidden
fallback paths.

The console entry point receives both paths through a JSON training
configuration selected on the command line:

```text
dotnet run --project NNtrain.Cli -- --config <training-config.json>
```

Paths may be absolute or relative to the configuration file's directory.
Deployment and test fixtures can therefore select a dataset location without
changing the data reader.

## MNIST IDX validation

Phase 6-2 parses IDX header integers as big-endian values before exposing the
dataset. Construction rejects files unless all of these conditions hold:

- the image and label headers are at least 16 and 8 bytes respectively;
- image magic number is `2051` and label magic number is `2049`;
- both declared sample counts are positive and equal;
- the image dimensions are exactly 28 rows by 28 columns;
- image and label file lengths exactly match their declared sample counts.

The validated sample count is exposed as `Mnist.Count`. The training entry point
uses the validated count and shape properties instead of duplicating dataset
dimensions and the training-set size.

## Image-classification dataset contract

Phase 6-3 introduces `IImageClassificationDataset` as the boundary between the
training workflow and a concrete data source. It exposes sample count, image
shape, image size, class count, and one atomic sample operation:

```text
int ReadSample(int index, Span<float> destination)
```

`ReadSample` writes the normalized image and returns the label for the same
index. It rejects an out-of-range index, a destination shorter than `ImageSize`,
and labels outside `0..ClassCount-1`. The earlier split `GetDataFloat` and
`GetLabel` methods are removed so callers cannot accidentally combine data from
different samples.

The training entry point holds MNIST through this interface and obtains model
input shape, output class count, sampling range, and buffer sizes from the
dataset contract rather than from MNIST-specific constants.

## Training orchestration

The CLI uses one internal `TrainingRunner` lifecycle for classification,
finite language-model training, and streaming language-model training. It owns
epoch/resume iteration, deterministic shuffling, progress-unit rounding, and
fractional checkpoint boundaries. Task implementations own dataset iteration,
forward/backward, evaluation, and task-specific checkpoint payloads.

Library consumers use the explicit PyTorch-style loop shown in the README;
there is no second public Trainer abstraction with different batching or
checkpoint semantics.

## JSON training configuration

Schema version 2 uses an explicit `task.type` and common `data`, `model`,
`training`, `runtime`, `optimization`, `checkpoint`, and `reporting` sections.
Unknown sections and properties are rejected. Checked-in configurations use
this layout.

Phase 6-6 moves run settings out of `Program`. The `--config` CLI option loads a
JSON object containing:

- `trainingData` and `evaluationData`, each containing `imagePath` and
  `labelPath` for MNIST, or `dataPath` for CIFAR-100; CIFAR-100 also accepts
  `patchSize` (default `4`), `normalize`, and an `augmentation` object
  containing `randomCropPadding`, `horizontalFlip`, and `verticalFlip`;
- `epochs`, `microBatchSize`, `microBatchCount`, `optimizer`, `learningRate`,
  `auxiliaryLearningRate`, `weightDecay`, `labelSmoothing`, `warmupEpochs`,
  `minimumLearningRateRatio`, `earlyStoppingPatience`,
  `earlyStoppingMinimumDelta`, and the shuffling
  `seed`; GainShareAdamW is the default optimizer with learning rate `3e-4`
  and weight decay `5e-4`; `gainShareBlockDepth`, `gainShareBeta1`,
  `gainShareBeta2`, `gainShareEpsilon`, `gainShareRho`, `gainShareGamma`,
  `gainShareMinScale`, and `gainShareMaxScale` configure its block updates;
  label smoothing defaults to `0.1` and must be in `[0, 1)`; supported
  optimizer names are `gainshareadamw`, `nekomuon`, `lion`, and `adamw`;
  `microBatchSize * microBatchCount` is the effective training batch size;
  the legacy `batchSize` setting remains a fallback for `microBatchSize`;
- `showLossGraph`, which defaults to `true`; the CLI writes an automatically
  refreshing HTML plot next to the configuration file and adds connected train
  and evaluation loss points after every epoch;
- model `heads`, `hiddenSize`, `layers`, initialization `seed`,
  `initializationScale`, and residual `dropout` probability.

Relative dataset paths are resolved from the JSON file location rather than the
process working directory. Unknown JSON properties are rejected to catch
misspelled settings. Counts and positive floating-point settings are validated
before dataset or model construction.

The checked-in CIFAR-100 configuration normalizes both training and evaluation
pixels with the training-set channel means `(0.50707516, 0.48654887,
0.44091784)` and standard deviations `(0.26733429, 0.25643846, 0.27615047)`.
The reader directly lays each image out as 64 row-major 4x4 patch tokens, each
containing 48 channel-first values, without allocating a second image buffer.
Training uses four-pixel zero-padded random crops and horizontal flips. Vertical
flips are disabled. Evaluation performs normalization without augmentation.

Each CLI epoch shuffles and consumes every training sample once. Forward passes
within a mini-batch use `Parallel.For` with `Environment.ProcessorCount`
workers. Backward passes then accumulate in deterministic batch order because
parameter gradient buffers are shared. Evaluation inference is fully parallel.

See `training.example.json` at the repository root for a complete configuration.

## Training and evaluation separation

Classification uses distinct training and evaluation datasets. Training
samples are selected using the configured seed and are the only samples that
run backward or optimizer updates. After each training epoch, every evaluation
sample is processed sequentially inside `AutogradContext.NoGrad()`. The epoch
result reports both metric sets independently.

CLI composition validates all four data files before constructing readers,
the model, optimizer, and scheduler.
Missing files report whether the training/evaluation image/label file is absent,
the resolved path, and where to correct it. `Program` only handles CLI/config
loading, composition, task execution, reporting, and user-facing errors.

The CLI retains the model state with the lowest evaluation loss, restores it
after training or early stopping, and atomically writes it beside the selected
configuration as `*.best-model.json`. Optimizer moments are not included in
this inference-oriented best-model checkpoint.
