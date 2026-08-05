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

Phase 6-4 moves epoch and step processing from `Program` into `Trainer`.
`Trainer` depends on three contracts: `IClassificationModel`,
`IImageClassificationDataset`, and `IOptimizer`.

For every step, Trainer selects a sample, builds the input and one-hot target,
clears gradients, performs forward and loss calculation, runs backward, updates
through the optimizer, and records loss and correctness. For every epoch, it
returns a `TrainingEpochResult` containing the one-based epoch number, counts,
and separate training and evaluation `TrainingMetrics` values. Each metrics
value groups average loss, accuracy ratio, and elapsed time.

`TrainerOptions` owns epoch count, steps per epoch, and the sampling random seed.
An optional epoch-completed callback supports progress reporting without moving
the loop back into the application entry point.

## JSON training configuration

Phase 6-6 moves run settings out of `Program`. The `--config` CLI option loads a
JSON object containing:

- `trainingData` and `evaluationData`, each containing `imagePath` and
  `labelPath`;
- `epochs`, `batchSize`, `learningRate`, `weightDecay`, `labelSmoothing`, and
  the shuffling `seed`; weight decay defaults to `0.05`; label smoothing
  defaults to `0.1` and must be in `[0, 1)`;
- model `heads`, `hiddenSize`, `layers`, initialization `seed`, and
  `initializationScale`.

Relative dataset paths are resolved from the JSON file location rather than the
process working directory. Unknown JSON properties are rejected to catch
misspelled settings. Counts and positive floating-point settings are validated
before dataset or model construction.

Each CLI epoch shuffles and consumes every training sample once. Forward passes
within a mini-batch use `Parallel.For` with `Environment.ProcessorCount`
workers. Backward passes then accumulate in deterministic batch order because
parameter gradient buffers are shared. Evaluation inference is fully parallel.

See `training.example.json` at the repository root for a complete configuration.

## Training and evaluation separation

Phase 6-7 gives Trainer distinct training and evaluation datasets. Training
samples are selected using the configured seed and are the only samples that
run backward or optimizer updates. After each training epoch, every evaluation
sample is processed sequentially inside `AutogradContext.NoGrad()`. The epoch
result reports both metric sets independently.

`TrainingComposition` validates all four data files before constructing MNIST
readers and produces `TrainingComponents` containing Trainer, model, and AdamW.
Missing files report whether the training/evaluation image/label file is absent,
the resolved path, and where to correct it. `Program` only handles CLI/config
loading, composition, `Trainer.Run()`, reporting, and user-facing errors.

Checkpoint persistence remains outside Trainer. A later checkpoint component
can use the exposed model and AdamW from `TrainingComponents` together with the
epoch-completed callback, without modifying the epoch or step implementation.
