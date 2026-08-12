using System.Diagnostics;

namespace NNtrain;

public sealed class Trainer
{
    private readonly IClassificationModel _model;
    private readonly IImageClassificationDataset _trainingDataset;
    private readonly IImageClassificationDataset _evaluationDataset;
    private readonly IOptimizer _optimizer;
    private readonly TrainerOptions _options;
    private readonly Random _random;

    public Trainer(
        IClassificationModel model,
        IImageClassificationDataset trainingDataset,
        IImageClassificationDataset evaluationDataset,
        IOptimizer optimizer,
        TrainerOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(trainingDataset);
        ArgumentNullException.ThrowIfNull(evaluationDataset);
        ArgumentNullException.ThrowIfNull(optimizer);

        _model = model;
        _trainingDataset = trainingDataset;
        _evaluationDataset = evaluationDataset;
        _optimizer = optimizer;
        _options = options ?? new TrainerOptions();
        ValidateContracts();
        _random = new Random(_options.RandomSeed);
    }

    public IReadOnlyList<TrainingEpochResult> Run(
        Action<TrainingEpochResult>? onEpochCompleted = null,
        Action<TrainingBatchResult>? onBatchCompleted = null)
    {
        var results = new List<TrainingEpochResult>(_options.Epochs);

        for (int epoch = 1; epoch <= _options.Epochs; epoch++)
        {
            TrainingMetrics training = TrainEpoch(epoch, onBatchCompleted);
            TrainingMetrics evaluation = Evaluate();
            var result = new TrainingEpochResult(
                epoch,
                _options.StepsPerEpoch,
                _evaluationDataset.Count,
                training,
                evaluation);
            results.Add(result);
            onEpochCompleted?.Invoke(result);
        }

        return results.AsReadOnly();
    }

    public IReadOnlyList<TrainingEpochResult> fit(
        Action<TrainingEpochResult>? on_epoch_completed = null,
        Action<TrainingBatchResult>? on_batch_completed = null)
        => Run(on_epoch_completed, on_batch_completed);

    private TrainingMetrics TrainEpoch(
        int epoch,
        Action<TrainingBatchResult>? onBatchCompleted)
    {
        if (_model is Module module)
            module.train();

        float lossSum = 0f;
        int correct = 0;
        var stopwatch = Stopwatch.StartNew();

        for (int step = 0; step < _options.StepsPerEpoch; step++)
        {
            TrainingStepResult result = TrainStep();
            lossSum += result.Loss;
            if (result.IsCorrect)
                correct++;
            onBatchCompleted?.Invoke(
                new TrainingBatchResult(
                    epoch,
                    step + 1,
                    _options.StepsPerEpoch,
                    result.Loss,
                    result.IsCorrect));
        }

        stopwatch.Stop();
        return new TrainingMetrics(
            lossSum / _options.StepsPerEpoch,
            (float)correct / _options.StepsPerEpoch,
            stopwatch.Elapsed);
    }

    private TrainingStepResult TrainStep()
    {
        int sampleIndex = _random.Next(_trainingDataset.Count);
        Sample sample = ReadSample(
            _trainingDataset,
            sampleIndex,
            useTrainingAugmentation: true);

        _optimizer.zero_grad();
        ForwardResult forward = Forward(
            sample,
            _options.LabelSmoothing);
        float lossValue = forward.Loss.item();

        forward.Loss.backward();
        _optimizer.step();

        return new TrainingStepResult(lossValue, forward.IsCorrect);
    }

    private TrainingMetrics Evaluate()
    {
        if (_model is Module module)
            module.eval();

        float lossSum = 0f;
        int correct = 0;
        var stopwatch = Stopwatch.StartNew();

        using (torch.no_grad())
        {
            for (int index = 0; index < _evaluationDataset.Count; index++)
            {
                Sample sample = ReadSample(
                    _evaluationDataset,
                    index,
                    useTrainingAugmentation: false);
                ForwardResult forward = Forward(sample, labelSmoothing: 0f);
                lossSum += forward.Loss.item();
                if (forward.IsCorrect)
                    correct++;
            }
        }

        stopwatch.Stop();
        return new TrainingMetrics(
            lossSum / _evaluationDataset.Count,
            (float)correct / _evaluationDataset.Count,
            stopwatch.Elapsed);
    }

    private Sample ReadSample(
        IImageClassificationDataset dataset,
        int index,
        bool useTrainingAugmentation)
    {
        var inputValues = new float[dataset.ImageSize];
        int answer = useTrainingAugmentation
            ? dataset.ReadTrainingSample(index, inputValues, _random)
            : dataset.ReadSample(index, inputValues);
        var input = Tensor.FromOwnedData(
            inputValues,
            [dataset.Rows, dataset.Columns],
            "classifierInput");
        return new Sample(input, answer);
    }

    private ForwardResult Forward(Sample sample, float labelSmoothing)
    {
        Tensor logits = _model.forward(sample.Input);
        Tensor loss = nn.functional.cross_entropy(
            logits,
            [sample.Answer],
            label_smoothing: labelSmoothing);
        int prediction = ArgMax(logits.Data);
        return new ForwardResult(loss, prediction == sample.Answer);
    }

    private void ValidateContracts()
    {
        if (_options.Epochs <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(TrainerOptions.Epochs),
                _options.Epochs,
                "Epoch count must be positive.");
        }

        if (_options.StepsPerEpoch <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(TrainerOptions.StepsPerEpoch),
                _options.StepsPerEpoch,
                "Steps per epoch must be positive.");
        }

        if (!float.IsFinite(_options.LabelSmoothing)
            || _options.LabelSmoothing < 0f
            || _options.LabelSmoothing >= 1f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(TrainerOptions.LabelSmoothing),
                _options.LabelSmoothing,
                "Label smoothing must be finite and in the range [0, 1).");
        }

        ValidateDataset(_trainingDataset, "Training");
        ValidateDataset(_evaluationDataset, "Evaluation");
    }

    private void ValidateDataset(
        IImageClassificationDataset dataset,
        string role)
    {
        if (dataset.Count <= 0)
            throw new ArgumentException($"{role} dataset must contain samples.");

        if (dataset.Rows != _model.InputRows
            || dataset.Columns != _model.InputColumns)
        {
            throw new ArgumentException(
                $"{role} dataset shape '{dataset.Rows}x{dataset.Columns}' " +
                $"does not match model input shape '{_model.InputRows}x" +
                $"{_model.InputColumns}'.");
        }

        if (dataset.ClassCount != _model.ClassCount)
        {
            throw new ArgumentException(
                $"{role} dataset class count '{dataset.ClassCount}' does " +
                $"not match model class count '{_model.ClassCount}'.");
        }
    }

    private static int ArgMax(IReadOnlyList<float> values)
    {
        int result = 0;
        float best = values[0];
        for (int index = 1; index < values.Count; index++)
        {
            if (values[index] > best)
            {
                best = values[index];
                result = index;
            }
        }

        return result;
    }

    private readonly record struct TrainingStepResult(
        float Loss,
        bool IsCorrect);

    private readonly record struct Sample(
        Tensor Input,
        int Answer);

    private readonly record struct ForwardResult(
        Tensor Loss,
        bool IsCorrect);
}
