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
        Action<TrainingEpochResult>? onEpochCompleted = null)
    {
        var results = new List<TrainingEpochResult>(_options.Epochs);

        for (int epoch = 1; epoch <= _options.Epochs; epoch++)
        {
            TrainingMetrics training = TrainEpoch();
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

    private TrainingMetrics TrainEpoch()
    {
        float lossSum = 0f;
        int correct = 0;
        var stopwatch = Stopwatch.StartNew();

        for (int step = 0; step < _options.StepsPerEpoch; step++)
        {
            TrainingStepResult result = TrainStep();
            lossSum += result.Loss;
            if (result.IsCorrect)
                correct++;
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
        Sample sample = ReadSample(_trainingDataset, sampleIndex);

        _optimizer.ZeroGrad();
        ForwardResult forward = Forward(sample);
        float lossValue = forward.Loss.Data[0];

        forward.Loss.Backward();
        _optimizer.Step();

        return new TrainingStepResult(lossValue, forward.IsCorrect);
    }

    private TrainingMetrics Evaluate()
    {
        float lossSum = 0f;
        int correct = 0;
        var stopwatch = Stopwatch.StartNew();

        using (AutogradContext.NoGrad())
        {
            for (int index = 0; index < _evaluationDataset.Count; index++)
            {
                Sample sample = ReadSample(_evaluationDataset, index);
                ForwardResult forward = Forward(sample);
                lossSum += forward.Loss.Data[0];
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
        int index)
    {
        var inputValues = new float[dataset.ImageSize];
        int answer = dataset.ReadSample(index, inputValues);
        var input = new Tensor(
            inputValues,
            [dataset.Rows, dataset.Columns],
            "classifierInput");
        var targetValues = new float[dataset.ClassCount];
        targetValues[answer] = 1f;
        Tensor target = Tensor.From1D(targetValues, "classifierTarget");
        return new Sample(input, target, answer);
    }

    private ForwardResult Forward(Sample sample)
    {
        Tensor logits = _model.Forward(sample.Input);
        Tensor logProbabilities = logits.LogSoftmaxLastDim();
        Tensor loss = -(sample.Target * logProbabilities).Sum();
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
        Tensor Target,
        int Answer);

    private readonly record struct ForwardResult(
        Tensor Loss,
        bool IsCorrect);
}
