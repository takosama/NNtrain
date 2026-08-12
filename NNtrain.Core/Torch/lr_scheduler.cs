#pragma warning disable CS8981

namespace NNtrain;

public interface ILRScheduler
{
    int LastEpoch { get; }

    IReadOnlyList<float> GetLastLearningRates();

    IReadOnlyList<float> Step();

    IReadOnlyList<float> step() => Step();

    IReadOnlyList<float> get_last_lr() => GetLastLearningRates();

    LRSchedulerStateDictionary state_dict();

    void load_state_dict(LRSchedulerStateDictionary state);
}

public sealed record LRSchedulerStateDictionary(
    string SchedulerType,
    int LastEpoch,
    double? LastProgress = null);

/// <summary>PyTorch-style learning-rate scheduler factories.</summary>
public static class lr_scheduler
{
    public static ILRScheduler CosineAnnealingLR(
        IOptimizer optimizer,
        int T_max,
        float eta_min = 0f)
        => new CosineAnnealingLRScheduler(optimizer, T_max, eta_min);

    public static ILRScheduler LinearWarmupCosineAnnealingLR(
        IOptimizer optimizer,
        int total_epochs,
        int warmup_epochs = 0,
        float min_lr_ratio = 0.01f)
        => new LinearWarmupCosineLRScheduler(
            optimizer,
            total_epochs,
            warmup_epochs,
            min_lr_ratio);

    public static WarmupCosineProgressLRScheduler WarmupCosineProgressLR(
        IOptimizer optimizer,
        float warmup_percent = 0f)
        => new(optimizer, warmup_percent);
}

public sealed class WarmupCosineProgressLRScheduler
{
    private readonly SchedulerOptimizerGroups _groups;
    private readonly float _warmupPercent;
    private double _lastProgress;

    public WarmupCosineProgressLRScheduler(
        IOptimizer optimizer,
        float warmupPercent = 0f)
    {
        if (!float.IsFinite(warmupPercent)
            || warmupPercent < 0f
            || warmupPercent >= 100f)
        {
            throw new ArgumentOutOfRangeException(nameof(warmupPercent));
        }
        _groups = new SchedulerOptimizerGroups(optimizer);
        _warmupPercent = warmupPercent;
    }

    public IReadOnlyList<float> step(double progress)
    {
        float factor = CalculateFactor(progress, _warmupPercent);
        IReadOnlyList<float> rates =
            _groups.Set(baseRate => baseRate * factor);
        _lastProgress = progress;
        return rates;
    }

    public IReadOnlyList<float> get_last_lr() => _groups.CurrentRates;

    public LRSchedulerStateDictionary state_dict()
        => new(
            nameof(WarmupCosineProgressLRScheduler),
            LastEpoch: 0,
            _lastProgress);

    public void load_state_dict(LRSchedulerStateDictionary state)
    {
        ArgumentNullException.ThrowIfNull(state);
        if (!string.Equals(
                state.SchedulerType,
                nameof(WarmupCosineProgressLRScheduler),
                StringComparison.Ordinal)
            || state.LastProgress is not double progress
            || !double.IsFinite(progress)
            || progress < 0d
            || progress > 1d)
        {
            throw new ArgumentException(
                "WarmupCosineProgressLR state is incompatible.",
                nameof(state));
        }
        _lastProgress = progress;
        _groups.RefreshCurrentRates();
    }

    public static float CalculateFactor(
        double overallProgress,
        float warmupPercent)
    {
        const float MinimumFactor = 1e-6f;
        if (!double.IsFinite(overallProgress)
            || overallProgress < 0d
            || overallProgress > 1d)
        {
            throw new ArgumentOutOfRangeException(nameof(overallProgress));
        }
        if (!float.IsFinite(warmupPercent)
            || warmupPercent < 0f
            || warmupPercent >= 100f)
        {
            throw new ArgumentOutOfRangeException(nameof(warmupPercent));
        }

        double warmupFraction = warmupPercent / 100d;
        if (warmupFraction > 0d && overallProgress <= warmupFraction)
        {
            return MathF.Max(
                MinimumFactor,
                (float)(overallProgress / warmupFraction));
        }

        double decayProgress = warmupFraction == 1d
            ? 1d
            : (overallProgress - warmupFraction)
                / (1d - warmupFraction);
        decayProgress = Math.Clamp(decayProgress, 0d, 1d);
        float cosine = 0.5f
            * (1f + MathF.Cos(MathF.PI * (float)decayProgress));
        return MathF.Max(MinimumFactor, cosine);
    }
}

public sealed class CosineAnnealingLRScheduler : ILRScheduler
{
    private readonly SchedulerOptimizerGroups _groups;
    private readonly int _maximumEpochs;
    private readonly float _minimumLearningRate;

    public CosineAnnealingLRScheduler(
        IOptimizer optimizer,
        int maximumEpochs,
        float minimumLearningRate = 0f)
    {
        if (maximumEpochs <= 0)
            throw new ArgumentOutOfRangeException(nameof(maximumEpochs));
        if (!float.IsFinite(minimumLearningRate)
            || minimumLearningRate < 0f)
        {
            throw new ArgumentOutOfRangeException(nameof(minimumLearningRate));
        }

        _groups = new SchedulerOptimizerGroups(optimizer);
        _maximumEpochs = maximumEpochs;
        _minimumLearningRate = minimumLearningRate;
    }

    public int LastEpoch { get; private set; }

    public IReadOnlyList<float> Step()
    {
        if (LastEpoch >= _maximumEpochs)
            throw new InvalidOperationException("Scheduler has reached T_max.");
        LastEpoch++;
        float progress = (float)LastEpoch / _maximumEpochs;
        float cosine = 0.5f * (1f + MathF.Cos(MathF.PI * progress));
        return _groups.Set((baseRate) =>
            _minimumLearningRate
            + (baseRate - _minimumLearningRate) * cosine);
    }

    public IReadOnlyList<float> GetLastLearningRates()
        => _groups.CurrentRates;

    public LRSchedulerStateDictionary state_dict()
        => new(nameof(CosineAnnealingLRScheduler), LastEpoch);

    public void load_state_dict(LRSchedulerStateDictionary state)
    {
        ArgumentNullException.ThrowIfNull(state);
        if (!string.Equals(
                state.SchedulerType,
                nameof(CosineAnnealingLRScheduler),
                StringComparison.Ordinal)
            || state.LastEpoch < 0
            || state.LastEpoch > _maximumEpochs)
        {
            throw new ArgumentException(
                "CosineAnnealingLR state is incompatible.",
                nameof(state));
        }
        LastEpoch = state.LastEpoch;
        _groups.RefreshCurrentRates();
    }
}

public sealed class LinearWarmupCosineLRScheduler : ILRScheduler
{
    private readonly SchedulerOptimizerGroups _groups;
    private readonly int _totalEpochs;
    private readonly int _warmupEpochs;
    private readonly float _minimumRatio;

    public LinearWarmupCosineLRScheduler(
        IOptimizer optimizer,
        int totalEpochs,
        int warmupEpochs = 0,
        float minimumLearningRateRatio = 0.01f)
    {
        if (totalEpochs <= 0)
            throw new ArgumentOutOfRangeException(nameof(totalEpochs));
        if (warmupEpochs < 0 || warmupEpochs >= totalEpochs)
            throw new ArgumentOutOfRangeException(nameof(warmupEpochs));
        if (!float.IsFinite(minimumLearningRateRatio)
            || minimumLearningRateRatio <= 0f
            || minimumLearningRateRatio > 1f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(minimumLearningRateRatio));
        }

        _groups = new SchedulerOptimizerGroups(optimizer);
        _totalEpochs = totalEpochs;
        _warmupEpochs = warmupEpochs;
        _minimumRatio = minimumLearningRateRatio;
    }

    public int LastEpoch { get; private set; }

    public IReadOnlyList<float> Step()
    {
        if (LastEpoch >= _totalEpochs)
            throw new InvalidOperationException(
                "Scheduler has reached the configured epoch count.");
        LastEpoch++;
        float factor = CalculateFactor(
            LastEpoch,
            _totalEpochs,
            _warmupEpochs,
            _minimumRatio);
        return _groups.Set(baseRate => baseRate * factor);
    }

    public IReadOnlyList<float> GetLastLearningRates()
        => _groups.CurrentRates;

    public LRSchedulerStateDictionary state_dict()
        => new(nameof(LinearWarmupCosineLRScheduler), LastEpoch);

    public void load_state_dict(LRSchedulerStateDictionary state)
    {
        ArgumentNullException.ThrowIfNull(state);
        if (!string.Equals(
                state.SchedulerType,
                nameof(LinearWarmupCosineLRScheduler),
                StringComparison.Ordinal)
            || state.LastEpoch < 0
            || state.LastEpoch > _totalEpochs)
        {
            throw new ArgumentException(
                "LinearWarmupCosineAnnealingLR state is incompatible.",
                nameof(state));
        }
        LastEpoch = state.LastEpoch;
        _groups.RefreshCurrentRates();
    }

    public static float CalculateFactor(
        int epoch,
        int totalEpochs,
        int warmupEpochs,
        float minimumRatio)
    {
        if (epoch <= 0 || epoch > totalEpochs)
            throw new ArgumentOutOfRangeException(nameof(epoch));
        if (totalEpochs <= 0)
            throw new ArgumentOutOfRangeException(nameof(totalEpochs));
        if (warmupEpochs < 0 || warmupEpochs >= totalEpochs)
            throw new ArgumentOutOfRangeException(nameof(warmupEpochs));
        if (!float.IsFinite(minimumRatio)
            || minimumRatio <= 0f
            || minimumRatio > 1f)
        {
            throw new ArgumentOutOfRangeException(nameof(minimumRatio));
        }

        if (warmupEpochs > 0 && epoch <= warmupEpochs)
            return (float)epoch / warmupEpochs;

        int decayEpochs = totalEpochs - warmupEpochs;
        float progress = (float)(epoch - warmupEpochs) / decayEpochs;
        float cosine = 0.5f * (1f + MathF.Cos(MathF.PI * progress));
        return minimumRatio + (1f - minimumRatio) * cosine;
    }
}

internal sealed class SchedulerOptimizerGroups
{
    private readonly ILearningRateAdjustable[] _groups;
    private readonly float[] _baseRates;
    private float[] _currentRates;

    internal SchedulerOptimizerGroups(IOptimizer optimizer)
    {
        ArgumentNullException.ThrowIfNull(optimizer);
        _groups = Flatten(optimizer).ToArray();
        if (_groups.Length == 0)
        {
            throw new ArgumentException(
                $"Optimizer '{optimizer.GetType().Name}' does not expose " +
                "adjustable learning-rate groups.",
                nameof(optimizer));
        }
        _baseRates = _groups.Select(group => group.LearningRate).ToArray();
        _currentRates = _baseRates.ToArray();
    }

    internal IReadOnlyList<float> CurrentRates
        => Array.AsReadOnly(_currentRates);

    internal IReadOnlyList<float> Set(Func<float, float> selector)
    {
        var rates = new float[_groups.Length];
        for (int index = 0; index < _groups.Length; index++)
        {
            float rate = selector(_baseRates[index]);
            if (!float.IsFinite(rate) || rate <= 0f)
            {
                throw new InvalidOperationException(
                    "Scheduler produced a non-positive learning rate.");
            }
            _groups[index].SetLearningRate(rate);
            rates[index] = rate;
        }
        _currentRates = rates;
        return Array.AsReadOnly(rates);
    }

    internal void RefreshCurrentRates()
        => _currentRates = _groups
            .Select(group => group.LearningRate)
            .ToArray();

    private static IEnumerable<ILearningRateAdjustable> Flatten(
        IOptimizer optimizer)
    {
        if (optimizer is CompositeOptimizer composite)
        {
            foreach (IOptimizer child in composite.Optimizers)
            {
                foreach (ILearningRateAdjustable group in Flatten(child))
                    yield return group;
            }
            yield break;
        }

        if (optimizer is ILearningRateAdjustable adjustable)
            yield return adjustable;
    }
}
