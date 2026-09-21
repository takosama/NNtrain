namespace NNtrain;

/// <summary>
/// Monotonic run-local checkpoint cadence, checked only at safe commit
/// boundaries. Saves are acknowledged after completion, so missed intervals
/// never cause back-to-back saves. Resume starts a fresh interval.
/// </summary>
internal sealed class CheckpointSchedule
{
    internal const double DefaultIntervalMinutes = 30;
    private readonly TimeProvider _clock;
    private readonly TimeSpan _interval;
    private long _lastSaved;

    internal CheckpointSchedule(double intervalMinutes = DefaultIntervalMinutes,
        TimeProvider? clock = null)
    {
        _interval = ParseInterval(intervalMinutes);
        _clock = clock ?? TimeProvider.System;
        _lastSaved = _clock.GetTimestamp();
    }

    internal bool IsDue => _clock.GetElapsedTime(_lastSaved) >= _interval;

    internal void RecordSaved() => _lastSaved = _clock.GetTimestamp();

    internal static TimeSpan ParseInterval(double minutes)
    {
        if (!double.IsFinite(minutes) || minutes <= 0
            || minutes >= TimeSpan.MaxValue.TotalMinutes)
            throw new ArgumentOutOfRangeException(nameof(minutes),
                "Checkpoint interval must be a finite positive number of minutes.");
        TimeSpan interval = TimeSpan.FromMinutes(minutes);
        if (interval <= TimeSpan.Zero)
            throw new ArgumentOutOfRangeException(nameof(minutes),
                "Checkpoint interval must be at least one clock tick.");
        return interval;
    }
}
