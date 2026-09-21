using NNtrain;
using Xunit;

public sealed class CheckpointScheduleTests
{
    [Fact]
    public void DefaultIsThirtyMinutesAndOnlySuccessfulSaveResetsIt()
    {
        var clock = new ManualClock();
        var schedule = new CheckpointSchedule(clock: clock);
        Assert.False(schedule.IsDue);
        clock.Advance(TimeSpan.FromMinutes(30) - TimeSpan.FromTicks(1));
        Assert.False(schedule.IsDue);
        clock.Advance(TimeSpan.FromTicks(1));
        Assert.True(schedule.IsDue);
        Assert.True(schedule.IsDue); // A check or failed save does not consume it.
        clock.Advance(TimeSpan.FromHours(2));
        schedule.RecordSaved();
        Assert.False(schedule.IsDue); // No catch-up burst after a slow step/save.
        clock.Advance(TimeSpan.FromMinutes(30));
        Assert.True(schedule.IsDue);
    }

    [Fact]
    public void EpochEndSaveAndResumeRestartTheInterval()
    {
        var clock = new ManualClock();
        var schedule = new CheckpointSchedule(clock: clock);
        clock.Advance(TimeSpan.FromMinutes(20));
        schedule.RecordSaved(); // Mandatory epoch-end save before timer is due.
        clock.Advance(TimeSpan.FromMinutes(10));
        Assert.False(schedule.IsDue);
        clock.Advance(TimeSpan.FromMinutes(20));
        Assert.True(schedule.IsDue);
        var resumed = new CheckpointSchedule(clock: clock);
        Assert.False(resumed.IsDue);
        clock.Advance(TimeSpan.FromMinutes(30));
        Assert.True(resumed.IsDue);
    }

    [Fact]
    public void WallClockChangesDoNotAffectTheMonotonicCadence()
    {
        var clock = new ManualClock();
        var schedule = new CheckpointSchedule(5, clock);
        clock.UtcNow = clock.UtcNow.AddDays(10);
        Assert.False(schedule.IsDue);
        clock.UtcNow = clock.UtcNow.AddDays(-20);
        clock.Advance(TimeSpan.FromMinutes(5));
        Assert.True(schedule.IsDue);
    }

    [Theory]
    [InlineData(0d)]
    [InlineData(-1d)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.MaxValue)]
    [InlineData(double.Epsilon)]
    public void RejectsInvalidIntervals(double minutes)
        => Assert.Throws<ArgumentOutOfRangeException>(() => new CheckpointSchedule(minutes));

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void WikiConfigurationAcceptsGroupedAndLegacyIntervals(bool versionTwo, bool grouped)
    {
        string json = "{" + (versionTwo
                ? "\"schemaVersion\":2,\"task\":{\"type\":\"wiki-language-model\"},"
                : "\"task\":\"gpt_rin_wiki_jp\",")
            + (grouped
                ? "\"checkpoint\":{\"directory\":\"checkpoints\",\"intervalMinutes\":17.5}"
                : versionTwo ? "\"training\":{\"checkpointIntervalMinutes\":17.5}"
                : "\"checkpointIntervalMinutes\":17.5") + "}";
        WithConfig(json, path =>
            Assert.Equal(17.5d, WikiTrainingConfiguration.Load(path).CheckpointIntervalMinutes));
    }

    [Fact]
    public void MissingIntervalDefaultsToThirtyAndInvalidGroupedIntervalIsRejected()
    {
        WithConfig("{}", path =>
            Assert.Equal(30d, WikiTrainingConfiguration.Load(path).CheckpointIntervalMinutes));
        WithConfig("""{"checkpoint":{"directory":"checkpoints","intervalMinutes":0}}""", path =>
            Assert.Throws<ArgumentOutOfRangeException>(() => WikiTrainingConfiguration.Load(path)));
    }

    private static void WithConfig(string json, Action<string> action)
    {
        string path = Path.Combine(Path.GetTempPath(), $"nntrain-checkpoint-interval-{Guid.NewGuid():N}.json");
        try
        {
            File.WriteAllText(path, json);
            action(path);
        }
        finally { File.Delete(path); }
    }

    private sealed class ManualClock : TimeProvider
    {
        private long _timestamp;
        internal DateTimeOffset UtcNow { get; set; } = DateTimeOffset.UnixEpoch;
        public override DateTimeOffset GetUtcNow() => UtcNow;
        public override long TimestampFrequency => TimeSpan.TicksPerSecond;
        public override long GetTimestamp() => _timestamp;
        internal void Advance(TimeSpan elapsed) => _timestamp += elapsed.Ticks;
    }
}
