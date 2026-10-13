using System.Threading.Channels;
using NNtrain.Audio;
using NNtrain.Gui;
using Xunit;

namespace NNtrain.IntegrationTests;

public sealed class AudioRecognitionRunnerRefactorTests
{
    [Fact]
    public async Task OneUtteranceKeepsAllSamplesAndAwaitsFinalPublicationBeforeDisposingItsStream()
    {
        var model = new Model();
        var channel = Channel.CreateUnbounded<float[]>();
        channel.Writer.TryWrite([1, 2]);
        channel.Writer.TryWrite([3]);
        channel.Writer.Complete();
        var partials = new List<string>();
        var processed = new List<long>();
        ILocalAsrPreviewControl? control = null;
        var publicationStarted = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var publicationFinished = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        Task recognition = AudioRecognitionRunner.RecognizeAsync(model, channel.Reader,
            preview => control = preview, partials.Add, processed.Add, async (text, cacheBytes) =>
            {
                Assert.Equal("final", text);
                Assert.Equal(123, cacheBytes);
                Assert.False(model.Streams.Single().Disposed);
                publicationStarted.SetResult();
                await publicationFinished.Task;
            }, TestContext.Current.CancellationToken);
        await publicationStarted.Task.WaitAsync(TestContext.Current.CancellationToken);
        Assert.False(recognition.IsCompleted);
        publicationFinished.SetResult();
        await recognition;
        Stream stream = Assert.Single(model.Streams);
        Assert.Same(stream, control);
        Assert.Equal(new[] { 1f, 2f, 3f }, stream.Samples);
        Assert.Equal(new[] { 2L, 3L }, processed);
        Assert.Equal(new[] { "partial", "partial" }, partials);
        Assert.Equal(1, stream.FinalCalls);
        Assert.True(stream.Disposed);
        Assert.False(model.Disposed); // The caller still owns the loaded model.
    }

    [Fact]
    public async Task EachInvocationCreatesFreshUtteranceState()
    {
        var model = new Model();
        for (int utterance = 0; utterance < 2; utterance++)
        {
            var channel = Channel.CreateUnbounded<float[]>();
            channel.Writer.TryWrite([(float)utterance]);
            channel.Writer.Complete();
            await AudioRecognitionRunner.RecognizeAsync(model, channel.Reader, _ => { }, _ => { }, _ => { },
                (_, _) => Task.CompletedTask, TestContext.Current.CancellationToken);
        }
        Assert.Equal(2, model.Streams.Count);
        Assert.Equal(new[] { 0f }, model.Streams[0].Samples);
        Assert.Equal(new[] { 1f }, model.Streams[1].Samples);
        Assert.All(model.Streams, stream => Assert.True(stream.Disposed));
    }

    [Fact]
    public async Task CancellationDisposesTheStreamWithoutPublishingAFinalResult()
    {
        using var cancellation = CancellationTokenSource.CreateLinkedTokenSource(TestContext.Current.CancellationToken);
        var model = new Model();
        var channel = Channel.CreateUnbounded<float[]>();
        channel.Writer.TryWrite([1]);
        int completions = 0;
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => AudioRecognitionRunner.RecognizeAsync(
            model, channel.Reader, _ => { }, _ => cancellation.Cancel(), _ => { }, (_, _) =>
            {
                completions++;
                return Task.CompletedTask;
            }, cancellation.Token));
        Assert.Equal(0, completions);
        Assert.True(Assert.Single(model.Streams).Disposed);
    }

    [Fact]
    public async Task RecordingDurationLimitReportsFailureAndDisposesTheStream()
    {
        var model = new Model();
        var channel = Channel.CreateUnbounded<float[]>();
        channel.Writer.TryWrite(new float[16000 * 300 + 1]);
        channel.Writer.Complete();
        InvalidOperationException error = await Assert.ThrowsAsync<InvalidOperationException>(() =>
            AudioRecognitionRunner.RecognizeAsync(model, channel.Reader, _ => { }, _ => { }, _ => { },
                (_, _) => throw new InvalidOperationException("Unexpected completion"), TestContext.Current.CancellationToken));
        Assert.Equal("録音は5分までです。", error.Message);
        Stream stream = Assert.Single(model.Streams);
        Assert.Empty(stream.Samples);
        Assert.True(stream.Disposed);
    }

    [Fact]
    public async Task WaveFileRecognitionRetains5120SamplePacingAndFinalizesOnce()
    {
        string path = Path.Combine(Path.GetTempPath(), "nntrain-asr-refactor-" + Guid.NewGuid().ToString("N") + ".wav");
        try
        {
            using (var writer = new BinaryWriter(File.Create(path)))
            {
                const int samples = 11000;
                writer.Write("RIFF"u8);
                writer.Write(36 + samples * 2);
                writer.Write("WAVEfmt "u8);
                writer.Write(16);
                writer.Write((short)1);
                writer.Write((short)1);
                writer.Write(16000);
                writer.Write(32000);
                writer.Write((short)2);
                writer.Write((short)16);
                writer.Write("data"u8);
                writer.Write(samples * 2);
                for (int i = 0; i < samples; i++) writer.Write((short)8192);
            }
            var model = new Model();
            string? completed = null;
            await AudioRecognitionRunner.RecognizeFileAsync(model, path, _ => { }, (text, _) =>
            {
                completed = text;
                return Task.CompletedTask;
            }, TestContext.Current.CancellationToken);
            Stream stream = Assert.Single(model.Streams);
            Assert.Equal(new[] { 5120, 5120, 760, 0 }, stream.InputLengths);
            Assert.Equal(11000, stream.Samples.Count);
            Assert.All(stream.Samples, sample => Assert.Equal(0.25f, sample));
            Assert.Equal("final", completed);
            Assert.Equal(1, stream.FinalCalls);
            Assert.True(stream.Disposed);
        }
        finally { File.Delete(path); }
    }

    private sealed class Model : ILocalAsrModel
    {
        internal List<Stream> Streams { get; } = [];
        internal bool Disposed { get; private set; }
        public long HostWeightBytes => 0;
        public long PeakDeviceBufferBytes => 0;
        public long ResidentDeviceBytes => 0;
        public string ExecutionDevice => "test";
        public void EnableArc(int deviceIndex, CancellationToken ct = default) => throw new NotSupportedException();
        public ILocalAsrStream CreateStream()
        {
            var stream = new Stream();
            Streams.Add(stream);
            return stream;
        }
        public void Dispose() => Disposed = true;
    }

    private sealed class Stream : ILocalAsrStream, ILocalAsrPreviewControl, IDisposable
    {
        internal List<float> Samples { get; } = [];
        internal List<int> InputLengths { get; } = [];
        internal bool Disposed { get; private set; }
        internal int FinalCalls { get; private set; }
        public long CacheBytes => 123;
        public string Append(ReadOnlySpan<float> samples, bool final = false, Action<string>? partial = null,
            CancellationToken ct = default)
        {
            ct.ThrowIfCancellationRequested();
            Samples.AddRange(samples.ToArray());
            InputLengths.Add(samples.Length);
            if (final) FinalCalls++;
            else partial?.Invoke("partial");
            return final ? "final" : "partial";
        }
        public void RequestFinalization() { }
        public void Dispose() => Disposed = true;
    }
}
