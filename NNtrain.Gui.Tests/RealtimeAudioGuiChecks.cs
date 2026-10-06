using System.Diagnostics;
using System.Reflection;
using System.Text.Json;
using System.Threading.Channels;
using System.Windows.Controls;
using System.Windows.Threading;
using NNtrain.Audio;
using NNtrain.Core.Tests;
using NNtrain.Gui;

namespace NNtrain.Gui.Tests;

// Paced PCM injection uses exactly the recording consumer and real WPF dispatcher.
// The window is never shown and no microphone/server is constructed.
internal static class RealtimeAudioGuiChecks
{
    private static object? Get(MainWindow w, string n) => typeof(MainWindow).GetField(n, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(w);
    private static void Set(MainWindow w, string n, object? value) => typeof(MainWindow).GetField(n, BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(w, value);
    private static object? Call(MainWindow w, string n, params object[] args) => typeof(MainWindow).GetMethod(n, BindingFlags.Instance | BindingFlags.NonPublic)!.Invoke(w, args);

    public static void Run()
    {
        Exception? error = null;
        var thread = new Thread(() =>
        {
            SynchronizationContext.SetSynchronizationContext(new DispatcherSynchronizationContext());
            _ = Dispatcher.CurrentDispatcher.BeginInvoke(new Action(async () =>
            {
                try { await RunAsync(); } catch (Exception ex) { error = ex; }
                finally { Dispatcher.CurrentDispatcher.InvokeShutdown(); }
            }));
            Dispatcher.Run();
        });
        thread.SetApartmentState(ApartmentState.STA); thread.Start();
        if (!thread.Join(TimeSpan.FromMinutes(5))) throw new TimeoutException("Realtime GUI test timed out.");
        if (error is not null) System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(error).Throw();
    }

    private static async Task RunAsync()
    {
        using var timeout = new CancellationTokenSource(TimeSpan.FromMinutes(4));
        CancellationToken ct = timeout.Token;
        var window = new MainWindow(GuiLaunchOptions.Parse([]));
        using var memory = new AsrGpuMemoryProbe();
        using var model = await Task.Run(() =>
        {
            var m = NemotronAsrModel.Load(Environment.GetEnvironmentVariable("NNTRAIN_ASR_MODEL")!, ct);
            try { m.EnableArc(0, ct); return m; } catch { m.Dispose(); throw; }
        }, ct);
        Set(window, "_asrModel", model);
        try
        {
            using var file = System.IO.File.OpenRead(Environment.GetEnvironmentVariable("NNTRAIN_ASR_TEST_WAV")!);
            float[] speech = Pcm16Wave.Read(file).To16Khz(ct);
            float[] repeated = Enumerable.Range(0, 5).SelectMany(_ => speech.Concat(new float[16000])).ToArray();
            var longInput = await Measure(window, repeated, cancelAfter: null, ct);
            Console.WriteLine("Long paced input complete: " + longInput.AudioSeconds + " s.");
            var cancelled = await Measure(window, speech, cancelAfter: 3.2, ct);
            if (!string.IsNullOrEmpty(((TextBox)window.FindName("MessageBox")).Text)) throw new InvalidOperationException("Cancelled transcript remained.");
            var restart = await Measure(window, speech, cancelAfter: null, ct);
            string reference = System.IO.File.ReadAllText(Environment.GetEnvironmentVariable("NNTRAIN_ASR_REFERENCE")!);
            if (longInput.UpdatesBeforeStop < 10 || longInput.FirstTextSeconds is null || longInput.FirstTextSeconds >= longInput.AudioSeconds)
                throw new InvalidOperationException("No live GUI updates before stop.");
            if (longInput.TailMaxProcessingLagSeconds > 1.5 || longInput.StopToFinalSeconds > 1.5 || longInput.FinalQueue != 0)
                throw new InvalidOperationException("Recognition backlog did not settle.");
            if (restart.FinalText.Length > longInput.FinalText.Length / 2 || restart.FinalText.Length == 0)
                throw new InvalidOperationException("Restart inherited the old utterance.");
            if (!longInput.DoubleStartRejected || !restart.DoubleStartRejected)
                throw new InvalidOperationException("Double start was not rejected.");
            string json = JsonSerializer.Serialize(new { longInput, cancelled, restart,
                longCharacterErrorRate = CharacterErrorRate(string.Concat(Enumerable.Repeat(reference, 5)), longInput.FinalText),
                restartCharacterErrorRate = CharacterErrorRate(reference, restart.FinalText),
                microphoneOpened = Get(window, "_microphone") is not null, serverStarted = Get(window, "_serverProcess") is not null,
                osDedicatedGpuPeakBytes = memory.DedicatedPeakBytes, osSharedGpuPeakBytes = memory.SharedPeakBytes,
                osMemorySamples = memory.Samples, osMemoryError = memory.Error, peakDeviceBufferBytes = model.PeakDeviceBufferBytes },
                new JsonSerializerOptions { WriteIndented = true });
            System.IO.File.WriteAllText(Environment.GetEnvironmentVariable("NNTRAIN_ASR_REALTIME_REPORT")!, json);
            Console.WriteLine(json);
        }
        finally { Set(window, "_asrModel", null); window.Close(); await window.Dispatcher.InvokeAsync(() => { }, DispatcherPriority.ApplicationIdle); }
    }

    private sealed record Observation(double Seconds, double ProcessedAudioSeconds, int Queue, double ProcessingLagSeconds);
    private static double CharacterErrorRate(string reference, string actual)
    {
        var a = reference.EnumerateRunes().Where(System.Text.Rune.IsLetterOrDigit).ToArray();
        var b = actual.EnumerateRunes().Where(System.Text.Rune.IsLetterOrDigit).ToArray();
        int[] previous = Enumerable.Range(0, b.Length + 1).ToArray();
        for (int i = 1; i <= a.Length; i++)
        {
            var next = new int[b.Length + 1]; next[0] = i;
            for (int j = 1; j <= b.Length; j++) next[j] = Math.Min(Math.Min(next[j - 1] + 1, previous[j] + 1), previous[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1));
            previous = next;
        }
        return previous[^1] / (double)Math.Max(1, a.Length);
    }
    private sealed record Update(double Seconds, int Characters);
    private sealed record Result(double AudioSeconds, bool Cancelled, double? FirstTextSeconds, int UpdatesBeforeStop,
        double StopToFinalSeconds, int MaxQueue, int FinalQueue, double MaxProcessingLagSeconds, double TailMaxProcessingLagSeconds,
        string FinalText, Update[] Updates, Observation[] Observations, bool DoubleStartRejected);

    private static async Task<Result> Measure(MainWindow window, float[] samples, double? cancelAfter, CancellationToken ct)
    {
        var channel = Channel.CreateBounded<float[]>(new BoundedChannelOptions(16) { SingleReader = true, SingleWriter = true });
        var clock = Stopwatch.StartNew();
        var updates = new List<Update>(); var observations = new List<Observation>();
        var box = (TextBox)window.FindName("MessageBox");
        box.Clear();
        TextChangedEventHandler onText = (_, _) => { if (box.Text.Length != 0) updates.Add(new(clock.Elapsed.TotalSeconds, box.Text.Length)); };
        box.TextChanged += onText;
        double stoppedAt = 0; int maxQueue = 0;
        Task operation = (Task)Call(window, "RunAudioAsync", new Func<long, CancellationToken, Task>((epoch, token)
            => (Task)Call(window, "ProcessAudioAsync", channel.Reader, epoch, token)!))!;
        bool secondStarted = false;
        await (Task)Call(window, "RunAudioAsync", new Func<long, CancellationToken, Task>((_, _) => { secondStarted = true; return Task.CompletedTask; }))!;
        Task producer = Task.Run(async () =>
        {
            try
            {
                for (int offset = 0; offset < samples.Length; offset += 5120)
                {
                    int count = Math.Min(5120, samples.Length - offset);
                    double due = (offset + count) / 16000d;
                    if (cancelAfter is { } cancel && due > cancel) break;
                    double delay = due - clock.Elapsed.TotalSeconds;
                    if (delay > 0) await Task.Delay(TimeSpan.FromSeconds(delay), ct);
                    if (!channel.Writer.TryWrite(samples.AsSpan(offset, count).ToArray())) throw new InvalidOperationException("Capture queue overflow.");
                }
            }
            finally { if (cancelAfter is null) channel.Writer.TryComplete(); stoppedAt = clock.Elapsed.TotalSeconds; }
        }, ct);
        try
        {
            while (!producer.IsCompleted)
            {
                await Task.Delay(20, ct);
                double elapsed = clock.Elapsed.TotalSeconds;
                double processed = (long)Get(window, "_asrProcessedSamples")! / 16000d;
                maxQueue = Math.Max(maxQueue, channel.Reader.Count);
                observations.Add(new(elapsed, processed, channel.Reader.Count, Math.Max(0, elapsed - processed)));
            }
            await producer;
            if (cancelAfter is not null) { Call(window, "CancelAudio"); channel.Writer.TryComplete(); }
            await operation;
            await window.Dispatcher.InvokeAsync(() => { }, DispatcherPriority.ApplicationIdle);
            double duration = cancelAfter ?? samples.Length / 16000d;
            return new(duration, cancelAfter is not null, updates.Count == 0 ? null : updates[0].Seconds,
                updates.Count(x => x.Seconds < stoppedAt), clock.Elapsed.TotalSeconds - stoppedAt, maxQueue, channel.Reader.Count,
                observations.Max(x => x.ProcessingLagSeconds), observations.Where(x => x.Seconds > duration - Math.Min(20, duration / 2)).Max(x => x.ProcessingLagSeconds),
                box.Text, updates.ToArray(), observations.ToArray(), !secondStarted);
        }
        finally { box.TextChanged -= onText; }
    }
}
