using System.Collections;
using System.Diagnostics;
using System.Reflection;
using System.Text.Json;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Threading;
using NNtrain.Audio;
using NNtrain.Core.Tests;
using NNtrain.Gui;

namespace NNtrain.Gui.Tests;

// Explicit opt-in, file-only integration check. Never shows/loads the window or opens a microphone.
internal static class RealAudioGuiChecks
{
    private static object? Get(MainWindow w, string n) => typeof(MainWindow).GetField(n, BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(w);
    private static void Set(MainWindow w, string n, object? v) => typeof(MainWindow).GetField(n, BindingFlags.NonPublic | BindingFlags.Instance)!.SetValue(w, v);
    private static object? Call(MainWindow w, string n, params object[] args) => typeof(MainWindow).GetMethod(n, BindingFlags.NonPublic | BindingFlags.Instance)!.Invoke(w, args);
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
        if (!thread.Join(TimeSpan.FromMinutes(8))) throw new TimeoutException("Real GUI integration timed out.");
        if (error is not null) System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(error).Throw();
    }

    private static async Task RunAsync()
    {
        using var timeout = new CancellationTokenSource(TimeSpan.FromMinutes(6));
        CancellationToken ct = timeout.Token;
        string modelPath = Environment.GetEnvironmentVariable("NNTRAIN_ASR_LLM")!;
        var window = new MainWindow(GuiLaunchOptions.Parse(["--model", modelPath, "--think", "off", "--stream", "off", "--maxtokens", "48"]));
        await using var server = new LocalOpenAiServer();
        using var process = Process.GetCurrentProcess();
        using var memory = new AsrGpuMemoryProbe();
        NemotronAsrModel? asr = null;
        var clock = Stopwatch.StartNew();
        try
        {
            Uri address = await server.StartAsync(0, null, ct);
            ((System.Net.Http.HttpClient)Get(window, "_client")!).BaseAddress = address;
            Set(window, "_serverProcess", process); // Own loopback server runs in this process.
            Set(window, "_serverReady", true);
            var gpu = (ComboBox)window.FindName("GpuComboBox");
            gpu.SelectedItem = gpu.Items.Cast<MainWindow.GpuChoice>().First(x => x.DeviceIndices.SequenceEqual(new[] { 0, 1 }));
            asr = await Task.Run(() =>
            {
                var model = NemotronAsrModel.Load(Environment.GetEnvironmentVariable("NNTRAIN_ASR_MODEL")!, ct);
                try { model.EnableArc(0, ct); return model; } catch { model.Dispose(); throw; }
            }, ct);
            Set(window, "_asrModel", asr);
            Call(window, "BeginAudioChatInput");
            double loadSeconds = clock.Elapsed.TotalSeconds;
            clock.Restart();
            string recognized = await Task.Run(() =>
            {
                using var file = System.IO.File.OpenRead(Environment.GetEnvironmentVariable("NNTRAIN_ASR_TEST_WAV")!);
                float[] samples = Pcm16Wave.Read(file).To16Khz(ct);
                var stream = asr.CreateStream();
                long epoch = (long)Get(window, "_asrEpoch")!;
                for (int offset = 0; offset < samples.Length; offset += 5120)
                    stream.Append(samples.AsSpan(offset, Math.Min(5120, samples.Length - offset)), partial: text => Call(window, "ShowAudioPartial", epoch, text), ct: ct);
                string text = stream.Append([], true, ct: ct);
                Call(window, "CompleteAudio", epoch, text, stream.CacheBytes);
                return text;
            }, ct);
            double recognitionSeconds = clock.Elapsed.TotalSeconds;
            await window.Dispatcher.InvokeAsync(() => { }, DispatcherPriority.ApplicationIdle);
            var transcript = (TextBox)window.FindName("MessageBox");
            if (transcript.Text != recognized.Trim()) throw new InvalidOperationException("Real recognition did not reach the GUI transcript.");
            Set(window, "_asrRecognizing", false); Call(window, "UpdateAudioControls");
            transcript.Text = "次の文章を短く要約してください。\n" + recognized;
            string sent = transcript.Text.Trim();
            clock.Restart();
            Call(window, "Send_Click", null!, new RoutedEventArgs());
            while ((bool)Get(window, "_busy")!) await Task.Delay(100, ct);
            var turns = ((IEnumerable)Get(window, "_conversation")!).Cast<ChatTurn>().ToArray();
            if (turns.Length != 2 || turns[0].Content != sent || turns[1].Role != "assistant" || string.IsNullOrWhiteSpace(turns[1].Content))
                throw new InvalidOperationException("Real LLM response missing: " + ((TextBlock)window.FindName("StatusText")).Text);
            string report = JsonSerializer.Serialize(new { modelPath, devices = new[] { 0, 1 }, recognized, sent,
                response = turns[1].Content, asrLoadSeconds = loadSeconds, recognitionSeconds,
                llmLoadAndResponseSeconds = clock.Elapsed.TotalSeconds, asrResidentDeviceBytes = asr.ResidentDeviceBytes,
                osDedicatedGpuPeakBytes = memory.DedicatedPeakBytes, osSharedGpuPeakBytes = memory.SharedPeakBytes,
                osMemorySamples = memory.Samples, osMemoryError = memory.Error, microphoneOpened = Get(window, "_microphone") is not null },
                new JsonSerializerOptions { WriteIndented = true });
            System.IO.File.WriteAllText(Environment.GetEnvironmentVariable("NNTRAIN_ASR_GUI_REPORT")!, report);
            Console.WriteLine(report);
        }
        finally
        {
            Set(window, "_serverProcess", null); Set(window, "_serverReady", false);
            asr?.Dispose(); Set(window, "_asrModel", null);
            window.Close();
            await window.Dispatcher.InvokeAsync(() => { }, DispatcherPriority.ApplicationIdle);
        }
    }
}
