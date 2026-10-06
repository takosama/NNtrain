using System.Diagnostics;
using System.Reflection;
using System.Text.Json;
using System.Threading.Channels;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Threading;
using NNtrain.Audio;
using NNtrain.Core.Tests;
namespace NNtrain.Gui.Tests;

// Actual ASR + real recording consumer + edited composer + mocked local LLM transport.
// No window Loaded event, microphone or LLM server is started.
internal static class RealComposerChecks
{
    private static object? Get(MainWindow w, string name) => typeof(MainWindow).GetField(name, BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(w);
    private static void Set(MainWindow w, string name, object? value) => typeof(MainWindow).GetField(name, BindingFlags.NonPublic | BindingFlags.Instance)!.SetValue(w, value);
    private static object? Call(MainWindow w, string name, params object[] values) => typeof(MainWindow).GetMethod(name, BindingFlags.NonPublic | BindingFlags.Instance)!.Invoke(w, values);
    public static void Run()
    {
        Exception? error = null;
        var thread = new Thread(() =>
        {
            SynchronizationContext.SetSynchronizationContext(new DispatcherSynchronizationContext());
            _ = Dispatcher.CurrentDispatcher.BeginInvoke(new Action(async () =>
            { try { await RunAsync(); } catch (Exception ex) { error = ex; } finally { Dispatcher.CurrentDispatcher.InvokeShutdown(); } }));
            Dispatcher.Run();
        });
        thread.SetApartmentState(ApartmentState.STA); thread.Start();
        if (!thread.Join(TimeSpan.FromMinutes(5))) throw new TimeoutException("Composer test timeout.");
        if (error is not null) System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(error).Throw();
    }
    private static async Task RunAsync()
    {
        using var timeout = new CancellationTokenSource(TimeSpan.FromMinutes(4)); CancellationToken ct = timeout.Token;
        var window = new MainWindow(GuiLaunchOptions.Parse([]));
        using var marker = Process.GetCurrentProcess(); string fakeModel = System.IO.Path.GetTempFileName();
        using var memory = new AsrGpuMemoryProbe(); var clock = Stopwatch.StartNew();
        try
        {
            bool parakeet = Environment.GetEnvironmentVariable("NNTRAIN_ASR_COMPOSER_MODEL") == "parakeet";
            ((ComboBox)window.FindName("AsrModelComboBox")).SelectedIndex = parakeet ? 1 : 0;
            ((TextBox)window.FindName("AsrDirectoryBox")).Text = Environment.GetEnvironmentVariable("NNTRAIN_ASR_MODEL")!;
            Call(window, "AsrLoad_Click", null!, new RoutedEventArgs());
            while ((bool)Get(window, "_asrActive")!) await Task.Delay(30, ct);
            var model = (ILocalAsrModel?)Get(window, "_asrModel") ?? throw new InvalidOperationException(((TextBlock)window.FindName("AsrStatusText")).Text);
            double loadSeconds = clock.Elapsed.TotalSeconds;
            var input = (TextBox)window.FindName("MessageBox"); input.Text = "元の下書き"; input.CaretIndex = input.Text.Length;
            int updates = 0; input.TextChanged += (_, _) => updates++;
            var channel = Channel.CreateBounded<float[]>(16);
            clock.Restart();
            Task operation = (Task)Call(window, "RunAudioAsync", new Func<long, CancellationToken, Task>((epoch, token) =>
                (Task)Call(window, "ProcessAudioAsync", channel.Reader, epoch, token)!))!;
            using (var file = System.IO.File.OpenRead(Environment.GetEnvironmentVariable("NNTRAIN_ASR_TEST_WAV")!))
            {
                float[] samples = Pcm16Wave.Read(file).To16Khz(ct);
                for (int offset = 0; offset < samples.Length; offset += 5120)
                    await channel.Writer.WriteAsync(samples.AsSpan(offset, Math.Min(5120, samples.Length - offset)).ToArray(), ct);
            }
            channel.Writer.TryComplete(); await operation;
            await window.Dispatcher.InvokeAsync(() => { }, DispatcherPriority.ApplicationIdle);
            double recognitionSeconds = clock.Elapsed.TotalSeconds;
            if (input.IsReadOnly || !input.Text.StartsWith("元の下書き\n", StringComparison.Ordinal)) throw new InvalidOperationException("Draft/readonly composer contract failed.");
            string recognized = input.Text["元の下書き\n".Length..];
            if (string.IsNullOrWhiteSpace(recognized)) throw new InvalidOperationException("Actual recognition returned no text.");
            input.Text += "\nこの内容を要約してください。"; string edited = input.Text;
            var handler = new Transport(fakeModel);
            ((System.Net.Http.HttpClient)Get(window, "_client")!).Dispose();
            Set(window, "_client", new System.Net.Http.HttpClient(handler) { BaseAddress = new Uri("http://localhost/") });
            Set(window, "_serverProcess", marker); Set(window, "_serverReady", true);
            var models = (ComboBox)window.FindName("ModelComboBox"); models.ItemsSource = null; models.Items.Clear(); models.Items.Add(new MainWindow.FileChoice("test", fakeModel)); models.SelectedIndex = 0;
            ((CheckBox)window.FindName("StreamCheckBox")).IsChecked = false;
            ((CheckBox)window.FindName("ThinkingCheckBox")).IsChecked = false;
            Call(window, "Send_Click", null!, new RoutedEventArgs());
            while ((bool)Get(window, "_busy")!) await Task.Delay(30, ct);
            if (handler.Body is null) throw new InvalidOperationException("Existing chat transport was not called.");
            using var body = JsonDocument.Parse(handler.Body);
            string? sent = body.RootElement.GetProperty("messages")[0].GetProperty("content").GetString();
            if (sent != edited || input.Text.Length != 0) throw new InvalidOperationException("Edited prompt send failed.");
            var report = new { model = parakeet ? "Parakeet Japanese CTC" : "Nemotron Japanese", recognized, edited, sent,
                loadSeconds, recognitionSeconds, updatesBeforeEdit = updates - 2, inputMode = "unpaced file through recording consumer",
                llmTransport = "mocked local HTTP handler; no LLM inference", microphoneOpened = Get(window, "_microphone") is not null,
                hostWeightBytes = model.HostWeightBytes, residentDeviceBytes = model.ResidentDeviceBytes, peakDeviceBufferBytes = model.PeakDeviceBufferBytes,
                osDedicatedGpuPeakBytes = memory.DedicatedPeakBytes, osSharedGpuPeakBytes = memory.SharedPeakBytes, osMemorySamples = memory.Samples, osMemoryError = memory.Error };
            string json = JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true });
            System.IO.File.WriteAllText(Environment.GetEnvironmentVariable("NNTRAIN_ASR_COMPOSER_REPORT")!, json); Console.WriteLine(json);
        }
        finally
        {
            Set(window, "_serverProcess", null); Set(window, "_serverReady", false);
            ((ILocalAsrModel?)Get(window, "_asrModel"))?.Dispose(); Set(window, "_asrModel", null);
            window.Close(); System.IO.File.Delete(fakeModel);
        }
    }
    private sealed class Transport(string model) : System.Net.Http.HttpMessageHandler
    {
        public string? Body;
        protected override async Task<System.Net.Http.HttpResponseMessage> SendAsync(System.Net.Http.HttpRequestMessage request, CancellationToken ct)
        {
            string json = "{}";
            if (request.RequestUri!.AbsolutePath == "/internal/state") json = JsonSerializer.Serialize(new { is_loaded = true, model, devices = new[] { 0, 1 } });
            if (request.RequestUri.AbsolutePath == "/v1/chat/completions")
            { Body = await request.Content!.ReadAsStringAsync(ct); json = "{\"choices\":[{\"message\":{\"content\":\"test response\"},\"finish_reason\":\"stop\"}]}"; }
            return new System.Net.Http.HttpResponseMessage(System.Net.HttpStatusCode.OK) { Content = new System.Net.Http.StringContent(json, System.Text.Encoding.UTF8, "application/json") };
        }
    }
}
