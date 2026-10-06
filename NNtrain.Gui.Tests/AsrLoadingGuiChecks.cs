using System.Diagnostics;
using System.Reflection;
using System.Text.Json;
using System.Windows.Controls;
using System.Windows.Threading;
using NNtrain.Audio;
using NNtrain.Gui;
using Xunit;

namespace NNtrain.Gui.Tests;

// Explicit CPU-only real-model checks. The window is never shown and Record is never invoked.
public static class AsrLoadingGuiChecks
{
    public static void Run()
    {
        Exception? failure = null;
        var results = new List<object>();
        var thread = new Thread(() =>
        {
            SynchronizationContext.SetSynchronizationContext(new DispatcherSynchronizationContext(Dispatcher.CurrentDispatcher));
            MainWindow? window = null;
            try
            {
                window = new MainWindow(GuiLaunchOptions.Parse([]));
                var gpu = (ComboBox)window.FindName("AsrGpuComboBox");
                gpu.SelectedItem = gpu.Items.Cast<object>().Single(item =>
                    ((int[])item.GetType().GetProperty("DeviceIndices")!.GetValue(item)!).Length == 0);
                var selector = (ComboBox)window.FindName("AsrModelComboBox");
                selector.SelectedIndex = 0;
                string nemotron = Environment.GetEnvironmentVariable("NNTRAIN_ASR_NEMOTRON_MODEL")!;
                string parakeet = Environment.GetEnvironmentVariable("NNTRAIN_ASR_PARAKEET_MODEL")!;

                StartLoad(window, nemotron); Pump(window, () => !Active(window), TimeSpan.FromMilliseconds(100), allowTimeout: true);
                Assert.True(Active(window)); Invoke(window, "CancelAudio"); Finish(window);
                Assert.Null(Field(window, "_asrModel")); CheckIdle(window); results.Add(new { check = "cancelDuringNemotronLoad", passed = true });

                var clock = Stopwatch.StartNew(); StartLoad(window, nemotron); Finish(window);
                Assert.IsType<NemotronAsrModel>(Field(window, "_asrModel")); CheckIdle(window);
                results.Add(new { check = "reloadNemotronAfterCancellation", passed = true, seconds = clock.Elapsed.TotalSeconds });

                StartLoad(window, nemotron); Pump(window, () => !Active(window), TimeSpan.FromMilliseconds(100), allowTimeout: true);
                Assert.True(Active(window)); selector.SelectedIndex = 1; Finish(window);
                Pump(window, () => ((TextBox)window.FindName("AsrDirectoryBox")).Text.EndsWith("parakeet-tdt_ctc-0.6b-ja"), TimeSpan.FromSeconds(5));
                Assert.Null(Field(window, "_asrModel")); CheckIdle(window);
                results.Add(new { check = "backendSwitchCancelsInFlightReload", passed = true });

                clock.Restart(); StartLoad(window, parakeet); Finish(window);
                var previous = Assert.IsType<ParakeetCtcModel>(Field(window, "_asrModel")); CheckIdle(window);
                results.Add(new { check = "loadParakeetAfterBackendSwitch", passed = true, seconds = clock.Elapsed.TotalSeconds });

                clock.Restart(); StartLoad(window, parakeet); Finish(window);
                Assert.NotSame(previous, Assert.IsType<ParakeetCtcModel>(Field(window, "_asrModel"))); CheckIdle(window);
                results.Add(new { check = "reloadParakeetReplacesModel", passed = true, seconds = clock.Elapsed.TotalSeconds });

                selector.SelectedIndex = 0;
                Pump(window, () => Field(window, "_asrModel") is null, TimeSpan.FromSeconds(5)); CheckIdle(window);
                Assert.True(((TextBox)window.FindName("AsrDirectoryBox")).Text.EndsWith("nemotron-3.5-asr-streaming-0.6b"));
                results.Add(new { check = "switchBackClearsPreviousModel", passed = true });
            }
            catch (Exception ex) { failure = ex; }
            finally
            {
                if (window is not null) { Invoke(window, "CancelAudio"); window.Close(); }
                Dispatcher.CurrentDispatcher.Invoke(() => { }, DispatcherPriority.ApplicationIdle);
                Dispatcher.CurrentDispatcher.InvokeShutdown();
            }
        }) { IsBackground = true };
        thread.SetApartmentState(ApartmentState.STA); thread.Start();
        Assert.True(thread.Join(TimeSpan.FromSeconds(60)), "CPU ASR loading GUI checks timed out.");
        if (failure is not null) System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(failure).Throw();
        File.WriteAllText(Environment.GetEnvironmentVariable("NNTRAIN_ASR_LOAD_GUI_REPORT")!,
            JsonSerializer.Serialize(new { results, gpuOpened = false, microphoneOpened = false, llmStarted = false }, new JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine("CPU ASR loading GUI checks passed: 6 (cancel, reload and backend switch). No GPU, microphone or server started.");
    }

    private static void StartLoad(MainWindow window, string directory)
    {
        ((TextBox)window.FindName("AsrDirectoryBox")).Text = directory;
        Invoke(window, "AsrLoad_Click", null!, null!);
    }
    private static bool Active(MainWindow window) => (bool)Field(window, "_asrActive")!;
    private static void Finish(MainWindow window) => Pump(window, () => !Active(window), TimeSpan.FromSeconds(30));
    private static void CheckIdle(MainWindow window)
    {
        Assert.False(Active(window)); Assert.Null(Field(window, "_microphone")); Assert.Null(Field(window, "_serverProcess"));
        Assert.True(((Button)window.FindName("AsrLoadButton")).IsEnabled);
        Assert.True(((ComboBox)window.FindName("AsrModelComboBox")).IsEnabled);
        Assert.False(((TextBox)window.FindName("MessageBox")).IsReadOnly);
        if (Field(window, "_asrModel") is ILocalAsrModel model) Assert.Equal(0, model.PeakDeviceBufferBytes);
    }
    private static void Pump(MainWindow window, Func<bool> completed, TimeSpan timeout, bool allowTimeout = false)
    {
        var clock = Stopwatch.StartNew();
        while (!completed() && clock.Elapsed < timeout)
        {
            window.Dispatcher.Invoke(() => { }, DispatcherPriority.ApplicationIdle); Thread.Sleep(2);
        }
        if (!allowTimeout) Assert.True(completed(), "ASR GUI operation did not reach its expected state.");
    }
    private static object? Field(MainWindow window, string name)
        => typeof(MainWindow).GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(window);
    private static void Invoke(MainWindow window, string name, params object[] args)
        => typeof(MainWindow).GetMethod(name, BindingFlags.Instance | BindingFlags.NonPublic)!.Invoke(window, args);
}
