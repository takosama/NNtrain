using System.Reflection;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Threading;
using NNtrain.Gui;
using Xunit;

namespace NNtrain.Gui.Tests;

public sealed class AudioGuiTests
{
    [Fact]
    public void JapaneseModelSelectorChangesBackendFolderAndReleasesPreviousModel()
        => OnSta(window =>
        {
            var selector = (ComboBox)window.FindName("AsrModelComboBox");
            Assert.Equal(2, selector.Items.Count);
            var previous = new SelectionModel(); Set(window, "_asrModel", previous);
            selector.SelectedIndex = 1;
            Assert.True(previous.Disposed); Assert.Null(Field(window, "_asrModel"));
            Assert.EndsWith("parakeet-tdt_ctc-0.6b-ja", Box(window, "AsrDirectoryBox").Text);
            Assert.Contains("CTC", ((TextBlock)window.FindName("AsrStatusText")).Text);
            selector.SelectedIndex = 0;
            Assert.EndsWith("nemotron-3.5-asr-streaming-0.6b", Box(window, "AsrDirectoryBox").Text);
            Assert.False(Box(window, "MessageBox").IsReadOnly);
        });
    private sealed class SelectionModel : NNtrain.Audio.ILocalAsrModel
    {
        public bool Disposed;
        public long HostWeightBytes => 0; public long PeakDeviceBufferBytes => 0; public long ResidentDeviceBytes => 0;
        public string ExecutionDevice => "selection test";
        public void EnableArc(int deviceIndex, CancellationToken ct = default) => throw new NotSupportedException();
        public NNtrain.Audio.ILocalAsrStream CreateStream() => throw new NotSupportedException();
        public void Dispose() => Disposed = true;
    }
    [Fact]
    public void AudioInsertionKeepsSelectedTextAndSuffix()
    {
        var draft = new AudioChatDraft("先頭選択末尾", 2, 2);
        Assert.Equal("先頭選択\n音声\n末尾", draft.Compose("音声"));
        Assert.Equal("先頭選択末尾", draft.Compose(""));
    }

    [Fact]
    public void NavigatingTabsKeepsLiveTextAndModelSwitchRestoresDraft()
        => OnSta(window =>
        {
            var tabs = (TabControl)window.Content;
            Assert.Equal(2, tabs.Items.Count);
            Assert.Null(window.FindName("AsrTranscriptBox"));
            Box(window, "MessageBox").Text = "保存する手入力";
            Box(window, "MessageBox").CaretIndex = Box(window, "MessageBox").Text.Length;
            Invoke(window, "BeginAudioChatInput");
            long epoch = (long)Field(window, "_asrEpoch")!;
            tabs.SelectedIndex = 1;
            Invoke(window, "ShowAudioPartial", epoch, "別画面でも録音の文字");
            window.Dispatcher.Invoke(() => { }, DispatcherPriority.ApplicationIdle);
            Assert.Equal("保存する手入力\n別画面でも録音の文字", Box(window, "MessageBox").Text);
            tabs.SelectedIndex = 0;
            Invoke(window, "ModelOrAdapter_SelectionChanged", window.FindName("ModelComboBox"),
                new SelectionChangedEventArgs(System.Windows.Controls.Primitives.Selector.SelectionChangedEvent, Array.Empty<object>(), Array.Empty<object>()));
            Invoke(window, "ShowAudioPartial", epoch, "切替前の遅れた文字");
            window.Dispatcher.Invoke(() => { }, DispatcherPriority.ApplicationIdle);
            Assert.Equal("保存する手入力", Box(window, "MessageBox").Text);
            Set(window, "_asrRecognizing", false);
        });

    [Fact]
    public void IntegratedLayoutFitsSmallAndDefaultWindowSizes()
        => OnSta(window =>
        {
            var content = (TabControl)window.Content;
            foreach (var size in new[] { new Size(760, 580), new Size(1080, 760) })
            {
                content.Measure(size); content.Arrange(new Rect(size)); content.UpdateLayout();
                TextBox input = Box(window, "MessageBox");
                Point location = input.TranslatePoint(new Point(), content);
                Assert.True(input.ActualWidth > 300, $"Width={input.ActualWidth}; size={size}; y={location.Y}; height={input.ActualHeight}");
                Assert.True(location.Y + input.ActualHeight <= size.Height, "Chat composer exceeds the window.");
                Assert.True(((ScrollViewer)window.FindName("ChatScrollViewer")).ActualHeight >= 80, $"Chat height={((ScrollViewer)window.FindName("ChatScrollViewer")).ActualHeight}; size={size}");
                var bitmap = new System.Windows.Media.Imaging.RenderTargetBitmap((int)size.Width, (int)size.Height, 96, 96, System.Windows.Media.PixelFormats.Pbgra32);
                bitmap.Render(content);
                string folder = Environment.GetEnvironmentVariable("NNTRAIN_ASR_LAYOUT_DIR") ?? System.IO.Path.GetTempPath();
                System.IO.Directory.CreateDirectory(folder);
                var png = new System.Windows.Media.Imaging.PngBitmapEncoder();
                png.Frames.Add(System.Windows.Media.Imaging.BitmapFrame.Create(bitmap));
                using var file = System.IO.File.Create(System.IO.Path.Combine(folder, $"chat-audio-{size.Width}.png")); png.Save(file);
                var settings = (Expander)window.FindName("AsrSettingsExpander"); settings.IsExpanded = true;
                content.Measure(size); content.Arrange(new Rect(size)); content.UpdateLayout();
                location = input.TranslatePoint(new Point(), content);
                Assert.True(location.Y + input.ActualHeight <= size.Height, "Expanded speech settings hide the chat composer.");
                Assert.True(((ComboBox)window.FindName("AsrModelComboBox")).ActualWidth > 180);
                var expanded = new System.Windows.Media.Imaging.RenderTargetBitmap((int)size.Width, (int)size.Height, 96, 96, System.Windows.Media.PixelFormats.Pbgra32);
                expanded.Render(content); var expandedPng = new System.Windows.Media.Imaging.PngBitmapEncoder();
                expandedPng.Frames.Add(System.Windows.Media.Imaging.BitmapFrame.Create(expanded));
                using var expandedFile = System.IO.File.Create(System.IO.Path.Combine(folder, $"chat-audio-expanded-{size.Width}.png")); expandedPng.Save(expandedFile);
                settings.IsExpanded = false;
            }
        });
    [Fact]
    public void FinalizationRejectsQueuedPartialsAndPreservesSubsequentEdits()
        => OnSta(window =>
        {
            long epoch = (long)Field(window, "_asrEpoch")!;
            Invoke(window, "BeginAudioChatInput");
            Invoke(window, "ShowAudioPartial", epoch, "まだ途中");
            Invoke(window, "CompleteAudio", epoch, "確定した全文", 0L);
            window.Dispatcher.Invoke(() => { }, DispatcherPriority.ApplicationIdle);
            Assert.Equal("確定した全文", Box(window, "MessageBox").Text);
            Box(window, "MessageBox").Text = "ユーザーの編集";
            Invoke(window, "ShowAudioPartial", epoch, "遅れた途中結果");
            window.Dispatcher.Invoke(() => { }, DispatcherPriority.ApplicationIdle);
            Assert.Equal("ユーザーの編集", Box(window, "MessageBox").Text);
        });
    [Fact]
    public void LivePartialUpdatesBothTextBoxesAndCancelClearsCurrentUtterance()
        => OnSta(window =>
        {
            long epoch = (long)Field(window, "_asrEpoch")!;
            Invoke(window, "BeginAudioChatInput");
            Invoke(window, "ShowAudioPartial", epoch, "録音中の日本語");
            window.Dispatcher.Invoke(() => { }, DispatcherPriority.ApplicationIdle);
            Assert.Equal("録音中の日本語", Box(window, "MessageBox").Text);
            Assert.Equal("録音中の日本語", Box(window, "MessageBox").Text);
            Invoke(window, "CancelAudio");
            Invoke(window, "ShowAudioPartial", epoch, "前の録音の遅れた結果");
            window.Dispatcher.Invoke(() => { }, DispatcherPriority.ApplicationIdle);
            Assert.Empty(Box(window, "MessageBox").Text);
            Set(window, "_asrRecognizing", false);
        });
    [Fact]
    public void EditedRecognitionUsesExistingLocalChatTransport()
        => OnSta(window =>
        {
            string model = System.IO.Path.GetTempFileName();
            var handler = new ChatTransport(model);
            var process = System.Diagnostics.Process.GetCurrentProcess();
            try
            {
                Set(window, "_serverProcess", process); // Readiness marker only; never start/stop a process.
                Set(window, "_serverReady", true);
                ((System.Net.Http.HttpClient)Field(window, "_client")!).Dispose();
                Set(window, "_client", new System.Net.Http.HttpClient(handler) { BaseAddress = new Uri("http://localhost/") });
                var models = (ComboBox)window.FindName("ModelComboBox");
                models.ItemsSource = null; models.Items.Clear();
                models.Items.Add(new MainWindow.FileChoice("test", model)); models.SelectedIndex = 0;
                var adapters = (ComboBox)window.FindName("AdapterComboBox");
                adapters.SelectedIndex = 0;
                ((CheckBox)window.FindName("StreamCheckBox")).IsChecked = false;
                ((CheckBox)window.FindName("ThinkingCheckBox")).IsChecked = false;
                Box(window, "MessageBox").Text = "日本語の認識文を編集して送信します。";
                string sentText = Box(window, "MessageBox").Text;
                Invoke(window, "Send_Click", null!, new RoutedEventArgs());
                for (int i = 0; i < 100 && (bool)Field(window, "_busy")!; ++i)
                    window.Dispatcher.Invoke(() => { }, DispatcherPriority.ApplicationIdle);
                Assert.False((bool)Field(window, "_busy")!);
                Assert.NotNull(handler.ChatBody);
                using var body = System.Text.Json.JsonDocument.Parse(handler.ChatBody!);
                Assert.Equal(sentText, body.RootElement.GetProperty("messages")[0].GetProperty("content").GetString());
                Assert.Equal(model, body.RootElement.GetProperty("model").GetString());
                Assert.Empty(Box(window, "MessageBox").Text);
                Assert.Null(Field(window, "_microphone"));
            }
            finally { Set(window, "_serverProcess", null); process.Dispose(); System.IO.File.Delete(model); }
        });

    private sealed class ChatTransport(string model) : System.Net.Http.HttpMessageHandler
    {
        public string? ChatBody;
        protected override async Task<System.Net.Http.HttpResponseMessage> SendAsync(System.Net.Http.HttpRequestMessage request, CancellationToken ct)
        {
            string path = request.RequestUri!.AbsolutePath;
            string json = "{}";
            if (path == "/internal/state")
                json = System.Text.Json.JsonSerializer.Serialize(new { is_loaded = true, model, devices = new[] { 0 } });
            if (path == "/v1/chat/completions")
            {
                ChatBody = await request.Content!.ReadAsStringAsync(ct);
                json = "{\"choices\":[{\"message\":{\"content\":\"test response\"},\"finish_reason\":\"stop\"}]}";
            }
            return new System.Net.Http.HttpResponseMessage(System.Net.HttpStatusCode.OK)
            { Content = new System.Net.Http.StringContent(json, System.Text.Encoding.UTF8, "application/json") };
        }
    }

    private static void Set(MainWindow window, string name, object? value)
        => typeof(MainWindow).GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(window, value);
    [Fact]
    public void EditedTranscriptMovesIntoExistingChatInputWithoutRecording()
        => OnSta(window =>
        {
            Assert.Null(Field(window, "_microphone")); Assert.Null(Field(window, "_serverProcess"));
            Box(window, "MessageBox").Text = "手入力の依頼";
            Box(window, "MessageBox").CaretIndex = Box(window, "MessageBox").Text.Length;
            Invoke(window, "BeginAudioChatInput");
            long epoch = (long)Field(window, "_asrEpoch")!;
            Invoke(window, "ShowAudioPartial", epoch, "録音からの認識文");
            window.Dispatcher.Invoke(() => { }, DispatcherPriority.ApplicationIdle);
            Assert.Equal("手入力の依頼\n録音からの認識文", Box(window, "MessageBox").Text);
            Invoke(window, "CompleteAudio", epoch, "録音からの認識文", 0L);
            window.Dispatcher.Invoke(() => { }, DispatcherPriority.ApplicationIdle);
            Assert.Equal("手入力の依頼\n録音からの認識文", Box(window, "MessageBox").Text);
            Set(window, "_asrRecognizing", false);
            Invoke(window, "UpdateAudioControls");
            Assert.False(Box(window, "MessageBox").IsReadOnly);
        });

    [Fact]
    public void CancelRejectsQueuedPartialAndFinalFromOldUtterance()
        => OnSta(window =>
        {
            Box(window, "MessageBox").Text = "続きの下書き";
            Invoke(window, "BeginAudioChatInput");
            long epoch = (long)Field(window, "_asrEpoch")!;
            Invoke(window, "ShowAudioPartial", epoch, "古い途中結果");
            Invoke(window, "CompleteAudio", epoch, "古い確定結果", 0L);
            Invoke(window, "CancelAudio");
            window.Dispatcher.Invoke(() => { }, DispatcherPriority.ApplicationIdle);
            Assert.Equal("続きの下書き", Box(window, "MessageBox").Text);
            Set(window, "_asrRecognizing", false);
        });

    [Fact]
    public void ActiveRecognitionCannotReplaceChatDraft()
        => OnSta(window =>
        {
            Box(window, "MessageBox").Text = "手入力";
            var pending = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            Task task = (Task)Invoke(window, "RunAudioAsync", new Func<long, CancellationToken, Task>((_, token) =>
            {
                Invoke(window, "BeginAudioChatInput");
                token.Register(() => pending.TrySetCanceled(token));
                return pending.Task;
            }))!;
            bool secondStarted = false;
            Task second = (Task)Invoke(window, "RunAudioAsync", new Func<long, CancellationToken, Task>((_, _) =>
            { secondStarted = true; return Task.CompletedTask; }))!;
            Assert.True(second.IsCompleted); Assert.False(secondStarted);
            Assert.True(Box(window, "MessageBox").IsReadOnly);
            Invoke(window, "CancelAudio");
            while (!task.IsCompleted) window.Dispatcher.Invoke(() => { }, DispatcherPriority.ApplicationIdle);
            Assert.Equal("手入力", Box(window, "MessageBox").Text);
            Assert.False(Box(window, "MessageBox").IsReadOnly);
        });

    private static TextBox Box(MainWindow window, string name) => (TextBox)window.FindName(name);
    private static object? Field(MainWindow window, string name) => typeof(MainWindow).GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(window);
    private static object? Invoke(MainWindow window, string name, params object[] arguments)
        => typeof(MainWindow).GetMethod(name, BindingFlags.Instance | BindingFlags.NonPublic)!.Invoke(window, arguments);
    private static void OnSta(Action<MainWindow> action)
    {
        Exception? error = null;
        var thread = new Thread(() =>
        {
            SynchronizationContext.SetSynchronizationContext(new DispatcherSynchronizationContext(Dispatcher.CurrentDispatcher));
            MainWindow? window = null;
            try { window = new MainWindow(GuiLaunchOptions.Parse([])); action(window); }
            catch (Exception ex) { error = ex; }
            finally
            {
                window?.Close();
                Dispatcher.CurrentDispatcher.Invoke(() => { }, DispatcherPriority.ApplicationIdle);
                Dispatcher.CurrentDispatcher.InvokeShutdown();
            }
        });
        thread.SetApartmentState(ApartmentState.STA); thread.Start();
        Assert.True(thread.Join(TimeSpan.FromSeconds(30)), "WPF test did not finish.");
        if (error is not null) System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(error).Throw();
    }
}
