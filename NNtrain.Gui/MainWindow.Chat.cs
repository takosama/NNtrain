using System.Collections.ObjectModel;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;

namespace NNtrain.Gui;

public partial class MainWindow
{
    private readonly ObservableCollection<ChatBubble> _messages = [];
    private readonly List<ChatTurn> _conversation = [];

    private void Thinking_Changed(object sender, RoutedEventArgs e)
    {
        if (ThinkingCheckBox is not null)
            ThinkingCheckBox.Content = ThinkingCheckBox.IsChecked == true ? "Thinking: On" : "Thinking: Off";
    }

    private void Stream_Changed(object sender, RoutedEventArgs e)
    {
        if (StreamCheckBox is not null)
            StreamCheckBox.Content = StreamCheckBox.IsChecked == true ? "Stream: On" : "Stream: Off";
    }

    private async void Send_Click(object sender, RoutedEventArgs e) => await SendMessageAsync();

    private void MessageBox_PreviewKeyDown(object sender, KeyEventArgs e)
    {
        if (e.Key == Key.Enter && Keyboard.Modifiers == ModifierKeys.Control)
        {
            e.Handled = true;
            _ = SendMessageAsync();
        }
    }

    private void CopyContent_Click(object sender, RoutedEventArgs e)
    {
        if (sender is Button { DataContext: ChatBubble bubble })
            CopyBubbleText(bubble.Content, bubble.CopyContentLabel);
    }

    private void CopyThinking_Click(object sender, RoutedEventArgs e)
    {
        if (sender is Button { DataContext: ChatBubble bubble })
            CopyBubbleText(bubble.ThinkingContent, "思考をコピー");
    }

    private void CopyBubbleText(string text, string label)
    {
        if (string.IsNullOrEmpty(text)) return;
        try
        {
            Clipboard.SetDataObject(text, true);
            SetStatus($"{label}しました。");
        }
        catch (ExternalException ex)
        {
            SetStatus($"クリップボードにコピーできませんでした: {ex.Message}");
        }
    }

    private async Task SendMessageAsync()
    {
        if (_asrActive) return;
        string message = MessageBox.Text.Trim();
        ChatImage? image = _pendingImage;
        if (message.Length == 0 && image is not null) message = "この画像について説明してください。";
        if (message.Length == 0 ||
            !TrySelectedFiles(out string? modelPath, out string? adapterPath, out GpuChoice? gpu)) return;
        if ((image is not null || _conversation.Any(turn => turn.Image is not null)) && SelectedMmprojPath is null)
        {
            SetStatus("画像を使うにはモデルに対応する mmproj GGUF を選択してください。");
            return;
        }
        if (!GenerationInput.TryParse(MaxTokensBox.Text, TemperatureBox.Text, TopPBox.Text,
            TopKBox.Text, out GenerationInput? input, out string? error))
        {
            SetStatus(error!);
            return;
        }
        int maxNewTokens = input!.MaxNewTokens;
        GenerationSampling sampling = input.Sampling;
        await ExecuteAsync(async ct =>
        {
            await WaitForPendingImagePreparationAsync(image, ct);
            await EnsureLoadedAsync(modelPath!, adapterPath, gpu!.DeviceIndices, ct, gpu.IsGemmaOpenClAuto);
            ct.ThrowIfCancellationRequested();
            MessageBox.Clear();
            _pendingImage = null;
            UpdateImageAttachment();
            int conversationCountBeforeMessage = _conversation.Count;
            _conversation.Add(new ChatTurn("user", message, Image: image));
            var question = new ChatBubble("あなた", image is null ? message : $"[画像: {image.Name}]\n{message}");
            _messages.Add(question);
            bool thinking = !_selectedModelIsGemma && ThinkingCheckBox.IsChecked == true;
            bool stream = StreamCheckBox.IsChecked == true;
            var answer = new ChatBubble("アシスタント", "", thinking);
            _messages.Add(answer);
            ScrollToEnd();
            SetStatus(thinking ? "思考中…" : "生成中…");
            var generation = new ChatGenerationSession(thinking);
            try
            {
                await generation.RunAsync((requestedThinking, onText, token) =>
                    GenerateViaServerAsync(_conversation.ToArray(), modelPath!, adapterPath,
                        gpu!.DeviceIndices, gpu.IsGemmaOpenClAuto, requestedThinking,
                        stream, maxNewTokens, sampling, onText, token), snapshot =>
                    {
                        answer.Update(snapshot);
                        if (snapshot.ThinkingInProgress) SetStatus("思考中…");
                        else if (snapshot.AnswerText.Length != 0) SetStatus("回答を生成中…");
                        ScrollToEnd();
                    }, () => SetStatus("思考だけで終了したため、回答を生成しています…"), ct);
                SetStatus(generation.Snapshot.AnswerText.Length == 0
                    ? "回答に到達できませんでした。入力を戻しました。"
                    : generation.RetriedWithoutThinking ? "回答を再生成しました。" : "生成完了。");
            }
            finally
            {
                answer.Update(generation.Snapshot);
                answer.FinishThinking(generation.StopReason, interrupted: !generation.Completed);
                if (answer.Content.Length == 0)
                {
                    // An unanswered request must not leave a dangling user turn in the
                    // next prompt or a duplicate failed turn on screen. Restore
                    // the input so the user can retry it.
                    _conversation.RemoveRange(conversationCountBeforeMessage,
                        _conversation.Count - conversationCountBeforeMessage);
                    MessageBox.Text = message;
                    _pendingImage = image;
                    UpdateImageAttachment();
                    _messages.Remove(answer);
                    _messages.Remove(question);
                }
                else
                {
                    // Keep the exact assistant prefix used by the preceding
                    // prompt while retaining only the visible answer in history.
                    // This also keeps interrupted partial answers paired with
                    // their user turn, so the next prompt remains well formed.
                    _conversation.Add(new ChatTurn("assistant", answer.Content, generation.AssistantPrefix));
                }
                ScrollToEnd();
            }
        });
    }
}
