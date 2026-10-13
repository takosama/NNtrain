using System.ComponentModel;
using System.Windows;

namespace NNtrain.Gui;

public partial class MainWindow
{
    public sealed class ChatBubble : INotifyPropertyChanged
    {
        private string _content;
        private string _thinkingContent = "";
        private bool _hasThinking;
        private bool _thinkingInProgress;
        private bool _thinkingExpanded;
        private bool _generationFinished;
        private bool _generationInterrupted;
        private GenerationStopReason? _stopReason;

        public ChatBubble(string author, string content, bool hasThinking = false)
        {
            Author = author;
            _content = content;
            _hasThinking = hasThinking;
            _thinkingInProgress = hasThinking;
        }

        public string Author { get; }
        public string CopyContentLabel => Author == "あなた" ? "質問をコピー" : "回答をコピー";
        public bool CanCopyContent => _content.Length != 0;
        public bool CanCopyThinking => !string.IsNullOrWhiteSpace(_thinkingContent);
        public bool HasThinking => _hasThinking;
        public Visibility ThinkingVisibility => _hasThinking && (_thinkingInProgress || _thinkingContent.Length != 0)
            ? Visibility.Visible : Visibility.Collapsed;
        public string ThinkingHeader => _thinkingInProgress ? "思考中…（クリックで表示）" : "思考の内容";
        public string DisplayContent
        {
            get
            {
                if (_content.Length != 0)
                {
                    if (_generationInterrupted)
                        return _content + "\n\n（生成が中断されました。ここまでの回答は会話に残しています。）";
                    return _stopReason switch
                    {
                        GenerationStopReason.MaximumTokens => _content + "\n\n（最大生成数に達しました。）",
                        GenerationStopReason.ContextLimit => _content + "\n\n（会話がモデルの文脈上限に達しました。）",
                        _ => _content
                    };
                }
                if (_generationFinished)
                {
                    if (_hasThinking && !string.IsNullOrWhiteSpace(_thinkingContent))
                        return _stopReason switch
                        {
                            GenerationStopReason.MaximumTokens => "思考中に最大生成数へ達しました。最大生成数を増やして再実行してください。",
                            GenerationStopReason.ContextLimit => "会話がモデルの文脈上限に達しました。会話を短くして再実行してください。",
                            GenerationStopReason.EndOfMessage => "思考の終了タグがないままモデルが終了しました。思考の内容を確認してください。",
                            _ => "生成が中断されました。思考の内容を確認してください。"
                        };
                    return _stopReason switch
                    {
                        GenerationStopReason.EndOfMessage => "モデルは回答を出さずに終了しました。",
                        GenerationStopReason.MaximumTokens => "最大生成数に達しましたが、表示できる回答がありません。",
                        GenerationStopReason.ContextLimit => "会話がモデルの文脈上限に達しました。",
                        _ => ""
                    };
                }
                if (_thinkingInProgress) return "";
                return _hasThinking ? "回答を生成中…" : "生成中…";
            }
        }

        public string Content
        {
            get => _content;
            set
            {
                if (_content == value) return;
                _content = value;
                Notify(nameof(Content));
                Notify(nameof(DisplayContent));
                Notify(nameof(CanCopyContent));
            }
        }

        public string ThinkingContent
        {
            get => _thinkingContent;
            set
            {
                if (_thinkingContent == value) return;
                _thinkingContent = value;
                Notify(nameof(ThinkingContent));
                Notify(nameof(ThinkingVisibility));
                Notify(nameof(DisplayContent));
                Notify(nameof(CanCopyThinking));
            }
        }

        public bool ThinkingExpanded
        {
            get => _thinkingExpanded;
            set
            {
                if (_thinkingExpanded == value) return;
                _thinkingExpanded = value;
                Notify(nameof(ThinkingExpanded));
            }
        }

        internal void Update(ThinkingStreamSnapshot snapshot)
        {
            if (_hasThinking != snapshot.HasThinking)
            {
                _hasThinking = snapshot.HasThinking;
                Notify(nameof(HasThinking));
                Notify(nameof(ThinkingVisibility));
            }
            ThinkingContent = snapshot.ThinkingText;
            Content = snapshot.AnswerText;
            SetThinkingInProgress(snapshot.ThinkingInProgress);
        }

        internal void FinishThinking(GenerationStopReason? stopReason = null, bool interrupted = false)
        {
            _generationFinished = true;
            _generationInterrupted = interrupted;
            _stopReason = stopReason;
            SetThinkingInProgress(false);
            ThinkingExpanded = false;
            Notify(nameof(DisplayContent));
        }

        private void SetThinkingInProgress(bool value)
        {
            if (_thinkingInProgress == value) return;
            _thinkingInProgress = value;
            Notify(nameof(ThinkingHeader));
            Notify(nameof(ThinkingVisibility));
            Notify(nameof(DisplayContent));
        }

        private void Notify(string name) => PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
        public event PropertyChangedEventHandler? PropertyChanged;
    }
}
