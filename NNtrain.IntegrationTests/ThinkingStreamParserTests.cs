using NNtrain.Gui;
using Xunit;

public sealed class ThinkingStreamParserTests
{
    [Fact]
    public void RepeatedClosingMarkerPreservesEarlierAnswerDuringStreaming()
    {
        var parser = new ThinkingStreamParser(requestedThinking: true);

        ThinkingStreamSnapshot first = parser.Append("<think>理由</think>詳しい回答がここにあります。");
        Assert.Equal("理由", first.ThinkingText);
        Assert.Equal("詳しい回答がここにあります。", first.AnswerText);

        ThinkingStreamSnapshot second = parser.Append("</think>\n- 一つ目\n- 二つ目");
        Assert.Equal("詳しい回答がここにあります。\n- 一つ目\n- 二つ目", second.AnswerText);
        Assert.Equal(second.AnswerText, parser.Complete(endedByEos: true).AnswerText);
    }

    [Fact]
    public void EmptyTrailingClosingMarkerDoesNotEraseAnswer()
    {
        var parser = new ThinkingStreamParser(requestedThinking: true);
        parser.Append("<think>理由</think>長い回答本文</thi");

        ThinkingStreamSnapshot completed = parser.Append("nk>");
        Assert.Equal("長い回答本文", completed.AnswerText);
        Assert.Equal("長い回答本文", parser.Complete(endedByEos: true).AnswerText);
    }

    [Fact]
    public void AdjacentAnswerSpansReceiveLineBoundary()
    {
        var parser = new ThinkingStreamParser(requestedThinking: false);
        parser.Append("</think>最初の回答</think>次の回答<|im_end|>無視する文字列");

        ThinkingStreamSnapshot completed = parser.Complete(endedByEos: true);
        Assert.Equal("最初の回答\n次の回答", completed.AnswerText);
    }
}
