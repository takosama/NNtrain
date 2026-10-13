using NNtrain.Gui;
using Xunit;

namespace NNtrain.IntegrationTests;

public sealed class ChatGenerationSessionRefactorTests
{
    [Fact]
    public async Task ThinkingAnswerFinishesWithoutRetryAndKeepsTheOriginalAssistantPrefix()
    {
        var session = new ChatGenerationSession(true);
        int calls = 0;
        await session.RunAsync((thinking, onText, _) =>
        {
            Assert.True(thinking);
            calls++;
            onText("考える</think>回答");
            return Task.FromResult(new LocalGeneration("考える</think>回答", GenerationStopReason.EndOfMessage));
        }, _ => { }, () => throw new InvalidOperationException("Unexpected retry"), TestContext.Current.CancellationToken);
        Assert.Equal(1, calls);
        Assert.True(session.Completed);
        Assert.False(session.RetriedWithoutThinking);
        Assert.Equal("考える", session.Snapshot.ThinkingText);
        Assert.Equal("回答", session.Snapshot.AnswerText);
        Assert.Equal("<think>\n</think>\n", session.AssistantPrefix);
    }

    [Theory]
    [InlineData(GenerationStopReason.EndOfMessage)]
    [InlineData(GenerationStopReason.MaximumTokens)]
    public async Task ThinkingOnlyCompletionRetriesOnceWithoutThinkingAndRetainsCollapsedReasoning(GenerationStopReason stop)
    {
        var session = new ChatGenerationSession(true);
        var requestedThinking = new List<bool>();
        int retries = 0;
        await session.RunAsync((thinking, onText, _) =>
        {
            requestedThinking.Add(thinking);
            string raw = thinking ? "<think>第一の思考" : "直接回答";
            onText(raw);
            return Task.FromResult(new LocalGeneration(raw, thinking ? stop : GenerationStopReason.EndOfMessage));
        }, _ => { }, () => retries++, TestContext.Current.CancellationToken);
        Assert.Equal(new[] { true, false }, requestedThinking);
        Assert.Equal(1, retries);
        Assert.True(session.RetriedWithoutThinking);
        Assert.True(session.Completed);
        Assert.Equal("第一の思考", session.Snapshot.ThinkingText);
        Assert.Equal("直接回答", session.Snapshot.AnswerText);
        Assert.False(session.Snapshot.ThinkingInProgress);
        Assert.Equal("<think>\n\n</think>\n\n", session.AssistantPrefix);
    }

    [Fact]
    public async Task ContextLimitDoesNotRetryAThinkingOnlyCompletion()
    {
        var session = new ChatGenerationSession(true);
        await session.RunAsync((_, _, _) => Task.FromResult(new LocalGeneration("思考", GenerationStopReason.ContextLimit)),
            _ => { }, () => throw new InvalidOperationException("Unexpected retry"), TestContext.Current.CancellationToken);
        Assert.False(session.RetriedWithoutThinking);
        Assert.Equal(GenerationStopReason.ContextLimit, session.StopReason);
        Assert.Empty(session.Snapshot.AnswerText);
    }

    [Fact]
    public async Task CancellationDuringDirectRetryKeepsItsPartialAnswerAndDoesNotReuseTheFirstStopReason()
    {
        var session = new ChatGenerationSession(true);
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => session.RunAsync((thinking, onText, _) =>
        {
            if (thinking) return Task.FromResult(new LocalGeneration("最初の思考", GenerationStopReason.MaximumTokens));
            onText("途中の回答<|im_");
            throw new OperationCanceledException();
        }, _ => { }, () => { }, TestContext.Current.CancellationToken));
        Assert.False(session.Completed);
        Assert.True(session.RetriedWithoutThinking);
        Assert.Null(session.StopReason);
        Assert.Equal("最初の思考", session.Snapshot.ThinkingText);
        Assert.Equal("途中の回答", session.Snapshot.AnswerText);
        Assert.False(session.Snapshot.ThinkingInProgress);
    }

    [Fact]
    public async Task CancellationBeforeFinalResponseSettlesStreamedAnswerAndIgnoresLateCallbacks()
    {
        var session = new ChatGenerationSession(false);
        Action<string>? lateCallback = null;
        int updates = 0;
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => session.RunAsync((_, onText, _) =>
        {
            lateCallback = onText;
            onText("途中まで");
            throw new OperationCanceledException();
        }, _ => updates++, () => { }, TestContext.Current.CancellationToken));
        Assert.Equal("途中まで", session.Snapshot.AnswerText);
        Assert.False(session.Completed);
        Assert.Equal(1, updates);
        lateCallback!("遅れた文字");
        Assert.Equal(1, updates);
        Assert.Equal("途中まで", session.Snapshot.AnswerText);
    }

    [Fact]
    public async Task RetryCombinesUnexpectedDirectThinkingWithFirstReasoning()
    {
        var session = new ChatGenerationSession(true);
        await session.RunAsync((thinking, _, _) => Task.FromResult(thinking
                ? new LocalGeneration("<think>最初", GenerationStopReason.EndOfMessage)
                : new LocalGeneration("<think>追加</think>回答", GenerationStopReason.EndOfMessage)),
            _ => { }, () => { }, TestContext.Current.CancellationToken);
        Assert.Equal("最初\n\n追加", session.Snapshot.ThinkingText);
        Assert.Equal("回答", session.Snapshot.AnswerText);
    }

    [Theory]
    [InlineData("0", "0.7", "0.9", "40", "最大生成数")]
    [InlineData("8193", "0.7", "0.9", "40", "最大生成数")]
    [InlineData("512", "NaN", "0.9", "40", "temperature")]
    [InlineData("512", "Infinity", "0.9", "40", "temperature")]
    [InlineData("512", "2.01", "0.9", "40", "temperature")]
    [InlineData("512", "0.7", "0", "40", "top-p")]
    [InlineData("512", "0.7", "NaN", "40", "top-p")]
    [InlineData("512", "0.7", "0.9", "257", "top-k")]
    public void InvalidGenerationInputsKeepTheirSpecificValidationMessage(string tokens, string temperature,
        string topP, string topK, string field)
    {
        Assert.False(GenerationInput.TryParse(tokens, temperature, topP, topK, out GenerationInput? input, out string? error));
        Assert.Null(input);
        Assert.StartsWith(field, error);
    }

    [Fact]
    public void GenerationInputsAcceptExistingBoundaryValues()
    {
        Assert.True(GenerationInput.TryParse("8192", "0", "1", "256", out GenerationInput? input, out string? error));
        Assert.Null(error);
        Assert.Equal(8192, input!.MaxNewTokens);
        Assert.Equal(new GenerationSampling(0, 1, 256), input.Sampling);
    }
}
