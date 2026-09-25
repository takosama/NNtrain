using NNtrain;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class TransformerStreamingGenerationTests
{
    [Fact]
    public void TransformerStreamsBeforeRestoringTrainingStateAndKeepsExecutionSessionAlive()
    {
        using ExecutionSession session = ProductionTrainingSessionFactory.CreateExecutionSession(
            TensorPrecisionMode.Float32, TensorDevice.Cpu, [0]);
        using IDisposable executionScope = session.Enter();
        BpeTokenizer tokenizer = BpeTokenizer.Train(["a"], BpeTokenizer.BaseVocabularySize);
        int nextToken = Assert.Single(tokenizer.Encode("a"));
        var model = new GptRinWikiJp(tokenizer.VocabularySize, 32, 8, 2, 16, 1,
            new Random(1), dropout: .1f);
        ModuleState state = model.state_dict();
        ModuleParameterState headBias = Assert.Single(state.Parameters, parameter =>
            parameter.Name == "B" && parameter.Shape.SequenceEqual([tokenizer.VocabularySize]));
        Array.Clear(headBias.Values);
        headBias.Values[nextToken] = 100;
        model.load_state_dict(state);
        Assert.True(model.IsTraining);
        using var output = new ObservingWriter(model, session);

        GenerateCommand.StreamGeneration(model, tokenizer, "a", 4, 0f, 1,
            new Random(2), output);

        Assert.Equal("aaaa", output.ToString());
        Assert.Equal(4, output.TokenWrites);
        Assert.True(model.IsTraining);
        Assert.Same(session, ExecutionSession.Current);
        Assert.False(session.IsDisposed);
    }

    private sealed class ObservingWriter(GptRinWikiJp model, ExecutionSession session) : StringWriter
    {
        internal int TokenWrites { get; private set; }

        public override void Write(string? value)
        {
            if (!string.IsNullOrEmpty(value))
            {
                // A per-token generate(1) loop or the base deferred callback
                // would write only after GenerateTokenIds restored Train().
                Assert.False(model.IsTraining);
                Assert.Same(session, ExecutionSession.Current);
                Assert.False(session.IsDisposed);
                TokenWrites++;
            }
            base.Write(value);
        }
    }
}
