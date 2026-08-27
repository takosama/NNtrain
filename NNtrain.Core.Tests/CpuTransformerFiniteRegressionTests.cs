using NNtrain;
using Xunit;

public sealed class CpuTransformerFiniteRegressionTests
{
    [Fact]
    public void WidthFourWikiTransformerRemainsFiniteAcrossTwoCpuUpdates()
    {
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cpu;
            const string document =
                "日本語の小さな学習文書です。境界テストを行います。";
            BpeTokenizer tokenizer = BpeTokenizer.Train(
                [document],
                vocabularySize: 300,
                maxTrainingBytes: 10_000);
            int[] documentTokens = tokenizer.Encode(document)
                .Take(8)
                .ToArray();
            int[] tokens = [BpeTokenizer.BosTokenId, .. documentTokens];
            var model = new GptRinWikiJp(
                vocabularySize: 300,
                contextLength: 4,
                dModel: 4,
                numHeads: 1,
                dHidden: 8,
                numLayers: 1,
                rng: new Random(1234),
                dropout: 0f,
                dtype: TensorDType.Float32);
            Parameter[] parameters = model.Parameters().ToArray();
            var optimizer = new AdamW(
                parameters,
                new AdamWOptions { LearningRate = 0.001f });

            for (int step = 0; step < 2; step++)
            {
                int[] input = tokens.Skip(step * 4).Take(4).ToArray();
                int[] targets = tokens.Skip(step * 4 + 1).Take(4).ToArray();

                optimizer.ZeroGrad();
                Tensor logits = model.Forward(
                    input,
                    batchSize: 1,
                    sequenceLength: 4);
                Assert.All(
                    logits.Data,
                    value => Assert.True(float.IsFinite(value)));

                Tensor loss = logits.CrossEntropyWithLogits(targets);
                Assert.True(float.IsFinite(loss.Data[0]));
                loss.Backward();

                Assert.All(
                    parameters.SelectMany(parameter => parameter.T.Grad),
                    value => Assert.True(float.IsFinite(value)));
                float gradientNorm = nn.utils.clip_grad_norm_(
                    parameters,
                    max_norm: 1f);
                Assert.True(float.IsFinite(gradientNorm));

                optimizer.Step();
                Assert.All(
                    parameters.SelectMany(parameter => parameter.T.Data),
                    value => Assert.True(float.IsFinite(value)));
            }
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
        }
    }
}
