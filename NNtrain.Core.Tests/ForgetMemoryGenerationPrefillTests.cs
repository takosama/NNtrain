using NNtrain;
using Xunit;

public sealed class ForgetMemoryGenerationPrefillTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void ChunkedPromptPreservesFullPrefixAndGreedyContinuation(int variant)
    {
        TensorDevice previous = Tensor.ExecutionDevice;
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cpu;
            ForgetMemoryV2Gpt model = variant switch
            {
                0 => new ForgetMemoryV2Gpt(64, 16, 16, 32, 2, 4, 4, random: new Random(173), dtype: TensorDType.Float32),
                1 => new ForgetMemoryV3Gpt(64, 16, 16, 32, 2, 4, 4, random: new Random(173), dtype: TensorDType.Float32),
                _ => new ForgetMemoryDRNGpt(64, 16, 16, 32, 2, 4, 4, random: new Random(173), dtype: TensorDType.Float32),
            };
            int[] prompt = Enumerable.Range(0, 257).Select(i => i % 63 + 1).ToArray();
            var expected = new List<int>(prompt);
            using (var state = model.CreateRecurrentState())
            {
                Tensor logits = model.AdvanceToLastLogits(prompt, state);
                for (int i = 0; i < 3; i++)
                {
                    int next = Enumerable.Range(0, 64).MaxBy(j => logits.Data[j]);
                    expected.Add(next);
                    if (i < 2) logits = model.AdvanceToLastLogits([next], state);
                }
                Assert.Equal(prompt.Length + 2, state.TokensSeen);
            }
            var streamed = new List<int>();
            int[] actual = model.GenerateTokenIds(prompt, 3, 0f, 1, null, new Random(19), streamed.Add);
            Assert.Equal(expected, actual);
            Assert.Equal(actual[^3..], streamed);
            Assert.True(model.IsTraining);
            Assert.True(AutogradContext.IsRecordingEnabled);
            Assert.Equal(prompt, model.GenerateTokenIds(prompt, 0));
            Assert.Throws<InvalidOperationException>(() => model.GenerateTokenIds(
                prompt, 2, 0f, 1, null, new Random(19), _ => throw new InvalidOperationException("writer failed")));
            Assert.True(model.IsTraining);
            Assert.True(AutogradContext.IsRecordingEnabled);
        }
        finally { Tensor.ExecutionDevice = previous; }
    }
}
