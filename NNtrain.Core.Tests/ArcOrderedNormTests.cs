using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcOrderedNormTests
{
    private const string ForwardKernel = "norm_row_sg16_w64_candidate";
    private const string BackwardKernel = "norm_dx_row_sg16_w64_candidate";

    [Theory]
    [InlineData(TensorPrecisionMode.Float32, false)]
    [InlineData(TensorPrecisionMode.Mix16_32, false)]
    [InlineData(TensorPrecisionMode.Mix8_32, false)]
    [InlineData(TensorPrecisionMode.Float32, true)]
    [InlineData(TensorPrecisionMode.Mix16_32, true)]
    [InlineData(TensorPrecisionMode.Mix8_32, true)]
    public void TwoTrainingStepsPreserveLossGradientsUpdatesAndResidency(TensorPrecisionMode precision, bool packed)
    {
        RequireSg16();
        StepSnapshot[] Run(bool ordered)
        {
            const int batch = 2, sequence = 64, width = 64, vocabulary = 67;
            using var execution = Tensor.BeginArcExecution(precision: precision, options: new() {
                OrderedTiledNorm = ordered, BlockResidualNorm = false,
                // Also compare the combined SG + packed residual candidate
                // against the original materialized residual + serial norm.
                PackedNormInput = ordered && packed,
            });
            ArcExecutionLane lane = Tensor.ArcLane;
            var model = new GptRinWikiJp(vocabulary, sequence, width, 2, 128, 2, new Random(57),
                dropout: .1f, tieWordEmbeddings: true);
            model.to(precision, 32);
            Parameter[] parameters = model.parameters().ToArray();
            IOptimizer optimizer = new AdamW(parameters, new AdamWOptions { LearningRate = .0003f, WeightDecay = .01f });
            var steps = new StepSnapshot[2];
            long retained = -1;
            try
            {
                for (int step = 0; step < steps.Length; step++)
                {
                    optimizer.zero_grad();
                    int[] tokens = Enumerable.Range(0, batch * sequence).Select(i => (i * 7 + step * 11) % vocabulary).ToArray();
                    int[] targets = tokens.Select((token, i) => i % 17 == 0 ? -1 : (token + 1) % vocabulary).ToArray();
                    Tensor loss = model.forward_loss(tokens, targets, batch, sequence);
                    float value = loss.item();
                    Assert.True(float.IsFinite(value));
                    long downloads = lane.D2HBytes;
                    loss.BackwardAndRelease();
                    lane.Synchronize();
                    Assert.Equal(downloads, lane.D2HBytes);
                    float norm = nn.utils.clip_grad_norm_(parameters, 1f);
                    Assert.True(float.IsFinite(norm));
                    optimizer.step();
                    lane.Synchronize();
                    Assert.Equal(0, lane.RetiredBytes);
                    Assert.InRange(lane.CachedBytes, 0, lane.Options.BufferPoolBytes);
                    if (step == 0) retained = lane.AllocatedBytes;
                    else Assert.Equal(retained, lane.AllocatedBytes);
                    steps[step] = new(value, norm, parameters.Select(p => p.T.Grad.ToArray()).ToArray(),
                        parameters.Select(p => p.T.CaptureData(preferMaster: true)).ToArray());
                }
                Assert.Equal(ordered, lane.KernelTimings.ContainsKey(ForwardKernel));
                Assert.Equal(ordered, lane.KernelTimings.ContainsKey(BackwardKernel));
                Assert.Equal(!ordered, lane.KernelTimings.ContainsKey("norm"));
                Assert.Equal(!ordered, lane.KernelTimings.ContainsKey("norm_dx_set"));
                Assert.Equal(ordered && packed, lane.KernelTimings.ContainsKey("norm_packed_residual_input"));
                Assert.All(parameters, p => Assert.Equal(precision.ToStorageDType(), p.T.DType));
                return steps;
            }
            finally
            {
                if (optimizer is IDisposable disposable) disposable.Dispose();
            }
        }

        var expected = Run(false); var actual = Run(true);
        for (int step = 0; step < expected.Length; step++)
        {
            // Existing transformer tolerance: never widen it for a new route.
            Assert.InRange(MathF.Abs(expected[step].Loss - actual[step].Loss), 0, 1e-4f);
            Assert.InRange(MathF.Abs(expected[step].Norm - actual[step].Norm), 0, 1e-4f);
            AssertClose(expected[step].Gradients, actual[step].Gradients);
            AssertClose(expected[step].Masters, actual[step].Masters);
        }
    }

    [Theory]
    [InlineData(128, 64, true, true)]
    [InlineData(137, 65, true, true)]
    [InlineData(129, 2048, true, true)]
    [InlineData(127, 64, true, false)]
    [InlineData(128, 63, true, false)]
    [InlineData(128, 2049, true, false)]
    [InlineData(128, 64, false, false)]
    public void ShapeAndCapabilityGateRetainsExactForwardAndAccumulatedGradient(
        int rows, int width, bool xmx, bool selected)
    {
        RequireSg16();
        (float[][] Outputs, float[][] Gradients) Run(bool ordered)
        {
            using var execution = Tensor.BeginArcExecution(options: new() {
                OrderedTiledNorm = ordered, BlockResidualNorm = false, XmxMatrices = xmx,
            });
            ArcExecutionLane lane = Tensor.ArcLane;
            Tensor Create(int length, int[] shape, float offset) => new(Enumerable.Range(0, length)
                .Select(i => MathF.Sin(i * .137f) * .13f + offset).ToArray(), shape);
            var input = Create(rows * width, [rows, width], 0f);
            var gamma = Create(width, [width], 1f);
            var beta = Create(width, [width], .01f);
            var outputs = new float[2][];
            for (int repeat = 0; repeat < outputs.Length; repeat++)
            {
                Tensor output = input.LayerNormLastDim(gamma, beta);
                outputs[repeat] = output.Data.ToArray();
                long downloads = lane.D2HBytes;
                output.BackwardAndRelease(Enumerable.Range(0, rows * width)
                    .Select(i => MathF.Cos(i * .017f + repeat) * .01f).ToArray());
                lane.Synchronize();
                Assert.Equal(downloads, lane.D2HBytes);
            }
            Assert.Equal(ordered && selected, lane.KernelTimings.ContainsKey(ForwardKernel));
            Assert.Equal(ordered && selected, lane.KernelTimings.ContainsKey(BackwardKernel));
            return (outputs, new[] { input, gamma, beta }.Select(t => t.Grad.ToArray()).ToArray());
        }
        var expected = Run(false); var actual = Run(true);
        for (int i = 0; i < expected.Outputs.Length; i++) Assert.Equal(expected.Outputs[i], actual.Outputs[i]);
        for (int i = 0; i < expected.Gradients.Length; i++) Assert.Equal(expected.Gradients[i], actual.Gradients[i]);
    }

    private static void RequireSg16()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        ArcDeviceInfo device = ArcDevices.Enumerate()[0];
        Assert.SkipWhen(!device.SupportsXmx || device.MinimumSubgroupSize != 16, "Intel XMX with SG16 is required.");
    }

    private static void AssertClose(float[][] expected, float[][] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int p = 0; p < expected.Length; p++)
        {
            Assert.Equal(expected[p].Length, actual[p].Length);
            for (int i = 0; i < expected[p].Length; i++)
                Assert.InRange(MathF.Abs(expected[p][i] - actual[p][i]), 0, 1e-4f);
        }
    }

    private sealed record StepSnapshot(float Loss, float Norm, float[][] Gradients, float[][] Masters);
}
