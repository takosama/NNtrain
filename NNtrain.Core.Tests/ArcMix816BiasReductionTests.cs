using System.Reflection;
using NNtrain;
using NNtrain.Arc;
using Xunit;
using static NNtrain.Arc.ArcExecutionLane;

public sealed class ArcMix816BiasReductionTests
{
    private static readonly MethodInfo BiasGradient = typeof(Tensor).GetMethod(
        "ArcBiasGradient", BindingFlags.NonPublic | BindingFlags.Static)
        ?? throw new MissingMethodException(typeof(Tensor).FullName, "ArcBiasGradient");

    [Theory]
    [InlineData(769, 1536, false)] // QKV width, partial final row tile.
    [InlineData(521, 512, true)]   // Projection width, gate and partial row tile.
    [InlineData(512, 11500, false)] // Loss-head width, partial 32-column tile.
    public void BiasOnlyBf16ReductionIsBitExactAcrossTwoAccumulations(
        int rows, int width, bool gated)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        var data = Inputs(rows, width, gated);
        var baseline = Run(TensorPrecisionMode.Mix8_16, false, data, rows, width);
        var candidate = Run(TensorPrecisionMode.Mix8_16, true, data, rows, width);

        Assert.Equal(baseline.Bits, candidate.Bits);
        Assert.Contains("gradient_rows", baseline.Kernels);
        Assert.DoesNotContain("gradient_rows_bias_bf16", baseline.Kernels);
        Assert.Contains("gradient_rows_bias_bf16", candidate.Kernels);
        Assert.DoesNotContain("gradient_rows", candidate.Kernels);
        Assert.Contains("gradient_rows_finish", candidate.Kernels);
    }

    [Fact]
    public void BiasOnlyBf16OptionKeepsMix8_32OnTheExistingReduction()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        const int rows = 513, width = 511;
        var data = Inputs(rows, width, true);
        var baseline = Run(TensorPrecisionMode.Mix8_32, false, data, rows, width);
        var optionEnabled = Run(TensorPrecisionMode.Mix8_32, true, data, rows, width);

        Assert.Equal(baseline.Bits, optionEnabled.Bits);
        Assert.Contains("gradient_rows", optionEnabled.Kernels);
        Assert.DoesNotContain("gradient_rows_bias_bf16", optionEnabled.Kernels);
    }

    private static (float[] First, float[] Second, float[] Initial, float[]? Gate)
        Inputs(int rows, int width, bool gated)
    {
        int length = checked(rows * width);
        float[] first = new float[length], second = new float[length];
        float[]? gate = gated ? new float[length] : null;
        for (int i = 0; i < length; i++)
        {
            first[i] = ((i * 17) % 101 - 50) * .000731f;
            second[i] = ((i * 29) % 89 - 44) * .000415f;
            if (gate is not null) gate[i] = i % 5 == 0 ? -.25f : .25f;
        }
        float[] initial = new float[width];
        for (int col = 0; col < width; col++)
            initial[col] = (col % 13 - 6) * .0011f;
        return (first, second, initial, gate);
    }

    private static (int[] Bits, string[] Kernels) Run(TensorPrecisionMode precision,
        bool biasOnly, (float[] First, float[] Second, float[] Initial, float[]? Gate) data,
        int rows, int width)
    {
        using var scope = Tensor.BeginArcExecution(precision: precision,
            options: new ArcExecutionOptions
            {
                ParallelReductions = true,
                Mix8_16BiasOnlyGradientReduction = biasOnly,
            });
        var lane = Tensor.ArcLane;
        using var first = lane.Upload(data.First);
        using var second = lane.Upload(data.Second);
        using var db = lane.Upload(data.Initial);
        using ArcBuffer? gate = data.Gate is null ? null : lane.Upload(data.Gate);

        foreach (ArcBuffer dy in new[] { first, second })
            BiasGradient.Invoke(null, [dy, gate, db, rows, width, true]);

        lane.Synchronize();
        lane.CheckNumericStatus();
        float[] values = new float[width];
        lane.Read(db, values);
        return (values.Select(BitConverter.SingleToInt32Bits).ToArray(),
            lane.KernelTimings.Keys.ToArray());
    }
}
