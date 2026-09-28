using NNtrain.Arc;
using Xunit;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Core.Tests;

public sealed class Qwen35Iq2XmxTrainingTests
{
    [Theory]
    [InlineData(5120)]
    [InlineData(257)]
    public void ProductionWideFp16TileMatchesPriorFp16TileWithRowTail(int outputWidth)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc is required.");
        const int rows = 130, inputWidth = 5120;
        var random = new Random(821);
        var packed = new byte[outputWidth * (inputWidth / 256) * 82];
        random.NextBytes(packed);
        for (int block = 0; block < packed.Length / 82; ++block)
        {
            ushort scale = BitConverter.HalfToUInt16Bits((Half).01f);
            packed[block * 82] = (byte)scale;
            packed[block * 82 + 1] = (byte)(scale >> 8);
        }
        float[] input = Enumerable.Range(0, rows * inputWidth)
            .Select(_ => (random.NextSingle() * 2 - 1) * .25f).ToArray();
        using var lane = new ArcExecutionLane(0, new()
        {
            Qwen35InferenceKernelsOnly = true, Qwen35TrainingKernels = true
        });
        using ArcBuffer x = lane.Upload(input);
        using ArcBuffer weight = lane.UploadRaw(packed);
        using ArcBuffer bias = lane.Upload(new float[outputWidth]);
        using ArcBuffer x16 = lane.AllocateBytes(rows * inputWidth * sizeof(ushort));
        using ArcBuffer exactOutput = lane.Allocate(rows * outputWidth);
        using ArcBuffer oldOutput = lane.Allocate(rows * outputWidth);
        using ArcBuffer wideOutput = lane.Allocate(rows * outputWidth);
        lane.Run("q35t_linear_iq2_s_rows4",
            ((((long)rows + 3) / 4 * outputWidth + 1) / 2) * 32, 32,
            x, weight, bias, exactOutput, rows, inputWidth, outputWidth);
        lane.Run("q35t_iq2_input_f16", (long)rows * inputWidth, 0,
            x, x16, rows * inputWidth);
        lane.Run2D("q35t_linear_iq2_s_xmx_f16",
            (long)((outputWidth + 31) / 32) * 16, (long)((rows + 63) / 64) * 8,
            16, 8, x16, weight, bias, oldOutput, rows, inputWidth, outputWidth);
        lane.Run2D("q35t_linear_iq2_s_xmx_f16_r128_n64",
            (long)((outputWidth + 63) / 64) * 16, (long)((rows + 127) / 128) * 16,
            16, 16, x16, weight, bias, wideOutput, rows, inputWidth, outputWidth);
        var exact = new float[rows * outputWidth];
        var old = new float[exact.Length];
        var wide = new float[exact.Length];
        lane.Read(exactOutput, exact);
        lane.Read(oldOutput, old);
        lane.Read(wideOutput, wide);
        double errorSq = 0, exactSq = 0;
        for (int i = 0; i < exact.Length; ++i)
        {
            Assert.Equal(old[i], wide[i]);
            double error = wide[i] - exact[i];
            errorSq += error * error;
            exactSq += (double)exact[i] * exact[i];
        }
        Assert.InRange(Math.Sqrt(errorSq / exactSq), 0, .01);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void OptInXmxForwardRunsInLoRaStepAndKeepsGradientsClose(bool fp16)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc is required.");
        using TemporaryQwenGguf fixture = Qwen35ResidentModelTests.CreateFixture(
            tiedOutput: false, iq2Qkv: true);
        var execution = new Qwen35ExecutionOptions
        {
            LoraTraining = true,
            CollectKernelTimings = true,
            TrainingGpuCheckpoints = true
        };
        using Qwen35QuantizedModel reference = Qwen35QuantizedModel.Load(
            fixture.Path, [0], options: execution);
        using Qwen35QuantizedModel experimental = Qwen35QuantizedModel.Load(
            fixture.Path, [0], options: execution with
            {
                TrainingIQ2Bf16XmxForward = !fp16,
                TrainingIQ2Fp16XmxForward = fp16
            });
        var adapter = new Qwen35LoraOptions
        {
            Rank = 2, Alpha = 4, IncludeOutput = true, Seed = 94, LearningRate = .001f
        };
        reference.AttachLora(adapter);
        experimental.AttachLora(adapter);
        reference.LoraCheckpointThresholdRows = 0;
        experimental.LoraCheckpointThresholdRows = 0;
        int[] tokens = [1, 2, 3, 0, 1, 2];

        Qwen35LoraStepResult exact = reference.ComputeLoraGradients(tokens, 3);
        Qwen35LoraStepResult approximate = experimental.ComputeLoraGradients(tokens, 3);
        Assert.Equal(exact.SupervisedTokens, approximate.SupervisedTokens);
        AssertClose(exact.Loss, approximate.Loss, .01);
        AssertClose(exact.GradientNorm, approximate.GradientNorm, .02);
        double gradientErrorSq = 0, gradientReferenceSq = 0;
        foreach (var pair in reference.LoraMatrices)
        {
            float[][] expected = pair.Value.ReadGradients();
            float[][] actual = experimental.LoraMatrices[pair.Key].ReadGradients();
            for (int field = 0; field < expected.Length; ++field)
            for (int i = 0; i < expected[field].Length; ++i)
            {
                double difference = expected[field][i] - actual[field][i];
                gradientErrorSq += difference * difference;
                gradientReferenceSq += (double)expected[field][i] * expected[field][i];
            }
        }
        double gradientRelativeRms = Math.Sqrt(gradientErrorSq / gradientReferenceSq);
        Assert.InRange(gradientRelativeRms, 0, .025);
        Assert.Contains("q35t_linear_iq2_s_rows4", reference.KernelMilliseconds.Keys);
        Assert.Contains(fp16 ? "q35t_iq2_input_f16" : "q35t_iq2_input_bf16",
            experimental.KernelMilliseconds.Keys);
        Assert.Contains(fp16 ? "q35t_linear_iq2_s_xmx_f16" : "q35t_linear_iq2_s_xmx_bf16",
            experimental.KernelMilliseconds.Keys);

        Qwen35LoraStepResult exactStep = reference.TrainLora(tokens, 3);
        Qwen35LoraStepResult approximateStep = experimental.TrainLora(tokens, 3);
        Assert.Equal(exactStep.Step, approximateStep.Step);
        AssertClose(exactStep.Loss, approximateStep.Loss, .01);
        AssertClose(exactStep.GradientNorm, approximateStep.GradientNorm, .02);
        Assert.Equal(reference.ResidentWeightBytes, experimental.ResidentWeightBytes);
    }

    private static void AssertClose(double expected, double actual, double tolerance)
        => Assert.True(double.IsFinite(actual)
            && Math.Abs(expected - actual) <= tolerance * (1 + Math.Abs(expected)),
            $"Expected {expected:R}, actual {actual:R}, tolerance {tolerance:R}");
}
