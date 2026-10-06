using System.Diagnostics;
using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;
using Xunit;

namespace NNtrain.Core.Tests;

public sealed class Qwen35VisionAttentionXmxTests(ITestOutputHelper output)
{
    [Theory]
    [InlineData(1, 1, 1)]
    [InlineData(3, 2, 7)]
    [InlineData(19, 3, 72)]
    [InlineData(129, 2, 72)]
    [InlineData(256, 2, 128)]
    public void FullAttentionRetainsFp32ProbabilitiesAndMatchesReferenceWithinRounding(
        int rows, int heads, int headWidth)
    {
        ArcDeviceInfo? device = ArcDevices.Enumerate().FirstOrDefault();
        Assert.SkipWhen(device is null || !device.SupportsXmx
            || device.MinimumSubgroupSize != 16 || !device.Extensions.Split(' ').Contains("cl_khr_fp16"),
            "Arc SG16 FP16 XMX GPU required.");
        using var lane = new ArcExecutionLane(options: new ArcExecutionOptions
        {
            Qwen35InferenceKernelsOnly = true, Qwen35VisionKernelsOnly = true,
            Qwen35VisionFlashAttention = true,
            CacheProgramBinary = true, CollectKernelTimings = false,
            BufferPoolBytes = 256L * 1024 * 1024, DeferredReleaseBytes = 128L * 1024 * 1024
        });
        Compare(lane, rows, heads, headWidth);
    }

    [Fact]
    public void Full2304PatchAttentionRetainsAllScoresAndContext()
        => FullAttentionRetainsFp32ProbabilitiesAndMatchesReferenceWithinRounding(2304, 16, 72);

    [Fact]
    public void OptionalFullShapeWarmBenchmark()
    {
        if (Environment.GetEnvironmentVariable("NNTRAIN_VISION_ATTENTION_BENCHMARK") != "1") return;
        ArcDeviceInfo? device = ArcDevices.Enumerate().FirstOrDefault();
        Assert.SkipWhen(device is null || !device.SupportsXmx
            || device.MinimumSubgroupSize != 16 || !device.Extensions.Split(' ').Contains("cl_khr_fp16"),
            "Arc SG16 FP16 XMX GPU required.");
        using var lane = new ArcExecutionLane(options: new()
        {
            Qwen35InferenceKernelsOnly = true, Qwen35VisionKernelsOnly = true,
            CacheProgramBinary = true, CollectKernelTimings = false, CacheKernelArguments = true,
            BufferPoolBytes = 256L * 1024 * 1024, DeferredReleaseBytes = 128L * 1024 * 1024
        });
        const int rows = 2304, heads = 16, width = 72;
        var random = new Random(943011);
        float[] source = Enumerable.Range(0, rows * 3 * heads * width)
            .Select(_ => (random.NextSingle() - .5f) * 3f).ToArray();
        using ArcBuffer qkv = lane.Upload(source);
        using ArcBuffer scores = lane.Allocate(rows * heads * rows);
        void Candidate(bool head80)
        {
            using ArcBuffer result = Qwen35VisionEncoder.AttentionXmx(lane, qkv, scores,
                rows, heads, width, specializeHead: head80);
        }
        Candidate(false); Candidate(true); lane.Synchronize();
        double Time(bool head80)
        {
            lane.Synchronize(); var timer = Stopwatch.StartNew();
            for (int i = 0; i < 5; i++) Candidate(head80);
            lane.Synchronize(); return timer.Elapsed.TotalMilliseconds / 5;
        }
        double generic = Time(false), specialized = Time(true), genericAgain = Time(false);
        output.WriteLine($"Warm full vision direct-P: generic {generic:F3} ms, direct80 {specialized:F3} ms, generic-repeat {genericAgain:F3} ms, speedup {(generic + genericAgain) / (2 * specialized):F2}x");
    }

    private void Compare(ArcExecutionLane lane, int rows, int heads, int headWidth)
    {
        int width = heads * headWidth, scoreCount = checked(rows * heads * rows);
        var random = new Random(702193 + rows);
        float[] source = Enumerable.Range(0, rows * 3 * width)
            .Select(_ => (random.NextSingle() - .5f) * 3f).ToArray();
        using ArcBuffer qkv = lane.Upload(source);
        using ArcBuffer referenceScores = lane.Allocate(scoreCount);
        using ArcBuffer candidateScores = lane.Allocate(scoreCount);
        using ArcBuffer referenceOutput = lane.Allocate(checked(rows * width));
        void Reference()
        {
            lane.Run2D("q35v_attention_scores_tiled", Round16(rows), heads * Round16(rows),
                16, 16, qkv, referenceScores, rows, heads, headWidth);
            lane.Run("q35v_attention_softmax", (long)rows * heads * 256, 256, referenceScores, rows);
            lane.Run2D("q35v_attention_context_tiled", Round16(headWidth), heads * Round16(rows),
                16, 16, qkv, referenceScores, referenceOutput, rows, heads, headWidth);
        }
        lane.Synchronize(); var timer = Stopwatch.StartNew(); Reference(); lane.Synchronize();
        double referenceMs = timer.Elapsed.TotalMilliseconds;
        timer.Restart();
        using ArcBuffer candidateOutput = Qwen35VisionEncoder.AttentionXmx(lane, qkv,
            candidateScores, rows, heads, headWidth);
        lane.Synchronize();
        double candidateMs = timer.Elapsed.TotalMilliseconds;
        CompareError(referenceScores, candidateScores, scoreCount, "probabilities", 1e-5, 5e-6);
        CompareError(referenceOutput, candidateOutput, rows * width, "context", 3e-5, 1e-5);
        if (headWidth <= 80)
        {
            timer.Restart();
            using ArcBuffer genericOutput = Qwen35VisionEncoder.AttentionXmx(lane, qkv,
                candidateScores, rows, heads, headWidth, specializeHead: false);
            lane.Synchronize();
            double genericMs = timer.Elapsed.TotalMilliseconds;
            CompareError(referenceOutput, genericOutput, rows * width, "generic-context", 3e-5, 1e-5);
            output.WriteLine($"Vision direct-P specialization {rows}x{heads}x{headWidth}: generic {genericMs:F3} ms, direct80 {candidateMs:F3} ms, speedup {genericMs / candidateMs:F2}x");
            timer.Restart();
            using ArcBuffer flashOutput = Qwen35VisionEncoder.AttentionXmxFlash(lane, qkv, rows, heads, headWidth);
            lane.Synchronize();
            double flashMs = timer.Elapsed.TotalMilliseconds;
            CompareError(referenceOutput, flashOutput, rows * width, "flash-context", 3e-5, 1e-5);
            output.WriteLine($"Vision FlashAttention {rows}x{heads}x{headWidth}: {flashMs:F3} ms, speedup {referenceMs / flashMs:F2}x");
        }
        output.WriteLine($"Vision attention {rows}x{heads}x{headWidth}: exact {referenceMs:F3} ms, XMX {candidateMs:F3} ms, speedup {referenceMs / candidateMs:F2}x");

        void CompareError(ArcBuffer reference, ArcBuffer candidate, int count, string stage,
            double maximumAbsolute, double maximumRelative)
        {
            const int chunk = 1024 * 1024;
            double squaredError = 0, squaredReference = 0, maxAbsolute = 0;
            for (int offset = 0; offset < count; offset += chunk)
            {
                int length = Math.Min(chunk, count - offset);
                var expected = new float[length]; var actual = new float[length];
                lane.ReadFloatRange(reference, offset, expected);
                lane.ReadFloatRange(candidate, offset, actual);
                for (int i = 0; i < length; i++)
                {
                    Assert.True(float.IsFinite(actual[i]), $"Nonfinite {stage} at {offset + i}");
                    double error = (double)actual[i] - expected[i];
                    squaredError += error * error;
                    squaredReference += (double)expected[i] * expected[i];
                    maxAbsolute = Math.Max(maxAbsolute, Math.Abs(error));
                }
            }
            double relative = Math.Sqrt(squaredError / Math.Max(squaredReference, 1e-30));
            output.WriteLine($"{stage}: count={count} max_abs={maxAbsolute:G9} relative_L2={relative:G9}");
            Assert.InRange(maxAbsolute, 0, maximumAbsolute);
            Assert.InRange(relative, 0, maximumRelative);
        }
    }

    private static long Round16(int value) => (value + 15L) / 16 * 16;
}
