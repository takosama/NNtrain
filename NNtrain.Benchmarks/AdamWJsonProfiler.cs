using System.Diagnostics;
using System.Text.Json;
using NNtrain;

namespace NNtrain.Benchmarks;

internal static class AdamWJsonProfiler
{
    private const int WarmupSteps = 8;
    private const int MeasuredSteps = 40;

    internal static void Run(
        string configurationPath,
        int? workerOverride = null,
        bool? simdOverride = null)
    {
        string path = Path.GetFullPath(configurationPath);
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
        JsonElement root = document.RootElement;
        string architecture = root.GetProperty("modelArchitecture").GetString()
            ?? throw new InvalidDataException("modelArchitecture is required.");
        if (!string.Equals(
                architecture,
                "forgetmemoryv2",
                StringComparison.OrdinalIgnoreCase)
            && !string.Equals(
                architecture,
                "forgetmemoryv2",
                StringComparison.OrdinalIgnoreCase))
        {
            throw new NotSupportedException(
                "The JSON AdamW profiler currently requires forgetmemoryv2.");
        }

        int vocabulary = ReadInt(root, "vocabularySize");
        int sequence = ReadInt(root, "contextLength");
        int width = ReadInt(root, "modelWidth");
        int hidden = ReadInt(root, "hiddenSize");
        int layers = ReadInt(root, "layers");
        int keyWidth = ReadInt(root, "forgetMemoryKeyWidth");
        int valueWidth = ReadInt(root, "forgetMemoryValueWidth");
        float retentionMinimum = ReadSingle(
            root,
            "forgetMemoryRetentionMinimum");
        float retentionMaximum = ReadSingle(
            root,
            "forgetMemoryRetentionMaximum");
        float initializationScale = ReadSingle(root, "initializationScale");
        float learningRate = ReadSingle(root, "learningRate");
        float weightDecay = ReadSingle(root, "weightDecay");
        int seed = ReadInt(root, "seed");

        Tensor.SimdEnabled = simdOverride
            ?? root.GetProperty("useSimd").GetBoolean();
        Tensor.MaxDegreeOfParallelism = workerOverride ?? ReadInt(
            root,
            "maxDegreeOfParallelism");

        var model = new ForgetMemoryV2Gpt(
            vocabulary,
            sequence,
            width,
            hidden,
            layers,
            keyWidth,
            valueWidth,
            retentionMinimum,
            retentionMaximum,
            new Random(seed),
            initializationScale,
            dropout: 0f);
        Parameter[] parameters = model.Parameters().ToArray();
        long elementCount = parameters.Sum(parameter => (long)parameter.T.Numel);
        var random = new Random(seed ^ 0x41D3);
        foreach (Parameter parameter in parameters)
        {
            Span<float> gradient = parameter.T.MutableGrad;
            for (int index = 0; index < gradient.Length; index++)
                gradient[index] = (float)(random.NextDouble() * 2d - 1d);
        }

        bool useBFloat16FirstMoment = root.TryGetProperty(
            "adamWUseBFloat16FirstMoment",
            out JsonElement useBFloat16First)
            && useBFloat16First.GetBoolean();
        bool useBFloat16SecondMoment = root.TryGetProperty(
            "adamWUseBFloat16SecondMoment",
            out JsonElement useBFloat16Second)
            && useBFloat16Second.GetBoolean();
        var optimizer = new AdamW(
            parameters,
            new AdamWOptions
            {
                LearningRate = learningRate,
                Beta1 = 0.9f,
                Beta2 = 0.95f,
                WeightDecay = weightDecay,
                UseBFloat16FirstMoment = useBFloat16FirstMoment,
                UseBFloat16SecondMoment = useBFloat16SecondMoment,
            });

        Console.WriteLine($"configuration = {path}");
        Console.WriteLine(
            $"AdamW JSON shape = {architecture}, width {width}, hidden " +
            $"{hidden}, layers {layers}, vocabulary {vocabulary}");
        Console.WriteLine(
            $"parameters = {parameters.Length:N0}, elements = " +
            $"{elementCount:N0}, SIMD = {Tensor.SimdEnabled}, workers = " +
            $"{Tensor.EffectiveMaxDegreeOfParallelism}");
        int firstMomentBytes = useBFloat16FirstMoment
            ? sizeof(short)
            : sizeof(float);
        int secondMomentBytes = useBFloat16SecondMoment
            ? sizeof(short)
            : sizeof(float);
        Console.WriteLine(
            $"largest parameter = {parameters.Max(parameter => parameter.T.Numel):N0}, " +
            $"state bytes = " +
            $"{elementCount * (firstMomentBytes + secondMomentBytes):N0}, " +
            $"moments = {(useBFloat16FirstMoment ? "bf16" : "f32")}/" +
            $"{(useBFloat16SecondMoment ? "bf16" : "f32")}");

        for (int step = 0; step < WarmupSteps; step++)
            optimizer.Step();

        var elapsed = new double[MeasuredSteps];
        long allocatedBefore = GC.GetTotalAllocatedBytes(precise: true);
        int gen0Before = GC.CollectionCount(0);
        int gen1Before = GC.CollectionCount(1);
        int gen2Before = GC.CollectionCount(2);
        for (int step = 0; step < elapsed.Length; step++)
        {
            long start = Stopwatch.GetTimestamp();
            optimizer.Step();
            elapsed[step] = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
        }
        long allocated = GC.GetTotalAllocatedBytes(precise: true)
            - allocatedBefore;
        Array.Sort(elapsed);
        double mean = elapsed.Average();
        double median = Median(elapsed);
        double p95 = Percentile(elapsed, 0.95d);
        double elementsPerSecond = elementCount / (median / 1000d);
        // Read gradient and parameter, read/write both moments, then write
        // the parameter. This estimates useful array traffic only.
        int approximateBytesPerElement = sizeof(float) * 3
            + firstMomentBytes * 2
            + secondMomentBytes * 2;
        double approximateBandwidth = elementsPerSecond
            * approximateBytesPerElement
            / 1_000_000_000d;

        Console.WriteLine(
            $"AdamW mean = {mean:F3} ms, median = {median:F3} ms, " +
            $"p95 = {p95:F3} ms");
        Console.WriteLine(
            $"throughput = {elementsPerSecond / 1_000_000d:F2} M elements/s, " +
            $"approx bandwidth = {approximateBandwidth:F2} GB/s");
        Console.WriteLine(
            $"allocation = {allocated / (double)MeasuredSteps:F0} bytes/step, " +
            $"GC = {GC.CollectionCount(0) - gen0Before}/" +
            $"{GC.CollectionCount(1) - gen1Before}/" +
            $"{GC.CollectionCount(2) - gen2Before}");
    }

    private static int ReadInt(JsonElement root, string name)
        => root.GetProperty(name).GetInt32();

    private static float ReadSingle(JsonElement root, string name)
        => root.GetProperty(name).GetSingle();

    private static double Median(double[] sorted)
        => sorted.Length % 2 == 0
            ? (sorted[sorted.Length / 2 - 1] + sorted[sorted.Length / 2]) / 2d
            : sorted[sorted.Length / 2];

    private static double Percentile(double[] sorted, double percentile)
    {
        int index = (int)Math.Ceiling(percentile * sorted.Length) - 1;
        return sorted[Math.Clamp(index, 0, sorted.Length - 1)];
    }
}
