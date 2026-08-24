using System.Text.Json;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using NNtrain;

namespace NNtrain.Benchmarks;

[MemoryDiagnoser]
[SimpleJob(
    RuntimeMoniker.Net10_0,
    launchCount: 1,
    warmupCount: 8,
    iterationCount: 20)]
public class AdamWJsonBenchmarks
{
    private AdamW _optimizer = null!;
    private Parameter _firstParameter = null!;
    private ReferenceAdamW _referenceOptimizer = null!;
    private Parameter _referenceFirstParameter = null!;

    [GlobalSetup]
    public void Setup()
    {
        string path = FindConfiguration();
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
        JsonElement root = document.RootElement;
        Tensor.SimdEnabled = root.GetProperty("useSimd").GetBoolean();
        Tensor.MaxDegreeOfParallelism = root
            .GetProperty("maxDegreeOfParallelism")
            .GetInt32();
        AdamWOptions options = new()
        {
            LearningRate = ReadSingle(root, "learningRate"),
            Beta1 = 0.9f,
            Beta2 = 0.95f,
            WeightDecay = ReadSingle(root, "weightDecay"),
            UseBFloat16FirstMoment = ReadBoolean(
                root,
                "adamWUseBFloat16FirstMoment"),
            UseBFloat16SecondMoment = ReadBoolean(
                root,
                "adamWUseBFloat16SecondMoment"),
        };
        Parameter[] parameters = CreateParameters(root);
        Parameter[] referenceParameters = CreateParameters(root);

        _firstParameter = parameters[0];
        _optimizer = new AdamW(parameters, options);
        _referenceFirstParameter = referenceParameters[0];
        _referenceOptimizer = new ReferenceAdamW(
            referenceParameters,
            options);
    }

    [Benchmark(Baseline = true)]
    public float ReferenceStep()
    {
        _referenceOptimizer.Step();
        return _referenceFirstParameter.T.Data[0];
    }

    [Benchmark]
    public float Step()
    {
        _optimizer.Step();
        return _firstParameter.T.Data[0];
    }

    [Benchmark]
    public float ReferenceZeroGrad()
    {
        _referenceOptimizer.ZeroGrad();
        return _referenceFirstParameter.T.Grad[0];
    }

    [Benchmark]
    public float ZeroGrad()
    {
        _optimizer.ZeroGrad();
        return _firstParameter.T.Grad[0];
    }

    private static Parameter[] CreateParameters(JsonElement root)
    {
        var model = new ForgetMemoryV2Gpt(
            vocabularySize: ReadInt(root, "vocabularySize"),
            contextLength: ReadInt(root, "contextLength"),
            modelWidth: ReadInt(root, "modelWidth"),
            hiddenWidth: ReadInt(root, "hiddenSize"),
            numLayers: ReadInt(root, "layers"),
            keyWidth: ReadInt(root, "forgetMemoryKeyWidth"),
            valueWidth: ReadInt(root, "forgetMemoryValueWidth"),
            retentionMinimum: ReadSingle(
                root,
                "forgetMemoryRetentionMinimum"),
            retentionMaximum: ReadSingle(
                root,
                "forgetMemoryRetentionMaximum"),
            random: new Random(ReadInt(root, "seed")),
            initializationScale: ReadSingle(root, "initializationScale"),
            dropout: 0f);
        Parameter[] parameters = model.Parameters().ToArray();
        var random = new Random(ReadInt(root, "seed") ^ 0x41D3);
        foreach (Parameter parameter in parameters)
        {
            Span<float> gradient = parameter.T.MutableGrad;
            for (int index = 0; index < gradient.Length; index++)
                gradient[index] = (float)(random.NextDouble() * 2d - 1d);
        }

        return parameters;
    }

    private static string FindConfiguration()
    {
        string? explicitPath = Environment.GetEnvironmentVariable(
            "NNTRAIN_ADAMW_CONFIG");
        if (!string.IsNullOrWhiteSpace(explicitPath))
            return Path.GetFullPath(explicitPath);

        foreach (string start in new[]
        {
            Directory.GetCurrentDirectory(),
            AppContext.BaseDirectory,
        })
        {
            var directory = new DirectoryInfo(start);
            while (directory is not null)
            {
                string candidate = Path.Combine(
                    directory.FullName,
                    "training.wiki-jp.json");
                if (File.Exists(candidate))
                    return candidate;
                directory = directory.Parent;
            }
        }

        throw new FileNotFoundException(
            "training.wiki-jp.json was not found. Set " +
            "NNTRAIN_ADAMW_CONFIG to an absolute path.");
    }

    private static int ReadInt(JsonElement root, string name)
        => root.GetProperty(name).GetInt32();

    private static float ReadSingle(JsonElement root, string name)
        => root.GetProperty(name).GetSingle();

    private static bool ReadBoolean(JsonElement root, string name)
        => root.TryGetProperty(name, out JsonElement value)
            && value.GetBoolean();
}
