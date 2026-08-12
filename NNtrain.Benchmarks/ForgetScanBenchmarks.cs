using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using NNtrain;

namespace NNtrain.Benchmarks;

[MemoryDiagnoser]
[SimpleJob(
    RuntimeMoniker.Net10_0,
    launchCount: 1,
    warmupCount: 5,
    iterationCount: 12)]
public class ForgetScanTrainingBenchmarks
{
    private float[] _projection = null!;
    private float[] _gradient = null!;

    [Params(64, 256, 1024)]
    public int SequenceLength { get; set; }

    [Params(4)]
    public int BatchSize { get; set; }

    [Params(512)]
    public int Width { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(1234);
        _projection = new float[BatchSize * SequenceLength * 3 * Width];
        _gradient = new float[BatchSize * SequenceLength * Width];
        for (int index = 0; index < _projection.Length; index++)
            _projection[index] = (float)(random.NextDouble() * 2.0 - 1.0);
        for (int index = 0; index < _gradient.Length; index++)
            _gradient[index] = (float)(random.NextDouble() * 2.0 - 1.0);
    }

    [Benchmark(Baseline = true)]
    public float ScalarSingleThread()
        => Train(useSimd: false, maxDegreeOfParallelism: 1);

    [Benchmark]
    public float SimdSingleThread()
        => Train(useSimd: true, maxDegreeOfParallelism: 1);

    [Benchmark]
    public float SimdParallel()
        => Train(useSimd: true, maxDegreeOfParallelism: 0);

    private float Train(bool useSimd, int maxDegreeOfParallelism)
    {
        Tensor.SimdEnabled = useSimd;
        Tensor.MaxDegreeOfParallelism = maxDegreeOfParallelism;
        var input = new Tensor(
            _projection,
            [BatchSize, SequenceLength, 3 * Width]);
        Tensor output = input.FusedForgetScan();
        output.Backward(_gradient);
        return output.Data[output.Data.Count - 1]
            + input.Grad[input.Grad.Count - 1];
    }
}

[MemoryDiagnoser]
[SimpleJob(
    RuntimeMoniker.Net10_0,
    launchCount: 1,
    warmupCount: 5,
    iterationCount: 12)]
public class ForgetScanInferenceBenchmarks
{
    private float[] _projection = null!;

    [Params(64, 256, 1024)]
    public int SequenceLength { get; set; }

    [Params(4)]
    public int BatchSize { get; set; }

    [Params(512)]
    public int Width { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(5678);
        _projection = new float[BatchSize * SequenceLength * 3 * Width];
        for (int index = 0; index < _projection.Length; index++)
            _projection[index] = (float)(random.NextDouble() * 2.0 - 1.0);
    }

    [Benchmark(Baseline = true)]
    public float ScalarSingleThread()
        => Infer(useSimd: false, maxDegreeOfParallelism: 1);

    [Benchmark]
    public float SimdSingleThread()
        => Infer(useSimd: true, maxDegreeOfParallelism: 1);

    [Benchmark]
    public float SimdParallel()
        => Infer(useSimd: true, maxDegreeOfParallelism: 0);

    private float Infer(bool useSimd, int maxDegreeOfParallelism)
    {
        Tensor.SimdEnabled = useSimd;
        Tensor.MaxDegreeOfParallelism = maxDegreeOfParallelism;
        var input = new Tensor(
            _projection,
            [BatchSize, SequenceLength, 3 * Width]);
        using (AutogradContext.NoGrad())
        {
            Tensor output = input.FusedForgetScan();
            return output.Data[output.Data.Count - 1];
        }
    }
}

[MemoryDiagnoser]
[SimpleJob(
    RuntimeMoniker.Net10_0,
    launchCount: 1,
    warmupCount: 5,
    iterationCount: 12)]
public class ForgetScanModelTrainingBenchmarks
{
    private const int VocabularySize = 256;
    private const int BatchSize = 4;
    private const int SequenceLength = 64;
    private const int ModelWidth = 128;
    private const int HiddenWidth = 256;
    private const int Layers = 2;

    private int[] _tokens = null!;
    private int[] _targets = null!;
    private ForgetScanGpt _scalarModel = null!;
    private ForgetScanGpt _simdModel = null!;
    private ForgetScanGpt _parallelModel = null!;

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(9012);
        _tokens = Enumerable.Range(0, BatchSize * SequenceLength)
            .Select(_ => random.Next(VocabularySize))
            .ToArray();
        _targets = Enumerable.Range(0, BatchSize * SequenceLength)
            .Select(_ => random.Next(VocabularySize))
            .ToArray();
        _scalarModel = CreateModel();
        _simdModel = CreateModel();
        _parallelModel = CreateModel();
    }

    [Benchmark(Baseline = true)]
    public float ScalarSingleThread()
        => Train(
            _scalarModel,
            useSimd: false,
            maxDegreeOfParallelism: 1);

    [Benchmark]
    public float SimdSingleThread()
        => Train(
            _simdModel,
            useSimd: true,
            maxDegreeOfParallelism: 1);

    [Benchmark]
    public float SimdParallel()
        => Train(
            _parallelModel,
            useSimd: true,
            maxDegreeOfParallelism: 0);

    private static ForgetScanGpt CreateModel()
        => new(
            vocabularySize: VocabularySize,
            contextLength: SequenceLength,
            modelWidth: ModelWidth,
            hiddenWidth: HiddenWidth,
            numLayers: Layers,
            random: new Random(3456),
            dropout: 0.1f);

    private float Train(
        ForgetScanGpt model,
        bool useSimd,
        int maxDegreeOfParallelism)
    {
        Tensor.SimdEnabled = useSimd;
        Tensor.MaxDegreeOfParallelism = maxDegreeOfParallelism;
        model.ZeroGrad();
        Tensor logits = model.Forward(_tokens, BatchSize, SequenceLength);
        Tensor loss = logits.CrossEntropyWithLogits(_targets);
        loss.Backward();
        return loss.Data[0];
    }
}

[MemoryDiagnoser]
[SimpleJob(
    RuntimeMoniker.Net10_0,
    launchCount: 1,
    warmupCount: 5,
    iterationCount: 12)]
public class ForgetScanOptimizerBenchmarks
{
    private const int VocabularySize = 256;
    private const int BatchSize = 4;
    private const int SequenceLength = 64;
    private const int ModelWidth = 128;
    private const int HiddenWidth = 256;
    private const int Layers = 2;

    private IOptimizer _scalarOptimizer = null!;
    private IOptimizer _simdOptimizer = null!;
    private IOptimizer _parallelOptimizer = null!;

    [GlobalSetup]
    public void Setup()
    {
        _scalarOptimizer = CreateInitializedOptimizer();
        _simdOptimizer = CreateInitializedOptimizer();
        _parallelOptimizer = CreateInitializedOptimizer();
    }

    [Benchmark(Baseline = true)]
    public void ScalarSingleThread()
        => Step(
            _scalarOptimizer,
            useSimd: false,
            maxDegreeOfParallelism: 1);

    [Benchmark]
    public void SimdSingleThread()
        => Step(
            _simdOptimizer,
            useSimd: true,
            maxDegreeOfParallelism: 1);

    [Benchmark]
    public void SimdParallel()
        => Step(
            _parallelOptimizer,
            useSimd: true,
            maxDegreeOfParallelism: 0);

    private static IOptimizer CreateInitializedOptimizer()
    {
        var model = new ForgetScanGpt(
            vocabularySize: VocabularySize,
            contextLength: SequenceLength,
            modelWidth: ModelWidth,
            hiddenWidth: HiddenWidth,
            numLayers: Layers,
            random: new Random(7890));
        int[] tokens = Enumerable.Range(0, BatchSize * SequenceLength)
            .Select(index => index % VocabularySize)
            .ToArray();
        int[] targets = Enumerable.Range(0, BatchSize * SequenceLength)
            .Select(index => (index + 1) % VocabularySize)
            .ToArray();
        Tensor logits = model.Forward(tokens, BatchSize, SequenceLength);
        logits.CrossEntropyWithLogits(targets).Backward();

        return new CompositeOptimizer(
            new NekoMuon(
                model.HiddenWeightParameters,
                new NekoMuonOptions
                {
                    LearningRate = 3e-4f,
                    WeightDecay = 0.01f,
                }),
            new AdamW(
                model.AuxiliaryParameters,
                new AdamWOptions
                {
                    LearningRate = 3e-4f,
                    Beta1 = 0.9f,
                    Beta2 = 0.95f,
                    WeightDecay = 0.01f,
                }));
    }

    private static void Step(
        IOptimizer optimizer,
        bool useSimd,
        int maxDegreeOfParallelism)
    {
        Tensor.SimdEnabled = useSimd;
        Tensor.MaxDegreeOfParallelism = maxDegreeOfParallelism;
        optimizer.Step();
    }
}

[MemoryDiagnoser]
[SimpleJob(
    RuntimeMoniker.Net10_0,
    launchCount: 1,
    warmupCount: 3,
    iterationCount: 8)]
public class FrogetMemoryV2AttentionBenchmarks
{
    private const int VocabularySize = 256;
    private const int BatchSize = 2;
    private const int ModelWidth = 64;
    private const int HiddenWidth = 128;
    private const int Layers = 2;

    private int[] _tokens = null!;
    private int[] _targets = null!;
    private GptRinWikiJp _attention = null!;
    private FrogetMemoryV2Gpt _memory = null!;

    [Params(64, 128, 256)]
    public int SequenceLength { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        Tensor.SimdEnabled = true;
        Tensor.MaxDegreeOfParallelism = 0;
        var random = new Random(2468);
        _tokens = Enumerable.Range(0, BatchSize * SequenceLength)
            .Select(_ => random.Next(VocabularySize))
            .ToArray();
        _targets = Enumerable.Range(0, BatchSize * SequenceLength)
            .Select(_ => random.Next(VocabularySize))
            .ToArray();
        _attention = new GptRinWikiJp(
            VocabularySize,
            SequenceLength,
            ModelWidth,
            numHeads: 4,
            HiddenWidth,
            Layers,
            rng: new Random(1357));
        _memory = new FrogetMemoryV2Gpt(
            VocabularySize,
            SequenceLength,
            ModelWidth,
            HiddenWidth,
            Layers,
            keyWidth: 32,
            valueWidth: 32,
            random: new Random(1357));
    }

    [Benchmark(Baseline = true)]
    public float Attention()
        => Train(_attention);

    [Benchmark]
    public float FrogetMemoryV2()
        => Train(_memory);

    private float Train(IWikiLanguageModel model)
    {
        foreach (Parameter parameter in model.Parameters())
            parameter.ZeroGrad();
        Tensor logits = model.Forward(
            _tokens,
            BatchSize,
            SequenceLength);
        Tensor loss = logits.CrossEntropyWithLogits(_targets);
        loss.Backward();
        return loss.Data[0];
    }
}
