using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace NNtrain.Benchmarks;

[MemoryDiagnoser]
[SimpleJob(
    RuntimeMoniker.Net10_0,
    launchCount: 1,
    warmupCount: 2,
    iterationCount: 5)]
public class ForgetMemoryV3DrnBenchmarks
{
    private const int VocabularySize = 256;
    private const int BatchSize = 2;
    private const int ModelWidth = 64;
    private const int HiddenWidth = 128;
    private const int Layers = 2;

    private int[] _tokens = null!;
    private int[] _targets = null!;
    private ForgetMemoryV3Gpt _v3 = null!;
    private ForgetMemoryDRNGpt _drn = null!;

    [Params(128, 256, 512, 1024)]
    public int SequenceLength { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        Tensor.ExecutionDevice = TensorDevice.Cpu;
        Tensor.SimdEnabled = true;
        Tensor.MaxDegreeOfParallelism = 0;
        var random = new Random(2468);
        _tokens = Enumerable.Range(0, BatchSize * SequenceLength)
            .Select(_ => random.Next(VocabularySize))
            .ToArray();
        _targets = Enumerable.Range(0, BatchSize * SequenceLength)
            .Select(_ => random.Next(VocabularySize))
            .ToArray();
        _v3 = new ForgetMemoryV3Gpt(
            VocabularySize,
            SequenceLength,
            ModelWidth,
            HiddenWidth,
            Layers,
            keyWidth: 32,
            valueWidth: 32,
            random: new Random(1357),
            dtype: TensorDType.Float32);
        _drn = new ForgetMemoryDRNGpt(
            VocabularySize,
            SequenceLength,
            ModelWidth,
            HiddenWidth,
            Layers,
            keyWidth: 32,
            valueWidth: 32,
            random: new Random(1357),
            dtype: TensorDType.Float32);
    }

    [Benchmark(Baseline = true)]
    public float V3()
        => Train(_v3);

    [Benchmark]
    public float Drn()
        => Train(_drn);

    private float Train(LanguageModel model)
    {
        foreach (Parameter parameter in model.Parameters())
            parameter.ZeroGrad();
        Tensor logits = model.Forward(_tokens, BatchSize, SequenceLength);
        Tensor loss = logits.CrossEntropyWithLogits(_targets);
        loss.Backward();
        return loss.Data[0];
    }
}
