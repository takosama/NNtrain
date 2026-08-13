using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using NNtrain;

namespace NNtrain.Benchmarks;

public enum LinearProjectionShape
{
    FeedForwardExpansion,
    VocabularyHead,
}

[MemoryDiagnoser]
[SimpleJob(
    RuntimeMoniker.Net10_0,
    launchCount: 1,
    warmupCount: 3,
    iterationCount: 8)]
public class LinearLastDimTrainingBenchmarks
{
    private const int BatchSize = 2;
    private const int SequenceLength = 128;
    private const int InputWidth = 256;

    private Linear _formerLinear = null!;
    private Linear _directLinear = null!;
    private Tensor _formerInput = null!;
    private Tensor _directInput = null!;
    private float[] _gradient = null!;
    private int _outputWidth;
    private int _previousParallelism;

    [Params(
        LinearProjectionShape.FeedForwardExpansion,
        LinearProjectionShape.VocabularyHead)]
    public LinearProjectionShape Projection { get; set; }

    [Params(TensorDType.Float32, TensorDType.Float16)]
    public TensorDType DType { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _previousParallelism = Tensor.MaxDegreeOfParallelism;
        Tensor.MaxDegreeOfParallelism = 1;
        _outputWidth = Projection switch
        {
            LinearProjectionShape.FeedForwardExpansion => 512,
            LinearProjectionShape.VocabularyHead => 4096,
            _ => throw new ArgumentOutOfRangeException(),
        };
        var random = new Random(1234);
        float[] values = Enumerable.Range(
                0,
                BatchSize * SequenceLength * InputWidth)
            .Select(_ => (float)(random.NextDouble() * 2d - 1d))
            .ToArray();
        _gradient = Enumerable.Range(
                0,
                BatchSize * SequenceLength * _outputWidth)
            .Select(index => ((index * 13) % 31 - 15) * 0.001f)
            .ToArray();
        _formerLinear = new Linear(
            InputWidth,
            _outputWidth,
            new Random(5678),
            dtype: DType);
        _directLinear = new Linear(
            InputWidth,
            _outputWidth,
            new Random(5678),
            dtype: DType);
        _formerInput = new Tensor(
            values,
            [BatchSize, SequenceLength, InputWidth],
            dtype: DType);
        _directInput = new Tensor(
            values,
            [BatchSize, SequenceLength, InputWidth],
            dtype: DType);
    }

    [GlobalCleanup]
    public void Cleanup()
        => Tensor.MaxDegreeOfParallelism = _previousParallelism;

    [Benchmark(Baseline = true)]
    public float FormerReshapeGraph()
    {
        _formerLinear.ZeroGrad();
        _formerInput.ZeroGrad();
        Tensor flattened = _formerInput.Reshape(
            BatchSize * SequenceLength,
            InputWidth);
        Tensor projected = flattened.MatMulTransposedRightAddRow(
            _formerLinear.W.T,
            _formerLinear.B.T);
        Tensor output = projected.Reshape(
            BatchSize,
            SequenceLength,
            _outputWidth);
        output.Backward(_gradient);
        return output.Data[^1] + _formerInput.Grad[^1];
    }

    [Benchmark]
    public float DirectLastDimension()
    {
        _directLinear.ZeroGrad();
        _directInput.ZeroGrad();
        Tensor output = _directLinear.ForwardBatch(_directInput);
        output.Backward(_gradient);
        return output.Data[^1] + _directInput.Grad[^1];
    }
}
