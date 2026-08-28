using System.Diagnostics;

namespace NNtrain.Benchmarks;

/// <summary>
/// Deterministic numerical probe for the pure tensor-wide BFP8 optimizer
/// contract.  It deliberately reads a handful of sidecar scalars after each
/// BFP8 step; this is a diagnostic command, not a throughput benchmark.
/// </summary>
internal static class PureBfp8StabilityProfiler
{
    private const int Vocabulary = 256;
    private const int Batch = 4;
    private const int Sequence = 16;
    private const int Width = 32;
    private const int Heads = 4;
    private const int Hidden = 64;
    private const int Layers = 1;
    private const int Seed = 991;
    private const float LearningRate = 3e-4f;
    private const float Epsilon = 1e-8f;
    private const float WeightDecay = 0.01f;

    internal static void Run(int steps = 12)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(steps);
        if (!Tensor.IsCudaAvailable())
            throw new InvalidOperationException("CUDA is unavailable.");

        int[] input = Enumerable.Range(0, Batch * Sequence)
            .Select(index => (index * 37 + 11) % Vocabulary)
            .ToArray();
        int[] target = Enumerable.Range(0, Batch * Sequence)
            .Select(index => (index * 53 + 7) % Vocabulary)
            .ToArray();
        Console.WriteLine("pure BFP8 optimizer stability comparison");
        Console.WriteLine(
            $"conditions: steps={steps}, batch={Batch}, sequence={Sequence}, " +
            $"vocab={Vocabulary}, width={Width}, heads={Heads}, " +
            $"hidden={Hidden}, layers={Layers}, dropout=0, " +
            $"lr={LearningRate:G}, epsilon={Epsilon:G}, " +
            $"weight_decay={WeightDecay:G}, clip_norm=1, " +
            "optimizer=NekoMuon(NS5)+AdamW, seed=" + Seed);

        ProbeResult bfp8 = RunCuda(
            TensorPrecisionMode.Bfp8,
            input,
            target,
            steps,
            traceBfp8: true);
        ProbeResult bf16 = RunCuda(
            TensorPrecisionMode.BFloat16,
            input,
            target,
            steps,
            traceBfp8: false);
        ProbeResult cpu = RunCpu(input, target, steps);

        Console.WriteLine("comparison summary");
        PrintSummary("CUDA pure BFP8", bfp8);
        PrintSummary("CUDA BF16", bf16);
        PrintSummary("CPU FP32", cpu);
    }

    private static ProbeResult RunCuda(
        TensorPrecisionMode mode,
        int[] input,
        int[] target,
        int steps,
        bool traceBfp8)
    {
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousDevices = Tensor.CudaDeviceIndices.ToArray();
        Parameter[] parameters = [];
        NekoMuon? neko = null;
        AdamW? adam = null;
        try
        {
            Tensor.CudaDeviceIndices = [0];
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            using IDisposable precision =
                TensorExecutionContext.PushPrecisionPolicy(
                    mode == TensorPrecisionMode.Bfp8
                        ? NNtrain.Runtime.Execution.PrecisionPolicy.Bfp8
                        : NNtrain.Runtime.Execution.PrecisionPolicy.BFloat16);
            var model = CreateModel(mode.ToStorageDType());
            model.to(mode);
            model.to(TensorDevice.Cuda);
            parameters = model.parameters().ToArray();
            (neko, adam, CompositeOptimizer optimizer) =
                CreateOptimizers(model, mode == TensorPrecisionMode.BFloat16);
            using var dataParallel = new CudaDataParallelEngine(model, [0]);
            dataParallel.PrepareForTraining(Batch);
            optimizer.prepare();
            var losses = new float[steps];
            var norms = new float[steps];
            var elapsed = new double[steps];
            for (int step = 0; step < steps; step++)
            {
                var timer = Stopwatch.StartNew();
                optimizer.zero_grad();
                losses[step] = dataParallel.ForwardBackward(
                    input, target, Batch, Sequence);
                norms[step] = nn.utils.clip_grad_norm_(parameters, 1f);
                optimizer.step();
                timer.Stop();
                elapsed[step] = timer.Elapsed.TotalMilliseconds;

                if (traceBfp8)
                {
                    var nekoMoments = neko.GetCudaBfp8Moments(0, 0);
                    var adamMoments = adam.GetCudaBfp8Moments(0, 0);
                    float weightScale = ReadScale(
                        model.AuxiliaryParameters[0].T
                            .EnsureCudaBfp8Buffer(0));
                    float fastScale = ReadScale(nekoMoments.Fast);
                    float slowScale = ReadScale(nekoMoments.Slow);
                    float firstScale = ReadScale(adamMoments.First);
                    float secondScale = ReadScale(adamMoments.Second);
                    float varianceFloor = 0.5f * secondScale;
                    NekoMuonDiagnostics diagnostics = neko.GetDiagnostics();
                    Console.WriteLine(
                        $"BFP8 step {step + 1,2}: loss={losses[step]:F6}, " +
                        $"grad_norm={norms[step]:F4}" +
                        $"{(norms[step] > 1f ? " clipped" : string.Empty)}, " +
                        $"weight_scale={weightScale:E3}, " +
                        $"neko_fast/slow={fastScale:E3}/{slowScale:E3}, " +
                        $"adam_m/v={firstScale:E3}/{secondScale:E3}, " +
                        $"v_floor={varianceFloor:E3}, eps={Epsilon:E1}, " +
                        $"confidence={diagnostics.MeanConfidence:F3}, " +
                        $"finite=yes, {elapsed[step]:F2} ms");
                }
            }
            return new ProbeResult(losses, norms, elapsed);
        }
        finally
        {
            adam?.DisposeCudaResources();
            neko?.DisposeCudaResources();
            foreach (Parameter parameter in parameters)
                parameter.T.InvalidateCudaBuffers();
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousDevices;
        }
    }

    private static ProbeResult RunCpu(
        int[] input,
        int[] target,
        int steps)
    {
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cpu;
            var model = CreateModel(TensorDType.Float32);
            model.to(TensorPrecisionMode.Float32);
            (NekoMuon neko, AdamW adam, CompositeOptimizer optimizer) =
                CreateOptimizers(model, bfloat16Moments: false);
            Parameter[] parameters = model.parameters().ToArray();
            var losses = new float[steps];
            var norms = new float[steps];
            var elapsed = new double[steps];
            for (int step = 0; step < steps; step++)
            {
                var timer = Stopwatch.StartNew();
                optimizer.zero_grad();
                Tensor loss = model.forward_loss(
                    input, target, Batch, Sequence);
                losses[step] = loss.item();
                loss.Backward();
                norms[step] = nn.utils.clip_grad_norm_(parameters, 1f);
                optimizer.step();
                timer.Stop();
                elapsed[step] = timer.Elapsed.TotalMilliseconds;
            }
            neko.DisposeCudaResources();
            adam.DisposeCudaResources();
            return new ProbeResult(losses, norms, elapsed);
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
        }
    }

    private static GptRinWikiJp CreateModel(TensorDType dtype)
        => new(
            Vocabulary,
            Sequence,
            Width,
            Heads,
            Hidden,
            Layers,
            new Random(Seed),
            initializationScale: 0.02f,
            dropout: 0f,
            dtype: dtype,
            tieWordEmbeddings: false);

    private static (NekoMuon Neko, AdamW Adam, CompositeOptimizer Composite)
        CreateOptimizers(GptRinWikiJp model, bool bfloat16Moments)
    {
        var neko = new NekoMuon(
            model.HiddenWeightParameters,
            new NekoMuonOptions
            {
                LearningRate = LearningRate,
                WeightDecay = WeightDecay,
                MaxNewtonSchulzSteps = 5,
                NewtonSchulzInterval = 1,
                NewtonSchulzDepthMode =
                    NekoMuonNewtonSchulzDepthMode.Fixed,
                NewtonSchulzDepth = 5f,
            });
        var adam = new AdamW(
            model.AuxiliaryParameters,
            new AdamWOptions
            {
                LearningRate = LearningRate,
                Beta1 = 0.9f,
                Beta2 = 0.95f,
                Epsilon = Epsilon,
                WeightDecay = WeightDecay,
                UseBFloat16FirstMoment = bfloat16Moments,
                UseBFloat16SecondMoment = bfloat16Moments,
            });
        return (neko, adam, new CompositeOptimizer(neko, adam));
    }

    private static float ReadScale(CudaBfp8BufferView view)
    {
        Span<float> scale = stackalloc float[1];
        view.Scales.CopyToCPU(scale);
        return scale[0];
    }

    private static void PrintSummary(string name, ProbeResult result)
    {
        int clipCount = result.Norms.Count(value => value > 1f);
        double[] ordered = result.Elapsed.Order().ToArray();
        double median = ordered[ordered.Length / 2];
        Console.WriteLine(
            $"{name,-15}: loss {result.Losses[0]:F6} -> " +
            $"{result.Losses[^1]:F6} " +
            $"({result.Losses[^1] - result.Losses[0]:+0.000000;-0.000000;0.000000}), " +
            $"finite={result.Losses.All(float.IsFinite)}, " +
            $"clip={clipCount}/{result.Losses.Length}, " +
            $"mean/p50={result.Elapsed.Average():F2}/{median:F2} ms");
    }

    private readonly record struct ProbeResult(
        float[] Losses,
        float[] Norms,
        double[] Elapsed);
}
