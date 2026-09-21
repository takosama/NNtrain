using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Serialization;
using NNtrain.Cuda.Execution;
using NNtrain.Runtime.Execution;

namespace NNtrain.Benchmarks;

/// <summary>
/// Small, deterministic depth diagnostic. It is deliberately separate from
/// the training CLI so layer tensors and host-side statistics can never enter
/// a production training step.
/// </summary>
internal static class DrnDepthConvergenceProfiler
{
    internal static void Run(string resultPath, int steps = 200)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(resultPath);
        ArgumentOutOfRangeException.ThrowIfLessThan(steps, 1);
        string output = Path.GetFullPath(resultPath);
        if (File.Exists(output))
            throw new IOException("Depth diagnostic output must be a new file.");
        if (!Tensor.IsCudaAvailable(0))
            throw new InvalidOperationException("CUDA device 0 is unavailable.");

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousDevices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndices = [0];
            VariantResult depth16 = RunVariant(16,
                TensorPrecisionMode.Mix8_32, depthScaled: true, steps);
            VariantResult depth32Unscaled = RunVariant(32,
                TensorPrecisionMode.Mix8_32, depthScaled: false, steps);
            VariantResult depth32 = RunVariant(32,
                TensorPrecisionMode.Mix8_32, depthScaled: true, steps);
            VariantResult depth32Mix16 = RunVariant(32,
                TensorPrecisionMode.Mix16_32, depthScaled: true, steps);

            object[] variants =
            [
                Present(depth16, reference: null),
                Present(depth32Unscaled, depth32Mix16),
                Present(depth32, depth32Mix16),
                Present(depth32Mix16, reference: null),
            ];
            Directory.CreateDirectory(Path.GetDirectoryName(output)!);
            File.WriteAllText(output, JsonSerializer.Serialize(new
            {
                TimestampUtc = DateTimeOffset.UtcNow,
                Conditions = new
                {
                    Device = 0,
                    Batch = 2,
                    Sequence = 16,
                    Vocabulary = 512,
                    Width = 64,
                    Hidden = 192,
                    KeyWidth = 16,
                    ValueWidth = 16,
                    Dropout = 0.1f,
                    InitializationScale = 0.02f,
                    MatrixLearningRate = 0.0001f,
                    AuxiliaryLearningRate = 0.0003f,
                    WeightDecay = 0.01f,
                    Optimizer = "Muon fixed NS5 every step + AdamW",
                    Steps = steps,
                    Workload = "same deterministic repeated synthetic batch",
                    Note = "Diagnostic scale only; not a FineWeb convergence claim.",
                },
                Variants = variants,
            }, new JsonSerializerOptions { WriteIndented = true }));
            Console.WriteLine($"depth diagnostic = {output}");
        }
        finally
        {
            Tensor.CudaDeviceIndices = previousDevices;
            Tensor.ExecutionDevice = previousDevice;
        }
    }

    private static VariantResult RunVariant(int layers,
        TensorPrecisionMode precision, bool depthScaled, int steps)
    {
        PrecisionPolicy policy = PrecisionPolicy.Parse(
            TensorPrecisionModeNames.Format(precision));
        using var session = new ExecutionSession(new ExecutionOptions
        {
            Device = ExecutionDeviceKind.Cuda,
            CudaDevices = new DeviceSet([0]),
            Precision = policy,
        }, [CudaExecutionLaneFactory.Create(0)]);
        using IDisposable scope = session.Enter();
        var rng = new CheckpointableRandom(1234);
        var model = new ForgetMemoryDRNGpt(512, 16, 64, 192, layers,
            16, 16, random: rng, initializationScale: 0.02f,
            dropout: 0.1f, dtype: TensorDType.Float32);
        if (!depthScaled)
        {
            float undo = MathF.Sqrt(2f * layers);
            foreach (ForgetMemoryV2Layer layer in model.Layers)
            {
                Scale(layer.MemoryOutputProjection.W, undo);
                Scale(layer.Ffn.Fc2.W, undo);
            }
        }
        rng.BeginRuntime();
        model.AttachTrainingRandom(rng);
        model.to(precision, 128);
        model.to(TensorDevice.Cuda);
        int[] input = Enumerable.Range(0, 32)
            .Select(index => (index * 37 + 11) % 509 + 3).ToArray();
        int[] target = input.Select(value => (value + 1) % 512).ToArray();

        var layerOutputs = new List<Tensor>(layers);
        Tensor diagnosticLoss = model.ForwardLossWithLayerOutputs(
            input, target, 2, 16, layerOutputs);
        float[][] layerValues = layerOutputs.Select(t => t.Data.ToArray()).ToArray();
        diagnosticLoss.Backward();
        float[][] layerGradients = layerOutputs.Select(t => t.Grad.ToArray()).ToArray();
        model.ZeroGrad();

        var matrix = (NekoMuon)optim.Muon(model.HiddenWeightParameters,
            lr: 0.0001f, momentum: 0.95f, weight_decay: 0.01f);
        var auxiliary = new AdamW(model.AuxiliaryParameters,
            new AdamWOptions { LearningRate = 0.0003f, Beta1 = 0.9f,
                Beta2 = 0.95f, WeightDecay = 0.01f });
        var optimizer = new CompositeOptimizer(matrix, auxiliary);
        float[] losses = new float[steps];
        float[] gradientNorms = new float[steps];
        var timer = Stopwatch.StartNew();
        try
        {
            using var engine = new CudaDataParallelEngine(model, [0]);
            engine.PrepareForTraining(2);
            optimizer.prepare();
            for (int step = 0; step < steps; step++)
            {
                optimizer.zero_grad();
                losses[step] = engine.ForwardBackward(input, target, 2, 16,
                    Tensor.DefaultCrossEntropyIgnoreIndex, step);
                gradientNorms[step] = nn.utils.clip_grad_norm_(
                    model.Parameters(), 1f);
                optimizer.step();
            }
            ForgetMemoryV2Cuda.GetAccelerator(0).Synchronize();
        }
        finally
        {
            try { matrix.DisposeCudaResources(); }
            finally { auxiliary.DisposeCudaResources(); }
        }
        timer.Stop();
        return new VariantResult(
            $"L{layers}-{TensorPrecisionModeNames.Format(precision)}-" +
                (depthScaled ? "depth-scaled" : "unscaled"),
            layers, precision, depthScaled,
            model.Layers[0].ResidualInitializationScale
                * (depthScaled ? 1f : MathF.Sqrt(2f * layers)),
            losses, gradientNorms, timer.Elapsed.TotalSeconds,
            layerValues, layerGradients);
    }

    private static object Present(VariantResult value, VariantResult? reference)
    {
        var layers = new object[value.LayerValues.Length];
        for (int index = 0; index < layers.Length; index++)
        {
            float? error = reference is not null
                && reference.LayerValues.Length == value.LayerValues.Length
                    ? RmsDifference(value.LayerValues[index],
                        reference.LayerValues[index])
                    : null;
            layers[index] = new
            {
                Layer = index + 1,
                ResidualRms = Rms(value.LayerValues[index]),
                GradientRms = Rms(value.LayerGradients[index]),
                CumulativeDifferenceFromMix16Rms = error,
                RelativeDifferenceFromMix16 = error is null
                    ? null
                    : (float?)(error.Value / MathF.Max(
                        Rms(reference!.LayerValues[index]), 1e-12f)),
            };
        }
        return new
        {
            value.Name,
            value.Layers,
            Precision = TensorPrecisionModeNames.Format(value.Precision),
            value.DepthScaled,
            value.ResidualInitializationScale,
            InitialLoss = value.Losses[0],
            FinalLoss = value.Losses[^1],
            LossReduction = value.Losses[0] - value.Losses[^1],
            MinimumLoss = value.Losses.Min(),
            MeanGradientNorm = value.GradientNorms.Average(),
            value.Seconds,
            value.Losses,
            LayerStatistics = layers,
        };
    }

    private static void Scale(Parameter parameter, float factor)
    {
        using Tensor.DataMutation mutation = parameter.BeginUpdate();
        for (int index = 0; index < mutation.Values.Length; index++)
            mutation.Values[index] *= factor;
        parameter.CompleteUpdate();
    }

    private static float Rms(IReadOnlyList<float> values)
    {
        double sum = 0;
        for (int index = 0; index < values.Count; index++)
            sum += (double)values[index] * values[index];
        return (float)Math.Sqrt(sum / values.Count);
    }

    private static float RmsDifference(IReadOnlyList<float> values,
        IReadOnlyList<float> reference)
    {
        if (values.Count != reference.Count)
            throw new ArgumentException("Diagnostic tensors have different sizes.");
        double sum = 0;
        for (int index = 0; index < values.Count; index++)
        {
            double difference = values[index] - reference[index];
            sum += difference * difference;
        }
        return (float)Math.Sqrt(sum / values.Count);
    }

    private sealed record VariantResult(string Name, int Layers,
        TensorPrecisionMode Precision, bool DepthScaled,
        float ResidualInitializationScale, float[] Losses,
        float[] GradientNorms, double Seconds, float[][] LayerValues,
        float[][] LayerGradients);
}
