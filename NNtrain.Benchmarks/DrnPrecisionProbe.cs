using System.Diagnostics;
using System.Text.Json;
using NNtrain.Cuda.Execution;
using NNtrain.Runtime.Execution;

namespace NNtrain.Benchmarks;

// Bounded experiments only. No production checkpoint, tokenizer or metrics writes.
internal static class DrnPrecisionProbe
{
    internal static void Run(string configPath, string outputPath, string mode, int batch, int examples, int sequence = 32)
    {
        string output = Path.GetFullPath(outputPath);
        if (File.Exists(output)) throw new IOException("Probe output must be a new file.");
        if (sequence is not (32 or 512)) throw new ArgumentException("Use sequence 32/512.");
        if (batch is not (5 or 12) || examples <= 0 || examples > 6000 || examples % 60 != 0)
            throw new ArgumentException("Use batch 5/12 and a positive multiple of 60 examples, at most 6000.");
        var precision = TensorPrecisionModeNames.Parse(mode);
        if (precision is not (TensorPrecisionMode.Mix8_32 or TensorPrecisionMode.Mix16_32))
            throw new ArgumentException("Use mix8_32 or mix16_32.");
        var config = WikiTrainingConfiguration.Load(configPath) with
        { BatchSize = 1, GradientAccumulationSteps = 1, ContextLength = sequence };
        Console.WriteLine("Loading identical read-only FineWeb probe tokens...");
        var data = DrnRealDataProbe.Load(config, examples);
        var oldDevice = Tensor.ExecutionDevice;
        int[] oldDevices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndices = [0];
            if (!Tensor.IsCudaAvailable(0)) throw new InvalidOperationException("GPU0 required.");
            Console.WriteLine($"{mode}, batch={batch}, examples={examples}, token SHA256={data.TokenHash}");
            var trained = Train(config, data, precision, batch);
            var reference = Snapshot(config, data, trained.State, TensorPrecisionMode.Mix16_32, 128, false);
            var weightOnly = Snapshot(config, data, trained.State, TensorPrecisionMode.Mix16_32, 128, true);
            var mixed128 = Snapshot(config, data, trained.State, TensorPrecisionMode.Mix8_32, 128, false);
            var mixed32 = Snapshot(config, data, trained.State, TensorPrecisionMode.Mix8_32, 32, false);
            object Compare(string name, SnapshotResult value) => new
            {
                Name = name,
                value.Loss,
                LossDelta = value.Loss - reference.Loss,
                GradientCosine = Cosine(reference.Gradient, value.Gradient),
                GradientRelativeError = RelativeError(reference.Gradient, value.Gradient),
            };
            Directory.CreateDirectory(Path.GetDirectoryName(output)!);
            using var file = new FileStream(output, FileMode.CreateNew);
            JsonSerializer.Serialize(file, new
            {
                TimestampUtc = DateTimeOffset.UtcNow,
                Conditions = new
                {
                    Mode = mode,
                    Batch = batch,
                    Examples = examples,
                    Sequence = sequence,
                    Width = 64,
                    Hidden = 192,
                    Layers = 32,
                    Key = 16,
                    Value = 16,
                    config.VocabularySize,
                    config.MaxDocumentTokens,
                    config.Seed,
                    Dropout = 0.1,
                    MatrixLR = 0.05,
                    AuxiliaryLR = 0.015,
                    Optimizer = "Muon momentum .95 NS5 + AdamW (.9,.95)",
                    Schedule = "constant",
                    WeightDecay = 0.01,
                    Clip = 1,
                    data.TokenHash,
                    data.LoadSeconds,
                    Device = 0,
                    Note = "Small fresh model; concurrent production training. Not the stalled checkpoint."
                },
                trained.Losses,
                trained.Seconds,
                trained.PeakAllocatedBytes,
                trained.Updates,
                SameMasterComparisons = new[] { Compare("mix16", reference),
                    Compare("BFP8-rounded weights with BF16 activations", weightOnly),
                    Compare("mix8 block128", mixed128), Compare("mix8 block32", mixed32) },
            }, new JsonSerializerOptions { WriteIndented = true });
            Console.WriteLine($"Saved {output}");
        }
        finally { Tensor.CudaDeviceIndices = oldDevices; Tensor.ExecutionDevice = oldDevice; }
    }

    private static ExecutionSession Session(TensorPrecisionMode mode) => new(new ExecutionOptions
    {
        Device = ExecutionDeviceKind.Cuda,
        CudaDevices = new DeviceSet([0]),
        Precision = PrecisionPolicy.Parse(TensorPrecisionModeNames.Format(mode))
    }, [CudaExecutionLaneFactory.Create(0)]);

    private static ForgetMemoryDRNGpt Model(WikiTrainingConfiguration config, TensorPrecisionMode precision,
        int block, ModuleState? state = null, bool roundedWeights = false)
    {
        var rng = new CheckpointableRandom(config.Seed);
        var model = new ForgetMemoryDRNGpt(config.VocabularySize, config.ContextLength, 64, 192, 32, 16, 16,
            random: rng, dropout: .1f, dtype: TensorDType.Float32);
        if (state is not null)
        {
            // Normalize state metadata because the destination starts in FP32.
            model.load_state_dict(state with
            {
                Parameters = state.Parameters.Select(p => p with
                { DType = TensorDType.Float32, StorageMetadata = null }).ToArray()
            });
        }
        if (roundedWeights)
        {
            foreach (var p in model.Parameters())
            {
                var encoded = Bfp8QuantizationCodec.Default.Encode(p.T.Data.ToArray(), Bfp8QuantizationDescriptor.Block(block));
                var decoded = new float[p.T.Numel];
                Bfp8QuantizationCodec.Default.Decode(encoded.Payload.Span, encoded.Scales.Span,
                    Bfp8QuantizationDescriptor.Block(block), decoded);
                using var update = p.BeginUpdate();
                decoded.AsSpan().CopyTo(update.Values);
            }
        }
        rng.BeginRuntime(); model.AttachTrainingRandom(rng);
        model.to(precision, block); model.to(TensorDevice.Cuda);
        return model;
    }

    private static TrainingResult Train(WikiTrainingConfiguration config, DrnRealDataProbe data,
        TensorPrecisionMode precision, int batch)
    {
        using var session = Session(precision); using var scope = session.Enter();
        var model = Model(config, precision, 128);
        using var engine = new CudaDataParallelEngine(model, [0]);
        var matrix = (NekoMuon)optim.Muon(model.HiddenWeightParameters, lr: .05f);
        var auxiliary = new AdamW(model.AuxiliaryParameters,
            new AdamWOptions { LearningRate = .015f, Beta1 = .9f, Beta2 = .95f, WeightDecay = .01f });
        try
        {
            var optimizer = new CompositeOptimizer(matrix, auxiliary);
            engine.PrepareForTraining(batch); optimizer.prepare();
            var losses = new List<object>(); long peak = 0;
            var timer = Stopwatch.StartNew(); double windowLoss = 0; int windowSteps = 0;
            int steps = data.Training.Length / batch;
            for (int step = 0; step < steps; step++)
            {
                var samples = data.Training.AsSpan(step * batch, batch).ToArray();
                int[] tokens = samples.SelectMany(x => x.Input).ToArray();
                int[] targets = samples.SelectMany(x => x.Target).ToArray();
                using (DeviceTransferGuard.EnterTrainingStep(1))
                {
                    optimizer.zero_grad();
                    float loss = engine.ForwardBackward(tokens, targets, batch, config.ContextLength, -1, step);
                    float norm = nn.utils.clip_grad_norm_(model.Parameters(), 1f);
                    if (!float.IsFinite(loss) || !float.IsFinite(norm)) throw new InvalidOperationException("Nonfinite probe step.");
                    optimizer.step(); windowLoss += loss; windowSteps++;
                }
                var lane = (CudaExecutionLane)session.GetRequiredLane(ExecutionDeviceKind.Cuda, 0);
                peak = Math.Max(peak, lane.Memory.Telemetry.AllocatedBytes);
                if (peak > 512L * 1024 * 1024)
                    throw new InvalidOperationException($"Probe exceeded 512 MiB allocator budget: {peak / 1048576d:F1} MiB.");
                if ((step + 1) % 20 == 0 || step + 1 == steps)
                {
                    losses.Add(new { Step = step + 1, Examples = (step + 1) * batch, Loss = windowLoss / windowSteps });
                    Console.WriteLine($"{precision} batch={batch} {step + 1}/{steps}: loss={windowLoss / windowSteps:F5}, allocator={peak / 1048576d:F1} MiB");
                    windowLoss = 0; windowSteps = 0;
                }
            }
            ForgetMemoryV2Cuda.GetAccelerator(0).Synchronize(); timer.Stop();
            engine.ReleaseCheckpointTransientMemory();
            return new(model.state_dict(), losses.ToArray(), steps, timer.Elapsed.TotalSeconds, peak);
        }
        finally
        {
            try { matrix.DisposeCudaResources(); }
            finally { auxiliary.DisposeCudaResources(); }
        }
    }

    private static SnapshotResult Snapshot(WikiTrainingConfiguration config, DrnRealDataProbe data,
        ModuleState state, TensorPrecisionMode precision, int block, bool roundedWeights)
    {
        using var session = Session(precision); using var scope = session.Enter();
        var model = Model(config, precision, block, state, roundedWeights);
        model.eval(); // Dropout disabled, autograd enabled: identical loss contract to training.
        double total = 0; float[] gradient = [];
        for (int i = 0; i < data.Evaluation.Length; i++)
        {
            model.ZeroGrad(); var batch = data.Evaluation[i];
            var loss = model.ForwardLoss(batch.Input, batch.Target, batch.BatchSize, batch.SequenceLength);
            float value = loss.item();
            if (!float.IsFinite(value)) throw new InvalidOperationException("Nonfinite probe evaluation.");
            total += value; loss.BackwardAndRelease();
            if (i == 0) gradient = model.Parameters().SelectMany(p => p.T.Grad.ToArray()).ToArray();
        }
        Console.WriteLine($"same-master {precision} block={block} rounded={roundedWeights}: eval={total / data.Evaluation.Length:F6}");
        return new(total / data.Evaluation.Length, gradient);
    }

    private static double Cosine(float[] a, float[] b)
    {
        double aa = 0, bb = 0, ab = 0;
        for (int i = 0; i < a.Length; i++) { aa += (double)a[i] * a[i]; bb += (double)b[i] * b[i]; ab += (double)a[i] * b[i]; }
        return ab / Math.Max(Math.Sqrt(aa * bb), 1e-30);
    }
    private static double RelativeError(float[] a, float[] b)
    {
        double aa = 0, error = 0;
        for (int i = 0; i < a.Length; i++) { aa += (double)a[i] * a[i]; double d = (double)a[i] - b[i]; error += d * d; }
        return Math.Sqrt(error / Math.Max(aa, 1e-30));
    }
    private sealed record TrainingResult(ModuleState State, object[] Losses, int Updates, double Seconds, long PeakAllocatedBytes);
    private sealed record SnapshotResult(double Loss, float[] Gradient);
}
