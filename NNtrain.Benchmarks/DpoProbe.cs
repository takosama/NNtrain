using System.Diagnostics;
using System.Text.Json;
using NNtrain;
using NNtrain.Runtime.Execution;
using NNtrain.Cuda.Execution;

namespace NNtrain.Benchmarks;

internal static class DpoProbe
{
    // Read-only production adapter smoke: restore, then generate exactly one token.
    // Never enters the training/metrics/checkpoint writer path.
    internal static void Resume(string modelPath, string configPath)
    {
        configPath = Path.GetFullPath(configPath);
        var config = LoraConfiguration.Load<LoraTrainingConfiguration>(configPath) with { MaxNewTokens = 1 };
        config.Validate();
        string adapterPath = Path.GetFullPath(config.AdapterPath, Path.GetDirectoryName(configPath)!);
        byte[] Hash() { using var file = File.OpenRead(adapterPath); return System.Security.Cryptography.SHA256.HashData(file); }
        byte[] before = Hash();
        try { DpoCommand.Run(Path.GetFullPath(modelPath), configPath, config, "テスト", Console.Out); }
        finally
        {
            if (!before.SequenceEqual(Hash())) throw new IOException("Resume probe changed the adapter checkpoint.");
            Console.WriteLine("\nresume probe: adapter checkpoint unchanged");
        }
    }

    internal static void Run(string modelPath, string outputPath, int batch, int sequence, bool legacy = false,
        string precision = "mix16_32", int blockSize = 32)
    {
        if (File.Exists(outputPath)) throw new IOException("Use a new benchmark output path.");
        var (model, _) = WikiLanguageModelCommand.LoadLoraBase(Path.GetFullPath(modelPath), TensorPrecisionModeNames.Parse(precision), 1234, blockSize);
        var adapters = model.AttachLora(8, 16, ["memoryOutput", "ffnInput", "ffnOutput"], 1234);
        model.FreezeLoraBaseLinear(!legacy);
        using var session = new ExecutionSession(new ExecutionOptions {
            Device = ExecutionDeviceKind.Cuda, CudaDevices = new DeviceSet([0]), Precision = PrecisionPolicy.Parse(precision)
        }, [CudaExecutionLaneFactory.Create(0)]);
        using var scope = session.Enter();
        using var policy = CudaDispatchPolicy.Push(CudaDispatchPolicy.Current with {
            DisableCudaGraphs = true, EnableBlockBfp8OptimizerState = false });
        model.to(TensorDevice.Cuda);
        var optimizer = new AdamW(adapters.parameters(), new AdamWOptions { LearningRate = .0001f });
        ((IOptimizer)optimizer).prepare();
        var random = new Random(456);
        var pairs = Enumerable.Range(0, batch).Select(i => {
            int cut = Math.Max(1, sequence * (i + 1) / (batch + 1));
            int[] Tokens(int n) => Enumerable.Range(0, n).Select(_ => random.Next(3, model.VocabularySize)).ToArray();
            return new DpoCommand.Pair(i, Tokens(cut), Tokens(sequence + 1 - cut), Tokens(sequence + 1 - cut));
        }).ToArray();
        var packed = DpoCommand.Pack(pairs);
        Tensor[] Scores()
        {
            if (!legacy) return DpoCommand.Scores(model, packed);
            Tensor logits = model.Forward(packed.Input, packed.Labels.Length, packed.Length);
            return packed.Labels.Select((row, i) => logits.Slice(0, i * packed.Length, packed.Length).CrossEntropyWithLogits(row)).ToArray();
        }
        var accelerator = ForgetMemoryV2Cuda.GetAccelerator(0);
        var lane = (CudaExecutionLane)session.GetRequiredLane(ExecutionDeviceKind.Cuda, 0);
        var results = new List<object>();
        IReadOnlyList<CudaOperationProfileSample> operations = [];
        double Time(Action work) { accelerator.Synchronize(); var w = Stopwatch.StartNew(); work(); accelerator.Synchronize(); return w.Elapsed.TotalMilliseconds; }
        try
        {
            for (int step = 0; step < 8; step++)
            {
                var nativeBefore = NNtrain.Cuda.Memory.CudaMemoryManager.NativeTelemetry;
                using var profiling = step == 7 ? CudaOperationProfiler.Begin() : null;
                float[] reference = []; float lossValue = 0;
                double zero = Time(model.ZeroGrad);
                double referenceMs = Time(() => {
                    adapters.SetEnabled(false); model.eval();
                    try {
                        using var noGrad = AutogradContext.NoGrad(); using var inference = CudaInferenceScope.Begin();
                        reference = Scores().Select(t => t.item()).ToArray();
                    } finally { adapters.SetEnabled(true); model.train(); }
                });
                Tensor? loss = null;
                double forward = Time(() => { loss = Tensor.DpoLoss(Scores(), packed.Counts, reference, .1f); lossValue = loss.item(); });
                long forwardAllocated = lane.Memory.Telemetry.AllocatedBytes;
                long forwardActive = lane.Memory.Telemetry.ActiveBytes;
                NativeCudaRuntime.Check(NativeCudaRuntime.MemoryInfoNative(0, out var freeForward, out var totalForward), "DPO probe memory info");
                double backward = Time(() => loss!.BackwardAndRelease([1f]));
                double update = Time(() => { nn.utils.clip_grad_norm_(adapters.parameters(), 1f); optimizer.step(); });
                long allocated = lane.Memory.Telemetry.AllocatedBytes;
                var nativeAfter = NNtrain.Cuda.Memory.CudaMemoryManager.NativeTelemetry;
                if (step == 7) { operations = CudaOperationProfiler.Snapshot(); continue; }
                results.Add(new { Step = step, Warmup = step < 2, Loss = lossValue, ZeroMs = zero, ReferenceMs = referenceMs,
                    ForwardMs = forward, BackwardMs = backward, UpdateMs = update, TotalMs = zero + referenceMs + forward + backward + update,
                    ForwardAllocatedMiB = forwardAllocated / 1048576d, AllocatedMiB = allocated / 1048576d,
                    ForwardActiveMiB = forwardActive / 1048576d, AfterBackwardActiveMiB = lane.Memory.Telemetry.ActiveBytes / 1048576d,
                    NativeAllocations = nativeAfter.AllocationCount - nativeBefore.AllocationCount,
                    NativeReleases = nativeAfter.ReleaseCount - nativeBefore.ReleaseCount,
                    ForwardBoardUsedMiB = (totalForward - freeForward) / 1048576d });
                Console.WriteLine($"step={step} total={zero + referenceMs + forward + backward + update:F1} ms ref={referenceMs:F1} fwd={forward:F1} back={backward:F1} update={update:F1} allocated={allocated / 1048576d:F1} MiB");
            }
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outputPath))!);
            using var file = new FileStream(outputPath, FileMode.CreateNew);
            JsonSerializer.Serialize(file, new { Legacy = legacy, Batch = batch, Sequence = sequence, model.ModelWidth, model.VocabularySize,
                TotalParameters = model.parameters().Sum(p => (long)p.T.Numel), AdapterParameters = adapters.parameters().Sum(p => (long)p.T.Numel),
                Layers = model.Layers.Count, Precision = precision, Bfp8BlockSize = blockSize, Device = 0, Warmup = 2, MeasuredSteps = 5,
                Note = "Fixed synthetic token pairs; copied pretrained base, fresh LoRA; no generation workers. Allocator bytes include cache, not total board VRAM. Extra profiled step excluded from timing.", Results = results, Operations = operations },
                new JsonSerializerOptions { WriteIndented = true });
        }
        finally { optimizer.DisposeCudaResources(); }
    }
}
