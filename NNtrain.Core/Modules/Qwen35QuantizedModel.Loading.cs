using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

public sealed partial class Qwen35QuantizedModel
{
    public static Qwen35QuantizedModel Load(
        string path, IReadOnlyList<int>? devices = null, Action<string>? progress = null,
        Qwen35ExecutionOptions? options = null)
    {
        using var reader = new GgufReader(path);
        return Load(reader, devices, progress, options);
    }

    /// <summary>Loads from an already parsed GGUF. The caller retains ownership of the reader.</summary>
    public static Qwen35QuantizedModel Load(
        GgufReader gguf, IReadOnlyList<int>? devices = null, Action<string>? progress = null,
        Qwen35ExecutionOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(gguf);
        string path = gguf.FilePath;
        if (progress is not null)
        {
            Action<string> callback = progress;
            var callbackLock = new object();
            progress = message => { lock (callbackLock) callback(message); };
        }
        options ??= new Qwen35ExecutionOptions();
        if (options.TrainingIQ2Bf16XmxForward && options.TrainingIQ2Fp16XmxForward)
            throw new ArgumentException("Choose either BF16 or FP16 IQ2_S/XMX forward, not both.", nameof(options));
        if ((options.TrainingIQ2Bf16XmxForward || options.TrainingIQ2Fp16XmxForward)
            && !options.LoraTraining)
            throw new ArgumentException("IQ2_S XMX forward is only available for LoRA training.", nameof(options));
        if (options.TrainingIQ2ProjectionCacheMiB is < 0 or > 16384)
            throw new ArgumentOutOfRangeException(nameof(options),
                "The experimental IQ2_S host projection cache cap must be 0..16384 MiB.");
        if (options.TrainingIQ2GpuProjectionCacheMiB is < 0 or > 1024)
            throw new ArgumentOutOfRangeException(nameof(options),
                "The experimental IQ2_S GPU projection cache cap per Arc must be 0..1024 MiB.");
        if (options.TrainingIQ2ProjectionCacheMiB > 0 && options.TrainingIQ2GpuProjectionCacheMiB > 0)
            throw new ArgumentException("Choose the IQ2_S host projection cache or GPU projection cache, not both.", nameof(options));
        if (options.TrainingIQ2GpuProjectionCacheMiB > 0 && !options.TrainingGpuCheckpoints)
            throw new ArgumentException("The IQ2_S GPU projection cache requires GPU training checkpoints.", nameof(options));
        if (options.TrainingFusedAttentionRows && !options.TrainingPackedAttentionScores)
            throw new ArgumentException("Fused attention rows require packed scores.", nameof(options));
        if (options.TrainingFusedAttentionOutput && !options.TrainingFusedAttentionRows)
            throw new ArgumentException("Fused attention output requires fused attention rows.", nameof(options));
        if (!Enum.IsDefined(options.QuantizedKernel) || options.QueuedKernelLimit is < 16 or > 4096
            || options.ProjectionWorkgroupSize is not (32 or 64 or 128)
            || options.InferencePairedProjectionTypes is < 0 or > 7
            || options.InferencePrefillChunkTokens is < 0 or > 1024
            || options.InferenceProjectionRows is not (1 or 2 or 4 or 8 or 16)
            || options.InferenceBufferPoolMiB is < 0 or > 2048
            || options.InferenceDeferredReleaseMiB is < 0 or > 1024
            || options.LoraReductionSize is not (16 or 128 or 256 or 512 or 1024)
            || options.TrainingTransposeRows is not (1 or 4 or 8 or 16 or 32)
            || options.TrainingTransposeOctetRows is not (0 or 4 or 8 or 16)
            || options.TrainingQ4TransposeOctetRows is not (0 or 4 or 8 or 16)
            || options.TrainingIQ3TransposeOctetRows is not (0 or 4 or 8 or 16)
            || options.TrainingQ5TransposeOctetRows is not (0 or 4 or 8 or 16)
            || options.TrainingNormSplits is < 1 or > 32
            || options.TrainingForwardRows is not (1 or 2 or 4 or 8 or 16)
            || options.TrainingQ5ForwardRows is not (1 or 4 or 8)
            || options.TrainingStreamedAttentionTileRows is not (0 or 64 or 128 or 256 or 512 or 1024)
            || options.TrainingHybridCheckpointMiBPerDevice is < 0 or > 1024
            || options.TrainingBufferPoolMiB is < 0 or > 2048)
            throw new ArgumentException("Invalid Qwen3.5 execution options.", nameof(options));
        if (options.TrainingHybridCheckpointMiBPerDevice > 0
            && options.TrainingHostCheckpointForwardCopyHandoff)
            throw new ArgumentException("Hybrid checkpoints cannot overwrite retained layer inputs through forward-copy handoff.", nameof(options));
        Qwen35GgufDescriptor d = Qwen35Gguf.Inspect(gguf);
        Qwen35PrismMetadata? prism = Qwen35PrismMetadata.Read(gguf);
        if (options.LoraTraining && (prism is not null || d.Tensors.Any(t => t.Type == Qwen2Gguf.BF16Type && IsQuantized(t))))
            throw new NotSupportedException("PQ2_0/PTQ1_0 and BF16 matrix backpropagation are not implemented; load this model for generation.");
        ArcDeviceInfo[] available = ArcDevices.Enumerate().ToArray();
        if (available.Length == 0)
            throw new NotSupportedException("Qwen3.5 quantized inference requires Intel Arc.");
        int[] selected = devices?.ToArray() ?? SelectDevices(d, available, options);
        if (selected.Length == 0 || selected.Length > d.LayerCount
            || selected.Distinct().Count() != selected.Length
            || selected.Any(index => index < 0 || index >= available.Length))
            throw new ArgumentException("Specify distinct, available Arc device indices.", nameof(devices));
        if ((options.TrainingIQ2Bf16XmxForward || options.TrainingIQ2Fp16XmxForward) && selected.Any(index =>
            !available[index].SupportsXmx || available[index].MinimumSubgroupSize != 16
            || !available[index].Extensions.Split(' ').Contains("cl_intel_subgroups")
            || (options.TrainingIQ2Fp16XmxForward && !available[index].Extensions.Split(' ').Contains("cl_khr_fp16"))))
            throw new NotSupportedException("IQ2_S XMX training forward requires Arc XMX, subgroup size 16, and FP16 support when selected.");
        bool residentIq2Panels = UseResidentIq2Panels(options, selected.Select(index => available[index]));
        long[] planned = PlanPersistentBytes(d, selected.Length, residentIq2Panels);
        for (int slot = 0; slot < selected.Length; slot++)
        {
            ArcDeviceInfo device = available[selected[slot]];
            if (planned[slot] + WorkspaceReserveBytes > DeviceBudget(device))
                throw new NotSupportedException(
                    $"Arc {device.Index} needs {planned[slot] / 1073741824.0:F2} GiB for weights and initial GPU state " +
                    "plus workspace. Select more GPUs with --devices 0,1.");
            foreach (GgufTensorInfo tensor in d.Tensors)
                if (DeviceSlot(tensor.Name, d.LayerCount, selected.Length) == slot
                    && (ulong)(IsQuantized(tensor) ? MaximumMatrixBufferBytes(tensor, residentIq2Panels) : DenseBytes(tensor)) > device.MaximumAllocationBytes)
                    throw new NotSupportedException($"Tensor '{tensor.Name}' exceeds Arc {device.Index}'s allocation limit.");
        }

        var model = new Qwen35QuantizedModel(d, options)
        {
            _prism = prism,
            _modelPath = Path.GetFullPath(path),
            // Keep the same base snapshot protected from writes/deletion while
            // resident weights and adapter identity refer to it.
            _modelSource = File.OpenRead(path)
        };
        Task<string>? fingerprint = options.ComputeModelFingerprintOnLoad
            ? Task.Run(model.ModelFingerprint) : null;
        try
        {
            ArcExecutionLane CreateLane(int index)
            {
                progress?.Invoke($"Preparing Arc {index}: {available[index].Name}");
                return new ArcExecutionLane(index, new ArcExecutionOptions
                {
                    Qwen35InferenceKernelsOnly = true,
                    Qwen35ResidentIq2Panels = residentIq2Panels,
                    Qwen35GgufBslmPrefill = !options.LoraTraining && options.InferenceXmxGgufBslmPrefill,
                    Qwen35TrainingKernels = options.LoraTraining,
                    Qwen35CooperativeLora = options.LoraTraining ? options.TrainingCooperativeLora : options.InferenceCooperativeLora,
                    Qwen35ProjectionWorkgroupSize = options.LoraTraining ? 32 : options.ProjectionWorkgroupSize,
                    Qwen35PairedProjection = !options.LoraTraining && options.InferencePairedProjection,
                    Qwen35PairedProjectionTypes = options.LoraTraining ? 0 : options.InferencePairedProjectionTypes,
                    Qwen35PairedLoraProjection = !options.LoraTraining && options.InferencePairedLoraProjection,
                    CacheKernelArguments = !options.LoraTraining && options.CacheKernelArguments,
                    CacheProgramBinary = options.CacheProgramBinary,
                    Qwen35FastRmsNorm = !options.LoraTraining && options.InferenceFastRmsNorm,
                    Qwen35ParallelArgmax = !options.LoraTraining && options.ParallelArgmax,
                    Qwen35ParallelDeltaNorm = !options.LoraTraining && options.ParallelDeltaNorm,
                    Qwen35DeltaSubgroupRms = !options.LoraTraining && options.InferenceBatchRecurrent
                        && options.InferenceSubgroupRecurrentRms && available[index].MinimumSubgroupSize == 16
                        && available[index].Extensions.Split(' ').Contains("cl_intel_subgroups"),
                    Qwen35LoraReductionSize = options.LoraTraining ? 128 : options.LoraReductionSize,
                    Qwen35UnrollQ4 = !options.LoraTraining && options.UnrollQ4,
                    Qwen35NativeHalfScale = options.NativeHalfScale,
                    CollectKernelTimings = options.CollectKernelTimings || options.DetailedProfiling,
                    Qwen35CooperativeDelta = options.TrainingCooperativeDelta,
                    DetailedProfiling = options.DetailedProfiling,
                    BufferPoolBytes = (long)(options.LoraTraining ? options.TrainingBufferPoolMiB : options.InferenceBufferPoolMiB) * 1024 * 1024,
                    DeferredReleaseBytes = options.LoraTraining ? 0 : (long)options.InferenceDeferredReleaseMiB * 1024 * 1024,
                    QueuedKernelLimit = options.QueuedKernelLimit,
                    PhysicalBufferBudgetBytes = DeviceBudget(available[index])
                });
            }
            if (options.ParallelModelLoad && selected.Length > 1)
            {
                Task<ArcExecutionLane>[] creating = selected.Select(index => Task.Run(() => CreateLane(index))).ToArray();
                try { Task.WhenAll(creating).GetAwaiter().GetResult(); }
                finally
                {
                    // Even if another device fails, completed lanes must be owned
                    // by the model so the outer failure path disposes them.
                    foreach (var task in creating)
                        if (task.IsCompletedSuccessfully) model._lanes.Add(task.Result);
                }
            }
            else foreach (int index in selected) model._lanes.Add(CreateLane(index));
            if (model._prism is not null)
            {
                foreach (ArcExecutionLane lane in model._lanes)
                foreach (var pair in model._prism.Signs)
                {
                    model._prismSigns.Add((lane, pair.Key), lane.Upload(pair.Value));
                    model._auxiliaryBytes[lane] = model._auxiliaryBytes.GetValueOrDefault(lane) + 4L * pair.Value.Length;
                }
                progress?.Invoke($"Prism: GPU Hadamard 1024, {model._prism.ForwardWeights.Count} projections, {model._prism.InverseWeights.Count} inverse embeddings, grouped GDN={model._prism.GroupedValueHeads}");
            }
            int loaded = 0;
            void LoadTensor(GgufTensorInfo tensor)
            {
                var lane = model._lanes[DeviceSlot(tensor.Name, d.LayerCount, selected.Length)];
                if (IsQuantized(tensor))
                {
                    // UploadRaw is blocking; no model-sized host payload is retained.
                    var matrix = new Matrix(lane, tensor,
                        gguf.ReadTensorBytes(tensor, EncodedBytes(tensor)), options.QuantizedKernel,
                        options.TrainingTransposeRows, options.LoraTraining ? options.TrainingForwardRows : options.InferenceProjectionRows,
                        options.LoraTraining ? options.TrainingQ5ForwardRows : 1,
                        options.TrainingIQ2Bf16XmxForward,
                        options.TrainingIQ2Fp16XmxForward,
                        options.TrainingIQ2RollingTranspose,
                        tensor.Type switch {
                            Qwen35Gguf.IQ2SType => options.TrainingTransposeOctetRows,
                            Qwen2Gguf.Q4KType => options.TrainingQ4TransposeOctetRows,
                            Qwen35Gguf.IQ3SType => options.TrainingIQ3TransposeOctetRows,
                            Qwen35Gguf.Q5KType => options.TrainingQ5TransposeOctetRows,
                            _ => 0 }, inferenceXmxPrefill: !options.LoraTraining && options.InferenceXmxPrefill,
                        inferenceXmxPackedPrefill: !options.LoraTraining && options.InferenceXmxPackedPrefill,
                        inferenceXmxFactoredPrefill: !options.LoraTraining && options.InferenceXmxFactoredPrefill,
                        inferenceResidentIq2Panels: residentIq2Panels,
                        inferenceXmxGgufBslmPrefill: !options.LoraTraining && options.InferenceXmxGgufBslmPrefill,
                        iq2TiledForward: options.IQ2TiledForward, iq2TiledBackward: options.IQ2TiledBackward,
                        inferenceIq2DecodePair4: !options.LoraTraining && options.InferenceIq2DecodePair4,
                        inferenceIq2DecodeLevel3: !options.LoraTraining && options.InferenceIq2DecodeLevel3,
                        inferencePackedIq2Decode: !options.LoraTraining && options.InferencePackedIq2Decode);
                    lock (model._matrices) model._matrices.Add(tensor.Name, matrix);
                }
                else
                {
                    model._dense.Add(tensor.Name, lane.Upload(Qwen2Gguf.ReadTensor(gguf, tensor)));
                    model._auxiliaryBytes[lane] = model._auxiliaryBytes.GetValueOrDefault(lane) + DenseBytes(tensor);
                }
                int count = Interlocked.Increment(ref loaded);
                if (count % 100 == 0) progress?.Invoke($"Loading tensors: {count}/{d.Tensors.Count}");
            }
            if (options.ParallelModelLoad && selected.Length > 1)
            {
                // Dense conversion uses the reader's sequential stream. Packed
                // reads are positional and each worker owns one device queue.
                foreach (var tensor in d.Tensors.Where(t => !IsQuantized(t))) LoadTensor(tensor);
                Parallel.For(0, selected.Length, slot =>
                {
                    foreach (var tensor in d.Tensors.Where(t => IsQuantized(t)
                        && DeviceSlot(t.Name, d.LayerCount, selected.Length) == slot).OrderBy(t => t.Offset))
                        LoadTensor(tensor);
                });
            }
            else foreach (var tensor in d.Tensors) LoadTensor(tensor);
            foreach (ArcExecutionLane lane in model._lanes)
            {
                int width = model._matrices.Values.Where(matrix => ReferenceEquals(matrix.Lane, lane))
                    .Max(matrix => matrix.OutputWidth);
                ArcBuffer zeroBias = lane.Allocate(width);
                model._zeroBias.Add(lane, zeroBias);
                lane.Run("q35a_zero", width, 0, zeroBias, width);
            }
            model._embedding = model._matrices["token_embd.weight"];
            model._outputNorm = model._dense["output_norm.weight"];
            for (int layer = 0; layer < d.LayerCount; layer++)
            {
                var lane = model._lanes[DeviceSlot($"blk.{layer}.", d.LayerCount, selected.Length)];
                var state = new LayerState(lane, d, d.IsRecurrent(layer));
                model._states.Add(state);
                state.Initialize();
                model._layerBindings.Add(new LayerBindings(model, layer, state));
            }
            model.InitializeSplitOutputHead(gguf, progress);
            for (int slot = 0; slot < selected.Length; slot++)
            {
                model._lanes[slot].Synchronize();
                progress?.Invoke($"Arc {selected[slot]}: quantized weights = {model.ResidentWeightBytes[slot]} bytes " +
                    $"({model.ResidentWeightBytes[slot] / 1073741824.0:F2} GiB), " +
                    $"GPU state = {model.ResidentStateBytes[slot]} bytes, " +
                    $"live device allocation = {model._lanes[slot].AllocatedBytes} bytes");
            }
            if (fingerprint is not null) fingerprint.GetAwaiter().GetResult();
            return model;
        }
        catch
        {
            // The fingerprint reads the protected source handle: join it before
            // disposing that handle, without masking the original load error.
            try { fingerprint?.GetAwaiter().GetResult(); } catch { }
            model.Dispose(); throw;
        }
    }

    internal static long[] PlanWeightBytes(Qwen35GgufDescriptor d, int deviceCount, bool inferenceResidentIq2Panels = false)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(deviceCount);
        var bytes = new long[deviceCount];
        foreach (GgufTensorInfo tensor in d.Tensors.Where(IsQuantized))
            bytes[DeviceSlot(tensor.Name, d.LayerCount, deviceCount)] += PlannedMatrixBytes(tensor, inferenceResidentIq2Panels);
        return bytes;
    }

    private static long[] PlanPersistentBytes(Qwen35GgufDescriptor d, int devices, bool inferenceResidentIq2Panels = false)
    {
        long[] bytes = PlanWeightBytes(d, devices, inferenceResidentIq2Panels);
        var maxOutput = new int[devices];
        foreach (GgufTensorInfo tensor in d.Tensors)
        {
            int slot = DeviceSlot(tensor.Name, d.LayerCount, devices);
            if (IsQuantized(tensor)) maxOutput[slot] = Math.Max(maxOutput[slot], checked((int)tensor.Shape[1]));
            else bytes[slot] = checked(bytes[slot] + DenseBytes(tensor));
        }
        for (int i = 0; i < devices; i++) bytes[i] += 4L * maxOutput[i];
        for (int layer = 0; layer < d.LayerCount; layer++)
        {
            int slot = DeviceSlot($"blk.{layer}.", d.LayerCount, devices);
            long stateBytes = d.IsRecurrent(layer)
                ? checked(4L * d.LinearValueHeads * d.LinearHeadWidth * d.LinearHeadWidth
                    + Math.Max(4L, 4L * (2 * d.LinearKeyHeads + d.LinearValueHeads) * d.LinearHeadWidth * (d.ConvKernel - 1)))
                : checked(8L * d.KvHeadCount * d.HeadWidth * Math.Min(16, d.ContextLength));
            bytes[slot] = checked(bytes[slot] + stateBytes);
        }
        return bytes;
    }

    private static int[] SelectDevices(Qwen35GgufDescriptor d, ArcDeviceInfo[] available, Qwen35ExecutionOptions options)
    {
        for (int count = 1; count <= Math.Min(available.Length, d.LayerCount); count++)
        {
            long[] bytes = PlanPersistentBytes(d, count, UseResidentIq2Panels(options, available.Take(count)));
            if (bytes.Select((value, slot) => value + WorkspaceReserveBytes <= DeviceBudget(available[slot])).All(fits => fits))
                return Enumerable.Range(0, count).ToArray();
        }
        throw new NotSupportedException("The Qwen3.5 weights and initial GPU state exceed the available Arc VRAM budget.");
    }

    private static long DeviceBudget(ArcDeviceInfo device) => checked((long)(device.GlobalMemoryBytes / 10 * 9));

    private static int DeviceSlot(string name, int layers, int devices)
    {
        if (name.StartsWith("blk.", StringComparison.Ordinal))
        {
            int end = name.IndexOf('.', 4);
            int layer = int.Parse(name.AsSpan(4, end - 4));
            return Math.Min(devices - 1, layer * devices / layers);
        }
        return name == "token_embd.weight" ? 0 : devices - 1;
    }

    private static bool IsQuantized(GgufTensorInfo tensor)
        => tensor.Shape.Count == 2 && !tensor.Name.EndsWith(".ssm_conv1d.weight", StringComparison.Ordinal)
            && Qwen35Gguf.IsSupportedMatrixStorage(tensor.Type);
    private static int EncodedBytes(GgufTensorInfo tensor)
        => checked((int)(tensor.Shape[0] / (ulong)Qwen35Gguf.QuantizedBlockElements(tensor.Type) * tensor.Shape[1]
            * (ulong)Qwen35Gguf.QuantizedBlockBytes(tensor.Type)));
    private static bool CanUseResidentIq2Panels(GgufTensorInfo tensor)
        => tensor.Type == Qwen35Gguf.IQ2SType && tensor.Name.StartsWith("blk.", StringComparison.Ordinal)
            && tensor.Shape[0] % 256 == 0;
    private static bool UseResidentIq2Panels(Qwen35ExecutionOptions options, IEnumerable<ArcDeviceInfo> devices)
        => !options.LoraTraining && options.InferenceResidentIq2Panels && !options.InferenceXmxGgufBslmPrefill
            && options.QuantizedKernel is Qwen35QuantizedKernel.Auto or Qwen35QuantizedKernel.Subgroup
            && devices.All(device => device.SupportsXmx && device.MinimumSubgroupSize == 16
                && device.Extensions.Split(' ').Contains("cl_intel_subgroups")
                && device.Extensions.Split(' ').Contains("cl_khr_fp16"));
    private static long PlannedMatrixBytes(GgufTensorInfo tensor, bool residentIq2Panels)
        => residentIq2Panels && CanUseResidentIq2Panels(tensor)
            ? checked((long)tensor.Shape[0] * (((long)tensor.Shape[1] + 15) / 16 * 16) / 256 * 106)
            : EncodedBytes(tensor);
    private static long MaximumMatrixBufferBytes(GgufTensorInfo tensor, bool residentIq2Panels)
        => residentIq2Panels && CanUseResidentIq2Panels(tensor)
            ? checked((long)tensor.Shape[0] * (((long)tensor.Shape[1] + 15) / 16 * 16) / 256 * 96)
            : EncodedBytes(tensor);
    private static long DenseBytes(GgufTensorInfo tensor)
        => checked(tensor.Shape.Aggregate(1L, (count, dimension) => checked(count * (long)dimension)) * 4);
}
