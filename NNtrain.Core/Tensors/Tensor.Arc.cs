using NNtrain.Arc;
using NNtrain.Runtime.Execution;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

partial class Tensor
{
    private int _arcDeviceIndex;

    // Backward allocates only the current node and its destinations. In particular,
    // the forward pass must not preallocate FP32 gradients for every BFP8 activation.
    internal void PrepareArcBackward()
    {
        if (ArcResident) return;
        EnsureGradientBuffer();
        foreach (Tensor parent in Node.Parents) parent.EnsureGradientBuffer();
    }

    internal void ReleaseArcIntermediateGradient() { if (ArcResident) ReleaseArcGraph(); else _grad = []; }

    /// <summary>Owns an Arc lane and its precision contract; dispose after backward/optimizer completes.</summary>
    public static IDisposable BeginArcExecution(int deviceIndex = 0, TensorPrecisionMode precision = TensorPrecisionMode.Float32,
        NNtrain.Arc.ArcExecutionOptions? options = null)
    {
        ValidateArcPrecision(precision);
        var session = new ExecutionSession(new ExecutionOptions {
            Device = ExecutionDeviceKind.Arc, ArcDeviceIndex = deviceIndex,
            Precision = PrecisionPolicy.Parse(TensorPrecisionModeNames.Format(precision)),
            RequireDeviceResidency = false });
        try {
            var lane = new ArcExecutionLane(deviceIndex, options);
            try { session.AttachLane(lane); } catch { lane.Dispose(); throw; }
            return new ArcScope(session, session.Enter());
        }
        catch { session.Dispose(); throw; }
    }

    /// <summary>
    /// Owns two Arc lanes for model-parallel inference. Training sessions
    /// attach their own lanes through the production session factory.
    /// </summary>
    public static IDisposable BeginArcInferenceExecution(
        IReadOnlyList<int> deviceIndices,
        TensorPrecisionMode precision = TensorPrecisionMode.Float32,
        ArcExecutionOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(deviceIndices);
        ValidateArcPrecision(precision);
        DeviceSet devices = new(deviceIndices);
        if (devices.Count != 2)
            throw new ArgumentException("Arc inference requires exactly two GPUs.", nameof(deviceIndices));

        IReadOnlyList<ArcDeviceInfo> available = ArcDevices.Enumerate();
        foreach (int index in devices)
        {
            if (index >= available.Count)
            {
                throw new InvalidOperationException(
                    $"Intel Arc device {index} is unavailable. Found {available.Count} Arc OpenCL GPUs.");
            }
        }

        var session = new ExecutionSession(new ExecutionOptions
        {
            Device = ExecutionDeviceKind.Arc,
            ArcDeviceIndex = devices[0],
            ArcDevices = devices,
            Precision = PrecisionPolicy.Parse(TensorPrecisionModeNames.Format(precision)),
            RequireDeviceResidency = false,
        });
        IDisposable? sessionScope = null;
        try
        {
            foreach (int index in devices)
            {
                var lane = new ArcExecutionLane(index, options);
                try { session.AttachLane(lane); }
                catch { lane.Dispose(); throw; }
            }
            sessionScope = session.Enter();
            IDisposable deviceScope = TensorExecutionContext.Push(
                new TorchDevice(TensorDevice.Arc, devices[0]));
            return new ArcScope(session, sessionScope, deviceScope);
        }
        catch
        {
            try { sessionScope?.Dispose(); }
            finally { session.Dispose(); }
            throw;
        }
    }

    internal static void ValidateArcPrecision(TensorPrecisionMode precision)
    {
        if (precision is not (TensorPrecisionMode.Float32 or TensorPrecisionMode.Mix16_32 or TensorPrecisionMode.Mix8_32 or TensorPrecisionMode.Mix8_16))
            throw new NotSupportedException("Arc currently supports float32, mix16_32, mix8_32 and mix8_16; pure low-precision arithmetic is not implemented.");
    }

    private sealed class ArcScope(ExecutionSession session, IDisposable scope, IDisposable? deviceScope = null) : IDisposable
    {
        private bool _disposed;
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            try { deviceScope?.Dispose(); }
            finally { try { scope.Dispose(); } finally { session.Dispose(); } }
        }
    }

    internal static ArcExecutionLane ArcLane => ExecutionSession.Current?.GetRequiredLane(
        ExecutionDeviceKind.Arc, TensorExecutionContext.Device.Index) as ArcExecutionLane
        ?? throw new InvalidOperationException("Arc execution requires an ExecutionSession or Tensor.BeginArcExecution().");

    private float[] ArcValues(bool matrixOperand = false)
    {
        float[] values = Data.ToArray();
        if (matrixOperand && DType == TensorDType.Bfp8)
            for (int i = 0; i < values.Length; i++) values[i] = TensorStorageCodec.RoundToBFloat16(values[i]);
        return values;
    }

    private ArcBuffer ArcUploadValues(bool matrixOperand = false)
    {
        if (ArcResident) return ArcResidentValues(matrixOperand);
        var lane = ArcLane;
        if (!lane.Options.PackedUploads) return lane.Upload(ArcValues(matrixOperand));
        EnsureHostDataCurrent();
        if (_data.TryGetFloat32Buffer(out float[] floats)) return lane.Upload(floats);
        var output = lane.Allocate(Numel);
        try
        {
            if (_data.TryGetBfp8Buffers(out sbyte[] payload, out float[] scales, out var descriptor))
            {
                using var bytes = lane.UploadRaw(payload);
                using var scale = lane.Upload(scales);
                int block = descriptor.Granularity == Bfp8ScaleGranularity.Tensor ? Numel : descriptor.BlockSize;
                lane.Run("decode_bfp8", Numel, 0, bytes, scale, output, Numel, block, matrixOperand ? 1 : 0);
            }
            else if (_data.TryGetBFloat16Buffer(out ushort[] bf16))
            {
                using var bytes = lane.UploadRaw(bf16);
                lane.Run("decode_bf16", Numel, 0, bytes, output, Numel);
            }
            else lane.Write(output, ArcValues(matrixOperand));
            return output;
        }
        catch { output.Dispose(); throw; }
    }

    private static Tensor ArcResult(float[] values, int[] shape, Tensor[] parents, TensorDType? dtype = null)
    {
        TensorDType format = dtype ?? TensorDTypeContract.Promote(parents);
        Tensor result = format == TensorDType.Bfp8
            ? FromStorageResult(TensorStorage.CreateBfp8(values, SelectBfp8ResultDescriptor(parents)), shape, parents)
            : new Tensor(values, shape, parents, dtype: format);
        result._device = TensorDevice.Arc;
        result._arcDeviceIndex = TensorExecutionContext.Device.Index;
        return result;
    }

    private Tensor ArcLinear(Tensor weight, Tensor bias, bool relu, TensorDType? outputDType = null)
    {
        if (ArcResident) return ArcResidentLinear(weight, bias, relu, outputDType);
        int ni = _shape[^1], no = weight._shape[0], rows = Numel / ni;
        if (weight.Rank != 2 || weight._shape[1] != ni || bias.Rank != 1 || bias.Numel != no)
            throw new ArgumentException("Arc linear dimensions do not match.");
        float[] values = new float[checked(rows * no)];
        if (ArcLane.Options.TiledMatrices)
        {
            using var input = ArcUploadValues(true);
            using var weights = weight.ArcUploadValues(true);
            using var biases = bias.ArcUploadValues();
            using var output = ArcLane.Allocate(values.Length);
            ArcMuonMath.Gemm(ArcLane, input, weights, output, rows, no, ni, tb: true, bias: biases, relu: relu);
            ArcLane.Read(output, values);
        }
        else ArcLane.Run("linear", values.Length, 0, In(ArcValues(true)), In(weight.ArcValues(true)), In(bias.ArcValues()), InOut(values), rows, ni, no, relu ? 1 : 0);
        int[] shape = (int[])_shape.Clone(); shape[^1] = no;
        Tensor result = ArcResult(values, shape, [this, weight, bias], outputDType);
        result.Node.BackwardAction = () => {
            bool mixed = ArcUsesMixedMatrixOperands && DType != TensorDType.Float32 && weight.DType != TensorDType.Float32;
            float[] upstream = result._grad;
            if (mixed)
            {
                float[] mask = relu ? result.ArcValues() : [];
                upstream = upstream.Select((v, i) => TensorStorageCodec.RoundToBFloat16(relu && mask[i] <= 0 ? 0 : v)).ToArray();
            }
            bool maskRelu = relu && !mixed;
            if (ArcLane.Options.TiledMatrices)
            {
                using var input = ArcUploadValues(true);
                using var weights = weight.ArcUploadValues(true);
                using var dy = ArcLane.Upload(upstream);
                using var dx = ArcLane.Upload(_grad);
                using var dw = ArcLane.Upload(weight._grad);
                using var db = ArcLane.Upload(bias._grad);
                using var gate = maskRelu ? result.ArcUploadValues() : null;
                ArcMuonMath.Gemm(ArcLane, dy, weights, dx, rows, ni, no, bf16: mixed ? 3 : 0, accumulate: true, gate: gate, gateOperand: maskRelu ? 1 : 0);
                ArcMuonMath.Gemm(ArcLane, dy, input, dw, no, ni, rows, ta: true, bf16: mixed ? 3 : 0, accumulate: true, gate: gate, gateOperand: maskRelu ? 1 : 0);
                for (int start = 0; start < rows; start += 2048)
                    ArcLane.Run("linear_db_chunk", no, 0, dy, gate ?? dy, db, rows, no, maskRelu ? 1 : 0, start, Math.Min(2048, rows - start));
                ArcLane.Read(dx, _grad); ArcLane.Read(dw, weight._grad); ArcLane.Read(db, bias._grad);
                return;
            }
            ArcLane.Run("linear_dx", Numel, 0, In(weight.ArcValues(true)), In(upstream), In(result.ArcValues()), InOut(_grad), rows, ni, no, maskRelu ? 1 : 0);
            ArcLane.Run("linear_dw", weight.Numel, 0, In(ArcValues(true)), In(upstream), In(result.ArcValues()), InOut(weight._grad), InOut(bias._grad), rows, ni, no, maskRelu ? 1 : 0);
        };
        return result;
    }

    private Tensor ArcEmbedding(Tensor? positions, int[] indices, int[] shape, int sequence)
    {
        if (ArcResident) return ArcResidentEmbedding(positions, indices, shape, sequence);
        int width = _shape[^1]; float[] y = new float[checked(indices.Length * width)];
        ArcLane.Run("embedding", y.Length, 0, In(ArcValues()), In(positions?.ArcValues() ?? [0f]), In(indices), InOut(y), indices.Length, width, sequence, positions is null ? 0 : 1);
        Tensor result = ArcResult(y, shape, positions is null ? [this] : [this, positions]);
        result.Node.BackwardAction = () => ArcLane.Run("embedding_back", y.Length, 0, In(result._grad), In(indices), InOut(_grad), InOut(positions?._grad ?? new float[1]), indices.Length, width, sequence, positions is null ? 0 : 1);
        return result;
    }

    private Tensor ArcDropout(Tensor? residual, uint seed, uint threshold, float scale)
    {
        if (ArcResident) return ArcResidentDropout(residual, seed, threshold, scale);
        float[] y = new float[Numel];
        ArcLane.Run("dropout", Numel, 0, In(ArcValues()), In(residual?.ArcValues() ?? [0f]), InOut(y), Numel, seed, threshold, scale, residual is null ? 0 : 1);
        Tensor result = ArcResult(y, _shape, residual is null ? [this] : [this, residual]);
        result.Node.BackwardAction = () => {
            ArcLane.Run("dropout_back", Numel, 0, In(result._grad), InOut(_grad), Numel, seed, threshold, scale);
            if (residual is not null) ArcAccumulate(result._grad, residual._grad);
        };
        return result;
    }

    private Tensor ArcNorm(Tensor gamma, Tensor beta, float eps, Tensor? branch = null, float probability = 0, Random? random = null)
    {
        if (ArcResident) return ArcResidentNorm(gamma, beta, eps, branch, probability, random);
        if (ArcLane.Options.FusedNormalization) return ArcNormDevice(gamma, beta, eps, branch, probability, random);
        int width = _shape[^1], rows = Numel / width;
        if (gamma.Numel != width || beta.Numel != width) throw new ArgumentException("LayerNorm parameter dimensions do not match.");
        float[] x = ArcValues(); uint seed = probability == 0 ? 0 : NextDropoutSeed(random ?? Random.Shared);
        uint threshold = (uint)(probability * (uint.MaxValue + 1d)); float scale = 1f / (1f - probability);
        if (branch is not null)
        {
            if (!_shape.AsSpan().SequenceEqual(branch._shape)) throw ShapeMismatch(this, branch, "Arc residual");
            float[] sum = new float[Numel];
            ArcLane.Run("dropout", Numel, 0, In(branch.ArcValues()), In(x), InOut(sum), Numel, seed, threshold, scale, 1);
            x = sum;
        }
        float[] y = new float[Numel], stats = new float[checked(rows * 2)];
        ArcLane.Run("norm", rows, 0, In(x), In(gamma.ArcValues()), In(beta.ArcValues()), InOut(y), InOut(stats), rows, width, eps);
        Tensor result = ArcResult(y, _shape, branch is null ? [this, gamma, beta] : [this, branch, gamma, beta]);
        result.Node.BackwardAction = () => {
            float[] dx = new float[Numel];
            ArcLane.Run("norm_dx", rows, 0, In(x), In(gamma.ArcValues()), In(result._grad), In(stats), InOut(dx), rows, width);
            ArcLane.Run("norm_dw", width, 0, In(x), In(result._grad), In(stats), InOut(gamma._grad), InOut(beta._grad), rows, width);
            ArcAccumulate(dx, _grad);
            if (branch is not null) ArcLane.Run("dropout_back", Numel, 0, In(dx), InOut(branch._grad), Numel, seed, threshold, scale);
        };
        return result;
    }

    private Tensor ArcAttention(int batch, int sequence, int width, int heads, bool causal)
    {
        if (ArcResident) return ArcResidentAttention(batch, sequence, width, heads, causal);
        int groups = checked(batch * heads * sequence), outputLength = checked(batch * sequence * width);
        if (ArcLane.Options.StreamingAttention)
        {
            float[] output = new float[outputLength], stats = new float[checked(groups * 2)];
            using (var input = ArcUploadValues(true))
                ArcLane.Run("attention_streaming", (long)groups * 64, 64, input, Out(output), Out(stats),
                    batch, sequence, width, heads, causal ? 1 : 0, new LocalMemory(checked((sequence + 64) * 4)));
            Tensor streaming = ArcResult(output, Rank == 3 ? [batch, sequence, width] : [sequence, width], [this]);
            streaming.Node.BackwardAction = () => {
                using var input = ArcUploadValues(true);
                using var dy = ArcLane.Upload(streaming._grad);
                using var stat = ArcLane.Upload(stats);
                using var dx = ArcLane.Upload(_grad);
                using var delta = ArcLane.Allocate(groups);
                ArcLane.Run("attention_streaming_dq", (long)groups * 64, 64, input, dy, stat, dx, delta,
                    batch, sequence, width, heads, causal ? 1 : 0, new LocalMemory(checked((sequence * 2 + 64) * 4)));
                ArcLane.Run("attention_streaming_dkv", (long)groups * 64, 64, input, dy, stat, delta, dx,
                    batch, sequence, width, heads, causal ? 1 : 0, new LocalMemory(checked(sequence * 2 * 4)));
                ArcLane.Read(dx, _grad);
            };
            return streaming;
        }
        float[] y = new float[outputLength];
        using (var probabilities = ArcLane.Allocate(checked(groups * sequence)))
            ArcLane.Run("attention", (long)groups * 64, 64, In(ArcValues(true)), InOut(y), probabilities, batch, sequence, width, heads, causal ? 1 : 0, new LocalMemory(checked(sequence * 4)));
        Tensor result = ArcResult(y, Rank == 3 ? [batch, sequence, width] : [sequence, width], [this]);
        result.Node.BackwardAction = () => {
            using var probabilities = ArcLane.Allocate(checked(groups * sequence));
            using var ds = ArcLane.Allocate(checked(groups * sequence));
            using var scratchOutput = ArcLane.Allocate(outputLength);
            float[] input = ArcValues(true);
            ArcLane.Run("attention", (long)groups * 64, 64, In(input), scratchOutput, probabilities, batch, sequence, width, heads, causal ? 1 : 0, new LocalMemory(checked(sequence * 4)));
            ArcLane.Run("attention_ds", (long)groups * 64, 64, In(input), In(result._grad), probabilities, ds, sequence, width, heads, new LocalMemory(checked((sequence + 1) * 4)));
            ArcLane.Run("attention_dq", outputLength, 0, In(input), ds, InOut(_grad), batch, sequence, width, heads);
            ArcLane.Run("attention_dkv", outputLength, 0, In(input), In(result._grad), probabilities, ds, InOut(_grad), batch, sequence, width, heads);
        };
        return result;
    }

    private Tensor ArcCrossEntropy(int[] labels, int rows, int columns, int ignore, int valid, float smoothing)
    {
        if (ArcResident) return ArcResidentCrossEntropy(labels, rows, columns, ignore, valid, smoothing);
        float[] stats = new float[checked(2 * rows)], losses = new float[rows], total = new float[1];
        ArcLane.Run("cross_entropy", rows, 0, In(ArcValues()), In(labels), InOut(stats), InOut(losses), rows, columns, ignore, valid, smoothing);
        total[0] = ArcTrainingMath.Sum(losses);
        Tensor result = ArcResult(total, [1], [this], TensorDType.Float32);
        result.Node.BackwardAction = () => ArcLane.Run("cross_entropy_back", Numel, 0, In(ArcValues()), In(labels), In(stats), InOut(_grad), rows, columns, ignore, valid, smoothing, result._grad[0]);
        return result;
    }

    internal static void ArcAccumulate(float[] source, float[] destination, float scale = 1f, bool add = true)
    {
        if (!add && ReferenceEquals(source, destination) && ArcLane.Options.TiledMatrices)
            ArcLane.Run("scale_in_place", source.Length, 0, InOut(destination), source.Length, scale);
        else ArcLane.Run("copy_scale", source.Length, 0, In(source), InOut(destination), source.Length, scale, add ? 1 : 0);
    }

    private Tensor ArcReshape(int[] shape)
    {
        if (ArcResident) return ArcResidentSlice(0, Numel, shape);
        // Shape-only host metadata shares the immutable packed payload, not a CPU numerical operation.
        Tensor result = FromStorageResult(_data, shape, [this]);
        result._device = TensorDevice.Arc; result._arcDeviceIndex = TensorExecutionContext.Device.Index;
        result.Node.BackwardAction = () => ArcAccumulate(result._grad, _grad);
        return result;
    }
}
