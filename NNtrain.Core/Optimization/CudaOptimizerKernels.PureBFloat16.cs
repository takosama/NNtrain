namespace NNtrain;

internal static partial class CudaOptimizerKernels
{
    internal sealed class NekoMuonBFloat16ResidentState : IDisposable
    {
        // Managed arrays are checkpoint shadows. Device BF16 buffers are the
        // sole hot-step authority after GetOrCreate until explicit capture.
        private readonly float[] _fastHost;
        private readonly float[] _slowHost;
        private readonly float _initialConfidence;
        private readonly Dictionary<int, NekoBuffers> _buffers = [];
        private int _deviceConfidenceAuthoritative;

        internal NekoMuonBFloat16ResidentState(
            float[] fastHost,
            float[] slowHost,
            float initialConfidence)
        {
            _fastHost = fastHost;
            _slowHost = slowHost;
            _initialConfidence = initialConfidence;
        }

        internal bool IsDeviceConfidenceAuthoritative
            => Volatile.Read(ref _deviceConfidenceAuthoritative) != 0;

        internal void MarkDeviceConfidenceAuthoritative()
            => Volatile.Write(ref _deviceConfidenceAuthoritative, 1);

        internal NekoBuffers GetOrCreate(int deviceIndex)
        {
            if (_buffers.TryGetValue(deviceIndex, out NekoBuffers? buffers))
                return buffers;
            NativeCudaDevice accelerator =
                ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
            buffers = AllocateCudaResources(own =>
            {
                NativeCudaBuffer<ushort> fast = AllocateMoment(
                    accelerator, _fastHost);
                own(fast);
                NativeCudaBuffer<ushort> slow = AllocateMoment(
                    accelerator, _slowHost);
                own(slow);
                NativeCudaBuffer<float> stats =
                    accelerator.Allocate1D<float>(4);
                own(stats);
                NativeCudaBuffer<float> confidence =
                    AllocateStateBuffer(accelerator, [_initialConfidence]);
                own(confidence);
                var created = new NekoBuffers(
                    fast, slow, stats, confidence, new float[4]);
                _buffers.Add(deviceIndex, created);
                return created;
            });
            return buffers;
        }

        internal void SynchronizeHost(int deviceIndex)
        {
            if (!_buffers.TryGetValue(deviceIndex, out NekoBuffers? buffers))
                return;
            DecodeMoment(buffers.Fast, _fastHost);
            DecodeMoment(buffers.Slow, _slowHost);
        }

        internal float SynchronizeConfidence(int deviceIndex)
        {
            if (!_buffers.TryGetValue(deviceIndex, out NekoBuffers? buffers))
                return _initialConfidence;
            Span<float> value = stackalloc float[1];
            buffers.Confidence.CopyToCPU(value);
            return value[0];
        }

        internal NativeCudaBuffer<ushort> GetFast(int deviceIndex)
            => GetOrCreate(deviceIndex).Fast;

        internal NativeCudaBuffer<ushort> GetSlow(int deviceIndex)
            => GetOrCreate(deviceIndex).Slow;

        public void Dispose()
        {
            List<Exception>? failures = null;
            foreach (NekoBuffers buffers in _buffers.Values)
            {
                try
                {
                    buffers.Dispose();
                }
                catch (Exception exception)
                {
                    (failures ??= []).Add(exception);
                }
            }
            _buffers.Clear();
            if (failures is not null)
            {
                throw new AggregateException(
                    "NekoMuon BF16 state cleanup failed.", failures);
            }
        }

        private static NativeCudaBuffer<ushort> AllocateMoment(
            NativeCudaDevice accelerator,
            ReadOnlySpan<float> source)
        {
            var encoded = new ushort[source.Length];
            TensorStorageCodec.EncodeBFloat16(source, encoded);
            return accelerator.Allocate1D(encoded);
        }

        private static void DecodeMoment(
            NativeCudaBuffer<ushort> source,
            Span<float> destination)
        {
            var encoded = new ushort[destination.Length];
            source.CopyToCPU(encoded);
            TensorStorageCodec.DecodeBFloat16(encoded, destination);
        }

        internal sealed class NekoBuffers(
            NativeCudaBuffer<ushort> fast,
            NativeCudaBuffer<ushort> slow,
            NativeCudaBuffer<float> stats,
            NativeCudaBuffer<float> confidence,
            float[] statsHost) : IDisposable
        {
            internal NativeCudaBuffer<ushort> Fast { get; } = fast;
            internal NativeCudaBuffer<ushort> Slow { get; } = slow;
            internal NativeCudaBuffer<float> Stats { get; } = stats;
            internal NativeCudaBuffer<float> Confidence { get; } = confidence;
            internal float[] StatsHost { get; } = statsHost;

            public void Dispose()
            {
                Fast.Dispose();
                Slow.Dispose();
                Stats.Dispose();
                Confidence.Dispose();
            }
        }
    }

    internal sealed class NekoMuonBFloat16StatsBatch : IDisposable
    {
        private readonly int _deviceIndex;
        private readonly NekoMuonBFloat16ResidentState.NekoBuffers[] _states;
        private readonly NativeCudaBuffer<nint> _sourcePointers;
        private readonly NativeCudaBuffer<float> _packed;
        private readonly float[] _host;

        internal NekoMuonBFloat16StatsBatch(
            int deviceIndex,
            IReadOnlyList<NekoMuonBFloat16ResidentState> states)
        {
            _deviceIndex = deviceIndex;
            NativeCudaDevice accelerator =
                ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
            _states = states.Select(state => state.GetOrCreate(deviceIndex))
                .ToArray();
            nint[] pointers = _states.Select(state => state.Stats.NativePtr)
                .ToArray();
            _sourcePointers = accelerator.Allocate1D(pointers);
            _packed = accelerator.Allocate1D<float>(checked(states.Count * 4));
            _host = new float[checked(states.Count * 4)];
        }

        internal void GatherAndRead()
        {
            CudaOptimizerNative.GatherStats(
                _deviceIndex,
                _sourcePointers.NativePtr,
                _packed.NativePtr,
                _states.Length);
            _packed.CopyToCPU(_host);
            for (int index = 0; index < _states.Length; index++)
                Array.Copy(_host, index * 4, _states[index].StatsHost, 0, 4);
        }

        public void Dispose()
        {
            _sourcePointers.Dispose();
            _packed.Dispose();
        }
    }

    internal static void NekoMuonPrepareBFloat16StatsResident(
        Tensor parameter,
        int deviceIndex,
        NekoMuonBFloat16ResidentState state,
        NativeCudaBuffer<int>? finiteStatus,
        float betaFast,
        float betaSlow,
        float fastCorrection,
        float slowCorrection,
        float epsilon,
        float rho,
        bool deviceControl)
    {
        if (!parameter.TryGetCudaBFloat16GradientBuffer(
                deviceIndex,
                out NativeCudaBuffer<ushort>? gradient))
        {
            throw new InvalidOperationException(
                $"Pure BFloat16 NekoMuon requires a resident BF16 gradient " +
                $"for tensor '{parameter.Name}' on CUDA device " +
                $"{deviceIndex}.");
        }
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        NekoMuonBFloat16ResidentState.NekoBuffers buffers =
            state.GetOrCreate(deviceIndex);
        buffers.Stats.MemSetToZero();
        if (!CudaNekoMuon.TryMomentsAndStatsBFloat16Compact(
                accelerator,
                gradient!,
                buffers.Fast,
                buffers.Slow,
                buffers.Stats,
                parameter.Numel,
                betaFast,
                betaSlow,
                fastCorrection,
                slowCorrection,
                finiteStatus))
        {
            throw new InvalidOperationException(
                "The ABI 1.18 pure-BF16 NekoMuon statistics kernel is " +
                "required.");
        }
        if (!deviceControl)
            return;
        if (finiteStatus is null)
        {
            throw new ArgumentNullException(
                nameof(finiteStatus),
                "Device-controlled BF16 NekoMuon requires finite status.");
        }
        CudaOptimizerNative.NekoUpdateDeviceControl(
            deviceIndex,
            buffers.Stats.NativePtr,
            buffers.Confidence.NativePtr,
            finiteStatus.NativePtr,
            epsilon,
            rho);
        state.MarkDeviceConfidenceAuthoritative();
    }

    internal static float NekoMuonFinishBFloat16StepResident(
        Tensor parameter,
        int deviceIndex,
        NekoMuonBFloat16ResidentState state,
        NekoMuonDeviceScratch scratch,
        NativeCudaBuffer<int> finiteStatus,
        int originalRows,
        int originalColumns,
        float fastCorrection,
        float epsilon,
        float previousConfidence,
        float rho,
        int maxNewtonSchulzSteps,
        NekoMuonNewtonSchulzDepthMode depthMode,
        float configuredDepth,
        bool runNewtonSchulz,
        float coefficientA,
        float coefficientB,
        float coefficientC,
        float learningRate,
        float weightDecay,
        bool applyWeightDecay,
        bool deviceOnlyFixedFive,
        bool forceFullNewtonSchulz)
    {
        NekoMuonBFloat16ResidentState.NekoBuffers buffers =
            state.GetOrCreate(deviceIndex);
        float confidence;
        float depth;
        if (deviceOnlyFixedFive)
        {
            confidence = previousConfidence;
            depth = 5f;
            CudaOptimizerNative.NekoInitializeBFloat16FromDeviceStats(
                deviceIndex,
                buffers.Fast.NativePtr,
                scratch.X.NativePtr,
                parameter.Numel,
                originalRows,
                originalColumns,
                originalRows > originalColumns,
                1f / fastCorrection,
                buffers.Stats.NativePtr,
                epsilon,
                finiteStatus.NativePtr);
        }
        else
        {
            float[] stats = buffers.StatsHost;
            confidence = CalculateNekoMuonConfidence(
                stats, epsilon, previousConfidence, rho);
            depth = forceFullNewtonSchulz && runNewtonSchulz
                ? maxNewtonSchulzSteps
                : NekoMuon.ResolveNewtonSchulzDepth(
                    maxNewtonSchulzSteps,
                    depthMode,
                    configuredDepth,
                    confidence,
                    runNewtonSchulz);
            float inverseNorm =
                1f / ((float)Math.Sqrt(stats[1]) + epsilon);
            CudaOptimizerNative.NekoInitializeBFloat16(
                deviceIndex,
                buffers.Fast.NativePtr,
                scratch.X.NativePtr,
                parameter.Numel,
                originalRows,
                originalColumns,
                originalRows > originalColumns,
                1f / fastCorrection,
                inverseNorm);
        }

        int rows = Math.Min(originalRows, originalColumns);
        int columns = Math.Max(originalRows, originalColumns);
        int wholeSteps = Math.Min(maxNewtonSchulzSteps, (int)MathF.Floor(depth));
        float fraction = depth - wholeSteps;
        NativeCudaBuffer<float> x = scratch.X;
        NativeCudaBuffer<float> next = scratch.Next;
        for (int step = 0; step < wholeSteps; step++)
        {
            NekoMuonNewtonSchulzResident(
                ForgetMemoryV2Cuda.GetAccelerator(deviceIndex),
                deviceIndex,
                x,
                next,
                scratch.Gram,
                scratch.GramSquared,
                rows,
                columns,
                coefficientA,
                coefficientB,
                coefficientC,
                scratch.UseBFloat16TensorCores);
            (x, next) = (next, x);
        }
        if (fraction > 0f)
        {
            NekoMuonNewtonSchulzResident(
                ForgetMemoryV2Cuda.GetAccelerator(deviceIndex),
                deviceIndex,
                x,
                next,
                scratch.Gram,
                scratch.GramSquared,
                rows,
                columns,
                coefficientA,
                coefficientB,
                coefficientC,
                scratch.UseBFloat16TensorCores);
            CudaOptimizerNative.NekoInterpolate(
                deviceIndex, x.NativePtr, next.NativePtr,
                parameter.Numel, fraction);
        }
        NativeCudaBuffer<float> update = x;
        if (originalRows > originalColumns)
        {
            CudaOptimizerNative.NekoTransposeBack(
                deviceIndex,
                x.NativePtr,
                next.NativePtr,
                parameter.Numel,
                originalRows,
                originalColumns);
            update = next;
        }
        NativeCudaBuffer<ushort> data =
            parameter.EnsureCudaBFloat16Buffer(deviceIndex);
        float finalScale = MathF.Sqrt(MathF.Max(
            1f, (float)originalRows / originalColumns));
        CudaOptimizerNative.NekoApplyBFloat16(
            deviceIndex,
            data.NativePtr,
            update.NativePtr,
            parameter.Numel,
            learningRate,
            finalScale,
            weightDecay,
            applyWeightDecay);
        return confidence;
    }
}
