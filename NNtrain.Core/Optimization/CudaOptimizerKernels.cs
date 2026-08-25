
namespace NNtrain;

internal static class CudaOptimizerKernels
{
    // cuBLAS launch/setup overhead dominates the tiny Gram matrices used by
    // vectors, norms, and narrow projections.  The direct kernels also fold
    // the polynomial combine into the final multiply.
    private const int DirectNewtonSchulzMaxRows = 32;

    internal static void PrewarmNekoMuon(IReadOnlyList<int> deviceIndices)
    {
        foreach (int deviceIndex in deviceIndices)
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex).Bind();
    }

    internal sealed class AdamWResidentState : IDisposable
    {
        private readonly float[] _firstHost;
        private readonly float[] _secondHost;
        private readonly Dictionary<int, Buffers> _buffers = [];

        internal AdamWResidentState(float[] firstHost, float[] secondHost)
        {
            _firstHost = firstHost;
            _secondHost = secondHost;
        }

        internal Buffers GetOrCreate(int deviceIndex)
        {
            if (_buffers.TryGetValue(deviceIndex, out Buffers? buffers))
                return buffers;
            NativeCudaDevice accelerator =
                ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
            buffers = new Buffers(
                accelerator.Allocate1D(_firstHost),
                accelerator.Allocate1D(_secondHost));
            _buffers.Add(deviceIndex, buffers);
            return buffers;
        }

        internal void SynchronizeHost(int deviceIndex)
        {
            if (!_buffers.TryGetValue(deviceIndex, out Buffers? buffers))
                return;
            buffers.First.CopyToCPU(_firstHost);
            buffers.Second.CopyToCPU(_secondHost);
        }

        public void Dispose()
        {
            foreach (Buffers buffers in _buffers.Values)
                buffers.Dispose();
            _buffers.Clear();
        }

        internal sealed class Buffers(
            NativeCudaBuffer<float> first,
            NativeCudaBuffer<float> second) : IDisposable
        {
            internal NativeCudaBuffer<float> First { get; } = first;
            internal NativeCudaBuffer<float> Second { get; } = second;
            public void Dispose()
            {
                First.Dispose();
                Second.Dispose();
            }
        }
    }

    internal sealed class AdamWBFloat16ResidentState : IDisposable
    {
        private readonly short[] _firstHost;
        private readonly short[] _secondHost;
        private readonly Dictionary<int, Buffers> _buffers = [];

        internal AdamWBFloat16ResidentState(
            short[] firstHost,
            short[] secondHost)
        {
            _firstHost = firstHost;
            _secondHost = secondHost;
        }

        internal Buffers GetOrCreate(int deviceIndex)
        {
            if (_buffers.TryGetValue(deviceIndex, out Buffers? buffers))
                return buffers;
            NativeCudaDevice accelerator =
                ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
            buffers = new Buffers(
                accelerator.Allocate1D(_firstHost),
                accelerator.Allocate1D(_secondHost));
            _buffers.Add(deviceIndex, buffers);
            return buffers;
        }

        internal void SynchronizeHost(int deviceIndex)
        {
            if (!_buffers.TryGetValue(deviceIndex, out Buffers? buffers))
                return;
            buffers.First.CopyToCPU(_firstHost);
            buffers.Second.CopyToCPU(_secondHost);
        }

        public void Dispose()
        {
            foreach (Buffers buffers in _buffers.Values)
                buffers.Dispose();
            _buffers.Clear();
        }

        internal sealed class Buffers(
            NativeCudaBuffer<short> first,
            NativeCudaBuffer<short> second) : IDisposable
        {
            internal NativeCudaBuffer<short> First { get; } = first;
            internal NativeCudaBuffer<short> Second { get; } = second;

            public void Dispose()
            {
                First.Dispose();
                Second.Dispose();
            }
        }
    }

    internal sealed record AdamWMultiTensorItem(
        Tensor Parameter,
        AdamWResidentState? FloatState,
        AdamWBFloat16ResidentState? BFloat16State,
        bool ApplyWeightDecay);

    internal sealed class AdamWMultiTensorPlan : IDisposable
    {
        private const int ElementsPerChunk = 4096;
        private readonly int _deviceIndex;
        private readonly NativeCudaBuffer<
            CudaOptimizerNative.AdamWChunkDescriptor> _chunks;

        internal AdamWMultiTensorPlan(
            int deviceIndex,
            IReadOnlyList<AdamWMultiTensorItem> items)
        {
            _deviceIndex = deviceIndex;
            var descriptors = new List<
                CudaOptimizerNative.AdamWChunkDescriptor>();
            foreach (AdamWMultiTensorItem item in items)
            {
                Tensor parameter = item.Parameter;
                NativeCudaBuffer<float> data =
                    parameter.EnsureCudaMasterFloat32Buffer(deviceIndex);
                NativeCudaBuffer<float> gradient =
                    parameter.EnsureCudaGradientBuffer(deviceIndex);
                nint first;
                nint second;
                bool bfloat16State;
                if (item.FloatState is not null)
                {
                    AdamWResidentState.Buffers state =
                        item.FloatState.GetOrCreate(deviceIndex);
                    first = state.First.NativePtr;
                    second = state.Second.NativePtr;
                    bfloat16State = false;
                }
                else if (item.BFloat16State is not null)
                {
                    AdamWBFloat16ResidentState.Buffers state =
                        item.BFloat16State.GetOrCreate(deviceIndex);
                    first = state.First.NativePtr;
                    second = state.Second.NativePtr;
                    bfloat16State = true;
                }
                else
                {
                    continue;
                }
                (nint compute, bool physicalBFloat16) =
                    GetComputeDestination(parameter, deviceIndex);
                for (int offset = 0; offset < parameter.Numel;
                    offset += ElementsPerChunk)
                {
                    int length = Math.Min(
                        ElementsPerChunk, parameter.Numel - offset);
                    descriptors.Add(new CudaOptimizerNative
                        .AdamWChunkDescriptor(
                            data.NativePtr,
                            gradient.NativePtr,
                            first,
                            second,
                            compute,
                            offset,
                            length,
                            item.ApplyWeightDecay ? 1 : 0,
                            physicalBFloat16 ? 1 : 0,
                            bfloat16State ? 1 : 0));
                }
            }
            if (descriptors.Count == 0)
            {
                throw new ArgumentException(
                    "AdamW multi-tensor plan requires CUDA-resident items.",
                    nameof(items));
            }
            _chunks = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex)
                .Allocate1D(System.Runtime.InteropServices.CollectionsMarshal
                    .AsSpan(descriptors));
        }

        internal void Execute(
            float beta1,
            float beta2,
            float learningRate,
            float weightDecay,
            float updateScale,
            float scaledEpsilon)
            => CudaOptimizerNative.AdamWMultiTensor(
                _deviceIndex,
                _chunks.NativePtr,
                _chunks.Length,
                beta1,
                beta2,
                learningRate,
                weightDecay,
                updateScale,
                scaledEpsilon);

        public void Dispose() => _chunks.Dispose();
    }

    internal sealed class NekoMuonResidentState : IDisposable
    {
        private readonly float[] _fastHost;
        private readonly float[] _slowHost;
        private readonly int _length;
        private readonly int _gramLength;
        private readonly Dictionary<int, NekoBuffers> _buffers = [];

        internal NekoMuonResidentState(
            float[] fastHost,
            float[] slowHost,
            int gramLength)
        {
            _fastHost = fastHost;
            _slowHost = slowHost;
            _length = fastHost.Length;
            _gramLength = gramLength;
        }

        internal NekoBuffers GetOrCreate(int deviceIndex)
        {
            if (_buffers.TryGetValue(deviceIndex, out NekoBuffers? buffers))
                return buffers;
            NativeCudaDevice accelerator =
                ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
            buffers = new NekoBuffers(
                accelerator.Allocate1D(_fastHost),
                accelerator.Allocate1D(_slowHost),
                accelerator.Allocate1D<float>(_length),
                accelerator.Allocate1D<float>(_length),
                accelerator.Allocate1D<float>(_length),
                accelerator.Allocate1D<float>(_length),
                accelerator.Allocate1D<float>(_gramLength),
                accelerator.Allocate1D<float>(_gramLength),
                accelerator.Allocate1D<float>(4),
                new float[4]);
            _buffers.Add(deviceIndex, buffers);
            return buffers;
        }

        internal void SynchronizeHost(int deviceIndex)
        {
            if (!_buffers.TryGetValue(deviceIndex, out NekoBuffers? buffers))
                return;
            buffers.Fast.CopyToCPU(_fastHost);
            buffers.Slow.CopyToCPU(_slowHost);
        }

        public void Dispose()
        {
            foreach (NekoBuffers buffers in _buffers.Values)
                buffers.Dispose();
            _buffers.Clear();
        }

        internal sealed class NekoBuffers(
            NativeCudaBuffer<float> fast,
            NativeCudaBuffer<float> slow,
            NativeCudaBuffer<float> fastHat,
            NativeCudaBuffer<float> slowHat,
            NativeCudaBuffer<float> x,
            NativeCudaBuffer<float> next,
            NativeCudaBuffer<float> gram,
            NativeCudaBuffer<float> gramSquared,
            NativeCudaBuffer<float> stats,
            float[] statsHost) : IDisposable
        {
            internal NativeCudaBuffer<float> Fast { get; } = fast;
            internal NativeCudaBuffer<float> Slow { get; } = slow;
            internal NativeCudaBuffer<float> FastHat { get; } = fastHat;
            internal NativeCudaBuffer<float> SlowHat { get; } = slowHat;
            internal NativeCudaBuffer<float> X { get; } = x;
            internal NativeCudaBuffer<float> Next { get; } = next;
            internal NativeCudaBuffer<float> Gram { get; } = gram;
            internal NativeCudaBuffer<float> GramSquared { get; } = gramSquared;
            internal NativeCudaBuffer<float> Stats { get; } = stats;
            internal float[] StatsHost { get; } = statsHost;

            public void Dispose()
            {
                Fast.Dispose();
                Slow.Dispose();
                FastHat.Dispose();
                SlowHat.Dispose();
                X.Dispose();
                Next.Dispose();
                Gram.Dispose();
                GramSquared.Dispose();
                Stats.Dispose();
            }
        }
    }

    /// <summary>
    /// Gathers every four-float NekoMuon statistic record on-device, then
    /// performs one D2H transfer per GPU instead of one per parameter.
    /// </summary>
    internal sealed class NekoMuonStatsBatch : IDisposable
    {
        private readonly int _deviceIndex;
        private readonly NekoMuonResidentState.NekoBuffers[] _states;
        private readonly NativeCudaBuffer<nint> _sourcePointers;
        private readonly NativeCudaBuffer<float> _packed;
        private readonly float[] _host;

        internal NekoMuonStatsBatch(
            int deviceIndex,
            IReadOnlyList<NekoMuonResidentState> states)
        {
            _deviceIndex = deviceIndex;
            NativeCudaDevice accelerator =
                ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
            _states = states
                .Select(state => state.GetOrCreate(deviceIndex))
                .ToArray();
            nint[] pointers = _states
                .Select(state => state.Stats.NativePtr)
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
            for (int index = 0; index < _states.Length; ++index)
            {
                Array.Copy(
                    _host,
                    index * 4,
                    _states[index].StatsHost,
                    0,
                    4);
            }
        }

        public void Dispose()
        {
            _sourcePointers.Dispose();
            _packed.Dispose();
        }
    }

    internal static void NekoMuonPrepareStatsResident(
        Tensor parameter,
        int deviceIndex,
        NekoMuonResidentState state,
        float betaFast,
        float betaSlow,
        float fastCorrection,
        float slowCorrection)
    {
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        var gradientBuffer = parameter.EnsureCudaGradientBuffer(deviceIndex);
        NekoMuonResidentState.NekoBuffers buffers =
            state.GetOrCreate(deviceIndex);
        buffers.Stats.MemSetToZero();
        if (!CudaNekoMuon.TryMomentsAndStats(
            accelerator,
            gradientBuffer,
            buffers.Fast,
            buffers.Slow,
            buffers.FastHat,
            buffers.SlowHat,
            buffers.Stats,
            parameter.Numel,
            betaFast,
            betaSlow,
            fastCorrection,
            slowCorrection))
        {
            throw new InvalidOperationException(
                "The native CUDA NekoMuon statistics kernel is required.");
        }
    }

    internal static void NekoMuonReadStatsResident(
        int deviceIndex,
        NekoMuonResidentState state)
    {
        NekoMuonResidentState.NekoBuffers buffers =
            state.GetOrCreate(deviceIndex);
        buffers.Stats.CopyToCPU(buffers.StatsHost);
    }

    internal static float NekoMuonFinishStepResident(
        Tensor parameter,
        int deviceIndex,
        NekoMuonResidentState state,
        int originalRows,
        int originalColumns,
        float betaFast,
        float betaSlow,
        float fastCorrection,
        float slowCorrection,
        float epsilon,
        float previousConfidence,
        float rho,
        int maxNewtonSchulzSteps,
        bool runNewtonSchulz,
        float coefficientA,
        float coefficientB,
        float coefficientC,
        float learningRate,
        float weightDecay,
        bool applyWeightDecay)
    {
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        var dataBuffer = parameter.EnsureCudaMasterFloat32Buffer(deviceIndex);
        NekoMuonResidentState.NekoBuffers buffers =
            state.GetOrCreate(deviceIndex);
        float[] stats = buffers.StatsHost;
        double alignmentDenominator =
            Math.Sqrt(stats[1]) * Math.Sqrt(stats[2]) + epsilon;
        double alignment = Math.Max(0d, stats[0] / alignmentDenominator);
        double persistence = stats[2] / (stats[2] + stats[3] + epsilon);
        float confidenceRaw =
            (float)Math.Clamp(alignment * persistence, 0d, 1d);
        float confidence = Math.Clamp(
            rho * previousConfidence + (1f - rho) * confidenceRaw,
            0f,
            1f);

        bool transpose = originalRows > originalColumns;
        int rows = Math.Min(originalRows, originalColumns);
        int columns = Math.Max(originalRows, originalColumns);
        float inverseNorm = 1f / ((float)Math.Sqrt(stats[1]) + epsilon);
        CudaOptimizerNative.NekoInitialize(
            deviceIndex,
            buffers.FastHat.NativePtr,
            buffers.X.NativePtr,
            parameter.Numel,
            originalRows,
            originalColumns,
            transpose,
            inverseNorm);

        float depth = runNewtonSchulz
            ? maxNewtonSchulzSteps * confidence
            : 0f;
        bool useBFloat16TensorCores = parameter.DType == TensorDType.BFloat16
            && Environment.GetEnvironmentVariable(
                "NNTRAIN_DISABLE_TENSOR_CORE_NEKOMUON") != "1";
        int wholeSteps = Math.Min(
            maxNewtonSchulzSteps,
            (int)MathF.Floor(depth));
        float fraction = depth - wholeSteps;
        NativeCudaBuffer<float> x = buffers.X;
        NativeCudaBuffer<float> next = buffers.Next;
        for (int step = 0; step < wholeSteps; step++)
        {
            NekoMuonNewtonSchulzResident(
                accelerator, deviceIndex, x, next, buffers.Gram, buffers.GramSquared,
                rows, columns, coefficientA, coefficientB, coefficientC,
                useBFloat16TensorCores);
            (x, next) = (next, x);
        }
        if (fraction > 0f)
        {
            NekoMuonNewtonSchulzResident(
                accelerator, deviceIndex, x, next, buffers.Gram, buffers.GramSquared,
                rows, columns, coefficientA, coefficientB, coefficientC,
                useBFloat16TensorCores);
            CudaOptimizerNative.NekoInterpolate(
                deviceIndex, x.NativePtr, next.NativePtr,
                parameter.Numel, fraction);
        }

        NativeCudaBuffer<float> update = x;
        if (transpose)
        {
            CudaOptimizerNative.NekoTransposeBack(
                deviceIndex,
                x.NativePtr,
                buffers.SlowHat.NativePtr,
                parameter.Numel,
                originalRows,
                originalColumns);
            update = buffers.SlowHat;
        }
        float finalScale = MathF.Sqrt(MathF.Max(
            1f,
            (float)originalRows / originalColumns));
        CudaOptimizerNative.NekoApply(
            deviceIndex,
            dataBuffer.NativePtr,
            update.NativePtr,
            parameter.Numel,
            learningRate,
            finalScale,
            weightDecay,
            applyWeightDecay);
        PublishMaster(parameter, accelerator, deviceIndex, dataBuffer);
        return confidence;
    }

    private static void PublishMaster(
        Tensor parameter,
        NativeCudaDevice accelerator,
        int deviceIndex,
        NativeCudaBuffer<float> master)
    {
        if (parameter.DType == TensorDType.Float32)
            return;
        if (parameter.DType == TensorDType.BFloat16)
        {
            var bfloat16Compute = parameter.EnsureCudaBFloat16Buffer(deviceIndex);
            CudaOptimizerNative.PublishBFloat16(
                deviceIndex, master.NativePtr, bfloat16Compute.NativePtr,
                parameter.Numel, physical: true);
            return;
        }
        var compute = parameter.EnsureCudaFloat32Buffer(deviceIndex);
        CudaOptimizerNative.PublishBFloat16(
            deviceIndex, master.NativePtr, compute.NativePtr,
            parameter.Numel, physical: false);
    }

    private static void NekoMuonNewtonSchulzResident(
        NativeCudaDevice accelerator,
        int deviceIndex,
        NativeCudaBuffer<float> source,
        NativeCudaBuffer<float> destination,
        NativeCudaBuffer<float> gram,
        NativeCudaBuffer<float> gramSquared,
        int rows,
        int columns,
        float coefficientA,
        float coefficientB,
        float coefficientC,
        bool useBFloat16TensorCores)
    {
        if (!useBFloat16TensorCores && rows <= DirectNewtonSchulzMaxRows)
        {
            CudaOptimizerNative.SymmetricGram(
                deviceIndex, source.NativePtr, gram.NativePtr, rows, columns);
            CudaOptimizerNative.SymmetricGram(
                deviceIndex, gram.NativePtr, gramSquared.NativePtr, rows, rows);
            CudaOptimizerNative.NewtonSchulz(
                deviceIndex,
                source.NativePtr,
                gram.NativePtr,
                gramSquared.NativePtr,
                destination.NativePtr,
                rows,
                columns,
                coefficientA,
                coefficientB,
                coefficientC);
            return;
        }

        CudaBlas.MuonGram(
            accelerator, deviceIndex, source, gram, rows, columns,
            useBFloat16TensorCores);
        CudaBlas.MuonGram(
            accelerator, deviceIndex, gram, gramSquared, rows, rows,
            useBFloat16TensorCores);
        CudaOptimizerNative.NekoCombine(
            deviceIndex,
            gram.NativePtr,
            gramSquared.NativePtr,
            checked(rows * rows),
            rows,
            coefficientA,
            coefficientB,
            coefficientC);
        CudaBlas.MuonPolynomialUpdate(
            accelerator,
            deviceIndex,
            source,
            gramSquared,
            destination,
            rows,
            columns,
            useBFloat16TensorCores);
    }

    internal static void AdamWUpdateResident(
        Tensor parameter,
        int deviceIndex,
        AdamWResidentState state,
        float beta1,
        float beta2,
        float learningRate,
        float weightDecay,
        float updateScale,
        float scaledEpsilon,
        bool applyWeightDecay)
    {
        var dataBuffer = parameter.EnsureCudaMasterFloat32Buffer(deviceIndex);
        var gradientBuffer = parameter.EnsureCudaGradientBuffer(deviceIndex);
        AdamWResidentState.Buffers stateBuffers =
            state.GetOrCreate(deviceIndex);
        (nint compute, bool physicalBFloat16) = GetComputeDestination(
            parameter, deviceIndex);
        CudaOptimizerNative.AdamWAndPublish(
            deviceIndex,
            dataBuffer.NativePtr,
            gradientBuffer.NativePtr,
            stateBuffers.First.NativePtr,
            stateBuffers.Second.NativePtr,
            compute,
            parameter.Numel,
            beta1,
            beta2,
            learningRate,
            weightDecay,
            updateScale,
            scaledEpsilon,
            applyWeightDecay,
            bfloat16State: false,
            physicalBFloat16);
    }

    internal static void AdamWUpdateBFloat16Resident(
        Tensor parameter,
        int deviceIndex,
        AdamWBFloat16ResidentState state,
        float beta1,
        float beta2,
        float learningRate,
        float weightDecay,
        float updateScale,
        float scaledEpsilon,
        bool applyWeightDecay)
    {
        var dataBuffer = parameter.EnsureCudaMasterFloat32Buffer(deviceIndex);
        var gradientBuffer = parameter.EnsureCudaGradientBuffer(deviceIndex);
        AdamWBFloat16ResidentState.Buffers stateBuffers =
            state.GetOrCreate(deviceIndex);
        (nint compute, bool physicalBFloat16) = GetComputeDestination(
            parameter, deviceIndex);
        CudaOptimizerNative.AdamWAndPublish(
            deviceIndex,
            dataBuffer.NativePtr,
            gradientBuffer.NativePtr,
            stateBuffers.First.NativePtr,
            stateBuffers.Second.NativePtr,
            compute,
            parameter.Numel,
            beta1,
            beta2,
            learningRate,
            weightDecay,
            updateScale,
            scaledEpsilon,
            applyWeightDecay,
            bfloat16State: true,
            physicalBFloat16);
    }

    private static (nint Compute, bool PhysicalBFloat16) GetComputeDestination(
        Tensor parameter,
        int deviceIndex)
    {
        if (parameter.DType == TensorDType.Float32)
            return (0, false);
        if (parameter.DType == TensorDType.BFloat16)
        {
            return (
                parameter.EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
                true);
        }
        return (parameter.EnsureCudaFloat32Buffer(deviceIndex).NativePtr, false);
    }

    internal static void SynchronizeDevices(IReadOnlyList<int> deviceIndices)
    {
        foreach (int deviceIndex in deviceIndices)
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex).Synchronize();
    }

    internal static void AdamWUpdate(
        float[] data,
        float[] gradient,
        float[] firstMoment,
        float[] secondMoment,
        float beta1,
        float beta2,
        float learningRate,
        float weightDecay,
        float updateScale,
        float scaledEpsilon,
        bool applyWeightDecay)
    {
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        using var dataBuffer = accelerator.Allocate1D(data);
        using var gradientBuffer = accelerator.Allocate1D(gradient);
        using var firstMomentBuffer = accelerator.Allocate1D(firstMoment);
        using var secondMomentBuffer = accelerator.Allocate1D(secondMoment);
        CudaOptimizerNative.AdamW(
            Tensor.CudaDeviceIndex,
            dataBuffer.NativePtr,
            gradientBuffer.NativePtr,
            firstMomentBuffer.NativePtr,
            secondMomentBuffer.NativePtr,
            data.Length,
            beta1,
            beta2,
            learningRate,
            weightDecay,
            updateScale,
            scaledEpsilon,
            applyWeightDecay,
            bfloat16State: false);
        accelerator.Synchronize();
        dataBuffer.CopyToCPU(data);
        firstMomentBuffer.CopyToCPU(firstMoment);
        secondMomentBuffer.CopyToCPU(secondMoment);
    }

    internal static void NekoMuonMoments(
        float[] gradient,
        float[] fast,
        float[] slow,
        float[] fastHat,
        float[] slowHat,
        float betaFast,
        float betaSlow,
        float fastCorrection,
        float slowCorrection)
    {
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        using var gradientBuffer = accelerator.Allocate1D(gradient);
        using var fastBuffer = accelerator.Allocate1D(fast);
        using var slowBuffer = accelerator.Allocate1D(slow);
        using var fastHatBuffer = accelerator.Allocate1D<float>(fastHat.Length);
        using var slowHatBuffer = accelerator.Allocate1D<float>(slowHat.Length);
        CudaOptimizerNative.NekoMoments(
            Tensor.CudaDeviceIndex,
            gradientBuffer.NativePtr,
            fastBuffer.NativePtr,
            slowBuffer.NativePtr,
            fastHatBuffer.NativePtr,
            slowHatBuffer.NativePtr,
            fast.Length,
            betaFast,
            betaSlow,
            fastCorrection,
            slowCorrection);
        accelerator.Synchronize();
        fastBuffer.CopyToCPU(fast);
        slowBuffer.CopyToCPU(slow);
        fastHatBuffer.CopyToCPU(fastHat);
        slowHatBuffer.CopyToCPU(slowHat);
    }

    internal static void NekoMuonApplyUpdate(
        float[] data,
        float[] update,
        float learningRate,
        float finalScale,
        float weightDecay,
        bool applyWeightDecay)
    {
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        using var dataBuffer = accelerator.Allocate1D(data);
        using var updateBuffer = accelerator.Allocate1D(update);
        CudaOptimizerNative.NekoApply(
            Tensor.CudaDeviceIndex,
            dataBuffer.NativePtr,
            updateBuffer.NativePtr,
            data.Length,
            learningRate,
            finalScale,
            weightDecay,
            applyWeightDecay);
        accelerator.Synchronize();
        dataBuffer.CopyToCPU(data);
    }

    internal static void NekoMuonNewtonSchulz(
        float[] source,
        float[] destination,
        float[] gram,
        float[] gramSquared,
        int rows,
        int columns,
        float coefficientA,
        float coefficientB,
        float coefficientC)
    {
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        using var sourceBuffer = accelerator.Allocate1D(source);
        using var destinationBuffer =
            accelerator.Allocate1D<float>(destination.Length);
        using var gramBuffer = accelerator.Allocate1D<float>(gram.Length);
        using var gramSquaredBuffer =
            accelerator.Allocate1D<float>(gramSquared.Length);
        CudaOptimizerNative.SymmetricGram(
            Tensor.CudaDeviceIndex,
            sourceBuffer.NativePtr,
            gramBuffer.NativePtr,
            rows,
            columns);
        CudaOptimizerNative.SymmetricGram(
            Tensor.CudaDeviceIndex,
            gramBuffer.NativePtr,
            gramSquaredBuffer.NativePtr,
            rows,
            rows);
        CudaOptimizerNative.NewtonSchulz(
            Tensor.CudaDeviceIndex,
            sourceBuffer.NativePtr,
            gramBuffer.NativePtr,
            gramSquaredBuffer.NativePtr,
            destinationBuffer.NativePtr,
            rows,
            columns,
            coefficientA,
            coefficientB,
            coefficientC);
        accelerator.Synchronize();
        destinationBuffer.CopyToCPU(destination);
        gramBuffer.CopyToCPU(gram);
        gramSquaredBuffer.CopyToCPU(gramSquared);
    }
}
