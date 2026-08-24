using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

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
        {
            CudaAccelerator accelerator =
                ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
            _ = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
                ArrayView<float>, ArrayView<float>, ArrayView<double>, float,
                float, float, float>(NekoMuonMomentsAndStatsKernel);
            _ = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, int, int, int, float>(
                    NekoMuonInitializeKernel);
            _ = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, float>(
                    NekoMuonInterpolateKernel);
            _ = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, int, int>(
                    NekoMuonTransposeBackKernel);
            _ = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, float, float,
                float, int>(NekoMuonApplyUpdateKernel);
            _ = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>>(
                    PublishBFloat16MasterKernel);
            _ = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, int, float,
                float, float>(NekoMuonCombinePolynomialKernel);
            _ = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, int, int>(
                    SymmetricGramKernel);
            _ = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
                ArrayView<float>, int, int, float, float, float>(
                    NewtonSchulzKernel);
        }
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
            CudaAccelerator accelerator =
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
            MemoryBuffer1D<float, Stride1D.Dense> first,
            MemoryBuffer1D<float, Stride1D.Dense> second) : IDisposable
        {
            internal MemoryBuffer1D<float, Stride1D.Dense> First { get; } = first;
            internal MemoryBuffer1D<float, Stride1D.Dense> Second { get; } = second;
            public void Dispose()
            {
                First.Dispose();
                Second.Dispose();
            }
        }
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
            CudaAccelerator accelerator =
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
                accelerator.Allocate1D<double>(4),
                new double[4]);
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
            MemoryBuffer1D<float, Stride1D.Dense> fast,
            MemoryBuffer1D<float, Stride1D.Dense> slow,
            MemoryBuffer1D<float, Stride1D.Dense> fastHat,
            MemoryBuffer1D<float, Stride1D.Dense> slowHat,
            MemoryBuffer1D<float, Stride1D.Dense> x,
            MemoryBuffer1D<float, Stride1D.Dense> next,
            MemoryBuffer1D<float, Stride1D.Dense> gram,
            MemoryBuffer1D<float, Stride1D.Dense> gramSquared,
            MemoryBuffer1D<double, Stride1D.Dense> stats,
            double[] statsHost) : IDisposable
        {
            internal MemoryBuffer1D<float, Stride1D.Dense> Fast { get; } = fast;
            internal MemoryBuffer1D<float, Stride1D.Dense> Slow { get; } = slow;
            internal MemoryBuffer1D<float, Stride1D.Dense> FastHat { get; } = fastHat;
            internal MemoryBuffer1D<float, Stride1D.Dense> SlowHat { get; } = slowHat;
            internal MemoryBuffer1D<float, Stride1D.Dense> X { get; } = x;
            internal MemoryBuffer1D<float, Stride1D.Dense> Next { get; } = next;
            internal MemoryBuffer1D<float, Stride1D.Dense> Gram { get; } = gram;
            internal MemoryBuffer1D<float, Stride1D.Dense> GramSquared { get; } = gramSquared;
            internal MemoryBuffer1D<double, Stride1D.Dense> Stats { get; } = stats;
            internal double[] StatsHost { get; } = statsHost;

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

    internal static void NekoMuonPrepareStatsResident(
        Tensor parameter,
        int deviceIndex,
        NekoMuonResidentState state,
        float betaFast,
        float betaSlow,
        float fastCorrection,
        float slowCorrection)
    {
        CudaAccelerator accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        var gradientBuffer = parameter.EnsureCudaGradientBuffer(deviceIndex);
        NekoMuonResidentState.NekoBuffers buffers =
            state.GetOrCreate(deviceIndex);
        buffers.Stats.MemSetToZero();
        var momentsKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
            ArrayView<float>, ArrayView<float>, ArrayView<double>, float,
            float, float, float>(NekoMuonMomentsAndStatsKernel);
        momentsKernel(
            parameter.Numel,
            gradientBuffer.View,
            buffers.Fast.View,
            buffers.Slow.View,
            buffers.FastHat.View,
            buffers.SlowHat.View,
            buffers.Stats.View,
            betaFast,
            betaSlow,
            fastCorrection,
            slowCorrection);
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
        CudaAccelerator accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        var dataBuffer = parameter.EnsureCudaMasterFloat32Buffer(deviceIndex);
        NekoMuonResidentState.NekoBuffers buffers =
            state.GetOrCreate(deviceIndex);
        double[] stats = buffers.StatsHost;
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
        var initializeKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, int, int, int, float>(
                NekoMuonInitializeKernel);
        initializeKernel(
            parameter.Numel,
            buffers.FastHat.View,
            buffers.X.View,
            originalRows,
            originalColumns,
            transpose ? 1 : 0,
            inverseNorm);

        float depth = runNewtonSchulz
            ? maxNewtonSchulzSteps * confidence
            : 0f;
        int wholeSteps = Math.Min(
            maxNewtonSchulzSteps,
            (int)MathF.Floor(depth));
        float fraction = depth - wholeSteps;
        MemoryBuffer1D<float, Stride1D.Dense> x = buffers.X;
        MemoryBuffer1D<float, Stride1D.Dense> next = buffers.Next;
        for (int step = 0; step < wholeSteps; step++)
        {
            NekoMuonNewtonSchulzResident(
                accelerator, deviceIndex, x, next, buffers.Gram, buffers.GramSquared,
                rows, columns, coefficientA, coefficientB, coefficientC);
            (x, next) = (next, x);
        }
        if (fraction > 0f)
        {
            NekoMuonNewtonSchulzResident(
                accelerator, deviceIndex, x, next, buffers.Gram, buffers.GramSquared,
                rows, columns, coefficientA, coefficientB, coefficientC);
            var interpolateKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, float>(
                    NekoMuonInterpolateKernel);
            interpolateKernel(parameter.Numel, x.View, next.View, fraction);
        }

        MemoryBuffer1D<float, Stride1D.Dense> update = x;
        if (transpose)
        {
            var transposeKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, int, int>(
                    NekoMuonTransposeBackKernel);
            transposeKernel(
                parameter.Numel,
                x.View,
                buffers.SlowHat.View,
                originalRows,
                originalColumns);
            update = buffers.SlowHat;
        }
        float finalScale = MathF.Sqrt(MathF.Max(
            1f,
            (float)originalRows / originalColumns));
        var applyKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, float, float, float,
            int>(NekoMuonApplyUpdateKernel);
        applyKernel(
            parameter.Numel,
            dataBuffer.View,
            update.View,
            learningRate,
            finalScale,
            weightDecay,
            applyWeightDecay ? 1 : 0);
        PublishMaster(parameter, accelerator, deviceIndex, dataBuffer);
        return confidence;
    }

    private static void PublishMaster(
        Tensor parameter,
        CudaAccelerator accelerator,
        int deviceIndex,
        MemoryBuffer1D<float, Stride1D.Dense> master)
    {
        if (parameter.DType == TensorDType.Float32)
            return;
        if (parameter.DType == TensorDType.BFloat16)
        {
            var bfloat16Compute = parameter.EnsureCudaBFloat16Buffer(deviceIndex);
            var bfloat16Kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<ushort>>(
                    PublishPhysicalBFloat16MasterKernel);
            bfloat16Kernel(parameter.Numel, master.View, bfloat16Compute.View);
            return;
        }
        var compute = parameter.EnsureCudaFloat32Buffer(deviceIndex);
        var publishKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>>(
                PublishBFloat16MasterKernel);
        publishKernel(parameter.Numel, master.View, compute.View);
    }

    private static void NekoMuonNewtonSchulzResident(
        CudaAccelerator accelerator,
        int deviceIndex,
        MemoryBuffer1D<float, Stride1D.Dense> source,
        MemoryBuffer1D<float, Stride1D.Dense> destination,
        MemoryBuffer1D<float, Stride1D.Dense> gram,
        MemoryBuffer1D<float, Stride1D.Dense> gramSquared,
        int rows,
        int columns,
        float coefficientA,
        float coefficientB,
        float coefficientC)
    {
        if (rows <= DirectNewtonSchulzMaxRows)
        {
            var gramKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, int, int>(
                    SymmetricGramKernel);
            gramKernel(
                checked(rows * rows), source.View, gram.View, rows, columns);
            gramKernel(
                checked(rows * rows), gram.View, gramSquared.View, rows, rows);
            var updateKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
                ArrayView<float>, int, int, float, float, float>(
                    NewtonSchulzKernel);
            updateKernel(
                checked(rows * columns),
                source.View,
                gram.View,
                gramSquared.View,
                destination.View,
                rows,
                columns,
                coefficientA,
                coefficientB,
                coefficientC);
            return;
        }

        CudaBlas.MuonGram(
            accelerator, deviceIndex, source, gram, rows, columns);
        CudaBlas.MuonGram(
            accelerator, deviceIndex, gram, gramSquared, rows, rows);
        var combineKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, int, float, float,
            float>(NekoMuonCombinePolynomialKernel);
        combineKernel(
            checked(rows * rows),
            gram.View,
            gramSquared.View,
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
            columns);
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
        CudaAccelerator accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        var dataBuffer = parameter.EnsureCudaMasterFloat32Buffer(deviceIndex);
        var gradientBuffer = parameter.EnsureCudaGradientBuffer(deviceIndex);
        AdamWResidentState.Buffers stateBuffers =
            state.GetOrCreate(deviceIndex);
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
            ArrayView<float>, float, float, float, float, float, float, int>(
                AdamWKernel);
        kernel(
            parameter.Numel,
            dataBuffer.View,
            gradientBuffer.View,
            stateBuffers.First.View,
            stateBuffers.Second.View,
            beta1,
            beta2,
            learningRate,
            weightDecay,
            updateScale,
            scaledEpsilon,
            applyWeightDecay ? 1 : 0);
        PublishMaster(parameter, accelerator, deviceIndex, dataBuffer);
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
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        using var dataBuffer = accelerator.Allocate1D(data);
        using var gradientBuffer = accelerator.Allocate1D(gradient);
        using var firstMomentBuffer = accelerator.Allocate1D(firstMoment);
        using var secondMomentBuffer = accelerator.Allocate1D(secondMoment);
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
            ArrayView<float>, float, float, float, float, float, float, int>(
                AdamWKernel);
        kernel(
            data.Length,
            dataBuffer.View,
            gradientBuffer.View,
            firstMomentBuffer.View,
            secondMomentBuffer.View,
            beta1,
            beta2,
            learningRate,
            weightDecay,
            updateScale,
            scaledEpsilon,
            applyWeightDecay ? 1 : 0);
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
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        using var gradientBuffer = accelerator.Allocate1D(gradient);
        using var fastBuffer = accelerator.Allocate1D(fast);
        using var slowBuffer = accelerator.Allocate1D(slow);
        using var fastHatBuffer = accelerator.Allocate1D<float>(fastHat.Length);
        using var slowHatBuffer = accelerator.Allocate1D<float>(slowHat.Length);
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
            ArrayView<float>, ArrayView<float>, float, float, float, float>(
                NekoMuonMomentsKernel);
        kernel(
            fast.Length,
            gradientBuffer.View,
            fastBuffer.View,
            slowBuffer.View,
            fastHatBuffer.View,
            slowHatBuffer.View,
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
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        using var dataBuffer = accelerator.Allocate1D(data);
        using var updateBuffer = accelerator.Allocate1D(update);
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, float, float, float,
            int>(NekoMuonApplyUpdateKernel);
        kernel(
            data.Length,
            dataBuffer.View,
            updateBuffer.View,
            learningRate,
            finalScale,
            weightDecay,
            applyWeightDecay ? 1 : 0);
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
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        using var sourceBuffer = accelerator.Allocate1D(source);
        using var destinationBuffer =
            accelerator.Allocate1D<float>(destination.Length);
        using var gramBuffer = accelerator.Allocate1D<float>(gram.Length);
        using var gramSquaredBuffer =
            accelerator.Allocate1D<float>(gramSquared.Length);
        var gramKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, int, int>(
                SymmetricGramKernel);
        gramKernel(
            checked(rows * rows),
            sourceBuffer.View,
            gramBuffer.View,
            rows,
            columns);
        gramKernel(
            checked(rows * rows),
            gramBuffer.View,
            gramSquaredBuffer.View,
            rows,
            rows);
        var updateKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
            ArrayView<float>, int, int, float, float, float>(
                NewtonSchulzKernel);
        updateKernel(
            source.Length,
            sourceBuffer.View,
            gramBuffer.View,
            gramSquaredBuffer.View,
            destinationBuffer.View,
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

    private static void AdamWKernel(
        Index1D index,
        ArrayView<float> data,
        ArrayView<float> gradient,
        ArrayView<float> firstMoment,
        ArrayView<float> secondMoment,
        float beta1,
        float beta2,
        float learningRate,
        float weightDecay,
        float updateScale,
        float scaledEpsilon,
        int applyWeightDecay)
    {
        int i = index;
        float g = gradient[i];
        float first = beta1 * firstMoment[i] + (1f - beta1) * g;
        float second = beta2 * secondMoment[i] + (1f - beta2) * g * g;
        firstMoment[i] = first;
        secondMoment[i] = second;
        float parameter = data[i];
        if (applyWeightDecay != 0)
            parameter *= 1f - learningRate * weightDecay;
        data[i] = parameter - updateScale * first /
            (XMath.Sqrt(second) + scaledEpsilon);
    }

    private static void PublishBFloat16MasterKernel(
        Index1D index,
        ArrayView<float> master,
        ArrayView<float> compute)
    {
        int i = index;
        uint bits = Interop.FloatAsInt(master[i]);
        uint roundingBias = 0x7FFFu + ((bits >> 16) & 1u);
        compute[i] = Interop.IntAsFloat(
            (bits + roundingBias) & 0xFFFF0000u);
    }

    private static void PublishPhysicalBFloat16MasterKernel(
        Index1D index,
        ArrayView<float> master,
        ArrayView<ushort> compute)
    {
        int i = index;
        uint bits = Interop.FloatAsInt(master[i]);
        uint roundingBias = 0x7FFFu + ((bits >> 16) & 1u);
        compute[i] = (ushort)((bits + roundingBias) >> 16);
    }

    private static void NekoMuonMomentsKernel(
        Index1D index,
        ArrayView<float> gradient,
        ArrayView<float> fast,
        ArrayView<float> slow,
        ArrayView<float> fastHat,
        ArrayView<float> slowHat,
        float betaFast,
        float betaSlow,
        float fastCorrection,
        float slowCorrection)
    {
        int i = index;
        float nextFast = betaFast * fast[i] +
            (1f - betaFast) * gradient[i];
        float nextSlow = betaSlow * slow[i] +
            (1f - betaSlow) * gradient[i];
        fast[i] = nextFast;
        slow[i] = nextSlow;
        fastHat[i] = nextFast / fastCorrection;
        slowHat[i] = nextSlow / slowCorrection;
    }

    private static void NekoMuonMomentsAndStatsKernel(
        Index1D index,
        ArrayView<float> gradient,
        ArrayView<float> fast,
        ArrayView<float> slow,
        ArrayView<float> fastHat,
        ArrayView<float> slowHat,
        ArrayView<double> stats,
        float betaFast,
        float betaSlow,
        float fastCorrection,
        float slowCorrection)
    {
        int i = index;
        float nextFast = betaFast * fast[i] +
            (1f - betaFast) * gradient[i];
        float nextSlow = betaSlow * slow[i] +
            (1f - betaSlow) * gradient[i];
        float correctedFast = nextFast / fastCorrection;
        float correctedSlow = nextSlow / slowCorrection;
        float residual = correctedFast - correctedSlow;
        fast[i] = nextFast;
        slow[i] = nextSlow;
        fastHat[i] = correctedFast;
        slowHat[i] = correctedSlow;
        Atomic.Add(ref stats[0], (double)correctedFast * correctedSlow);
        Atomic.Add(ref stats[1], (double)correctedFast * correctedFast);
        Atomic.Add(ref stats[2], (double)correctedSlow * correctedSlow);
        Atomic.Add(ref stats[3], (double)residual * residual);
    }

    private static void NekoMuonApplyUpdateKernel(
        Index1D index,
        ArrayView<float> data,
        ArrayView<float> update,
        float learningRate,
        float finalScale,
        float weightDecay,
        int applyWeightDecay)
    {
        int i = index;
        float parameter = data[i];
        if (applyWeightDecay != 0)
            parameter -= learningRate * weightDecay * parameter;
        data[i] = parameter - learningRate * finalScale * update[i];
    }

    private static void NekoMuonStatsKernel(
        Index1D index,
        ArrayView<float> fastHat,
        ArrayView<float> slowHat,
        ArrayView<double> stats)
    {
        int i = index;
        float fast = fastHat[i];
        float slow = slowHat[i];
        float residual = fast - slow;
        Atomic.Add(ref stats[0], (double)fast * slow);
        Atomic.Add(ref stats[1], (double)fast * fast);
        Atomic.Add(ref stats[2], (double)slow * slow);
        Atomic.Add(ref stats[3], (double)residual * residual);
    }

    private static void NekoMuonInitializeKernel(
        Index1D index,
        ArrayView<float> source,
        ArrayView<float> destination,
        int originalRows,
        int originalColumns,
        int transpose,
        float inverseNorm)
    {
        int linear = index;
        if (transpose == 0)
        {
            destination[linear] = source[linear] * inverseNorm;
            return;
        }
        int row = linear / originalColumns;
        int column = linear - row * originalColumns;
        destination[column * originalRows + row] =
            source[linear] * inverseNorm;
    }

    private static void NekoMuonInterpolateKernel(
        Index1D index,
        ArrayView<float> current,
        ArrayView<float> next,
        float fraction)
    {
        int i = index;
        current[i] += fraction * (next[i] - current[i]);
    }

    private static void NekoMuonTransposeBackKernel(
        Index1D index,
        ArrayView<float> source,
        ArrayView<float> destination,
        int originalRows,
        int originalColumns)
    {
        int linear = index;
        int row = linear / originalColumns;
        int column = linear - row * originalColumns;
        destination[linear] = source[column * originalRows + row];
    }

    private static void NekoMuonCombinePolynomialKernel(
        Index1D index,
        ArrayView<float> gram,
        ArrayView<float> gramSquared,
        int rows,
        float coefficientA,
        float coefficientB,
        float coefficientC)
    {
        int linear = index;
        int row = linear / rows;
        int column = linear - row * rows;
        gramSquared[linear] = coefficientB * gram[linear]
            + coefficientC * gramSquared[linear]
            + (row == column ? coefficientA : 0f);
    }

    private static void SymmetricGramKernel(
        Index1D index,
        ArrayView<float> source,
        ArrayView<float> destination,
        int rows,
        int columns)
    {
        int linear = index;
        int row = linear / rows;
        int other = linear - row * rows;
        float sum = 0f;
        int rowOffset = row * columns;
        int otherOffset = other * columns;
        for (int column = 0; column < columns; column++)
            sum += source[rowOffset + column] * source[otherOffset + column];
        destination[linear] = sum;
    }

    private static void NewtonSchulzKernel(
        Index1D index,
        ArrayView<float> source,
        ArrayView<float> gram,
        ArrayView<float> gramSquared,
        ArrayView<float> destination,
        int rows,
        int columns,
        float coefficientA,
        float coefficientB,
        float coefficientC)
    {
        int linear = index;
        int row = linear / columns;
        int column = linear - row * columns;
        float result = coefficientA * source[linear];
        int coefficientOffset = row * rows;
        for (int inner = 0; inner < rows; inner++)
        {
            float coefficient =
                coefficientB * gram[coefficientOffset + inner] +
                coefficientC * gramSquared[coefficientOffset + inner];
            result += coefficient * source[inner * columns + column];
        }
        destination[linear] = result;
    }
}
