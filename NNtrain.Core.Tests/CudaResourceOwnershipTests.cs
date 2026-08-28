using NNtrain;
using NNtrain.Cuda.Execution;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class CudaResourceOwnershipTests
{
    [Fact]
    public void LaneResidentArrayEntriesStayBoundedAcrossTransientArrays()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        using var session = CreateSession();
        using IDisposable scope = session.Enter();
        NativeCudaDevice device = NativeCudaRuntime.GetDevice(0);
        for (int index = 0;
            index < CudaResidentArrayCache.MaximumLaneEntries + 32;
            index++)
        {
            float[] values = [index, index + 1f, index + 2f, index + 3f];
            using CudaResidentArrayLease lease =
                CudaResidentArrayCache.GetOrUpload(device, values);
            Assert.NotEqual(nint.Zero, lease.Buffer.NativePtr);
        }

        Assert.Equal(
            CudaResidentArrayCache.MaximumLaneEntries,
            CudaResidentArrayCache.GetActiveLaneEntryCount(0));
    }

    [Fact]
    public void ForgetMemoryUsesCanonicalValidatedDeviceWrapper()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        Assert.Same(
            NativeCudaRuntime.GetDevice(0),
            ForgetMemoryV2Cuda.GetAccelerator(0));
    }

    [Fact]
    public void ResettableFallbackDefersRetiredValueUntilLeaseReturns()
    {
        var cache = new ResettableBoundedDisposableLeaseCache<
            int,
            TrackingResource>(capacity: 1);
        using BoundedDisposableLeaseCache<int, TrackingResource>.Lease first =
            Assert.IsType<
                BoundedDisposableLeaseCache<int, TrackingResource>.Lease>(
                cache.Acquire(1, static _ => new TrackingResource()));
        TrackingResource retired = first.Value;

        cache.Dispose();
        Assert.False(retired.Disposed);

        using BoundedDisposableLeaseCache<int, TrackingResource>.Lease second =
            Assert.IsType<
                BoundedDisposableLeaseCache<int, TrackingResource>.Lease>(
                cache.Acquire(1, static _ => new TrackingResource()));
        Assert.NotSame(retired, second.Value);
        first.Dispose();
        Assert.True(retired.Disposed);
    }

    [Fact]
    public void SameDeviceLanesOwnIndependentHotPathResources()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        ResourceCounts baseline = ResourceCounts.Capture();
        ExecutionSession first = CreateSession();
        ExecutionSession second = CreateSession();
        try
        {
            ExerciseHotPath(first, seed: 3);
            ExerciseHotPath(second, seed: 11);

            ResourceCounts active = ResourceCounts.Capture();
            Assert.Equal(baseline.Cublas + 2, active.Cublas);
            Assert.Equal(baseline.CublasLt + 2, active.CublasLt);
            Assert.Equal(baseline.CublasLtInt8 + 2, active.CublasLtInt8);
            Assert.Equal(baseline.LayerNorm + 2, active.LayerNorm);
            Assert.Equal(baseline.FloatScalar + 2, active.FloatScalar);
            Assert.Equal(baseline.IntScalar + 2, active.IntScalar);
            Assert.Equal(baseline.GradientNorm + 2, active.GradientNorm);
            Assert.Equal(baseline.ResidentArray + 2, active.ResidentArray);

            second.Dispose();
            second = null!;
            ResourceCounts oneActive = ResourceCounts.Capture();
            Assert.Equal(baseline.Cublas + 1, oneActive.Cublas);
            Assert.Equal(baseline.CublasLt + 1, oneActive.CublasLt);
            Assert.Equal(baseline.CublasLtInt8 + 1, oneActive.CublasLtInt8);
            Assert.Equal(baseline.LayerNorm + 1, oneActive.LayerNorm);
            Assert.Equal(baseline.FloatScalar + 1, oneActive.FloatScalar);
            Assert.Equal(baseline.IntScalar + 1, oneActive.IntScalar);
            Assert.Equal(baseline.GradientNorm + 1, oneActive.GradientNorm);
            Assert.Equal(baseline.ResidentArray + 1, oneActive.ResidentArray);
        }
        finally
        {
            second?.Dispose();
            first.Dispose();
        }

        Assert.Equal(baseline, ResourceCounts.Capture());
    }

    [Fact]
    public void SessionExceptionStillReleasesEveryCreatedHotPathResource()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        ResourceCounts baseline = ResourceCounts.Capture();
        var session = CreateSession();
        try
        {
            ExerciseHotPath(session, seed: 19);
            throw new ScriptedTrainingFailure();
        }
        catch (ScriptedTrainingFailure)
        {
        }
        finally
        {
            session.Dispose();
        }

        Assert.Equal(baseline, ResourceCounts.Capture());
    }

    [Fact]
    public void RepeatedHotPathSessionsReturnEveryTrackedNativeAllocation()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        // Exclude one-time driver and JIT initialization from the measured
        // region while still requiring every lane-owned allocation to return.
        using (ExecutionSession warmup = CreateSession())
            ExerciseHotPath(warmup, seed: 23);
        NativeCudaDevice device = NativeCudaRuntime.GetDevice(0);
        device.Synchronize();
        ResourceCounts baseline = ResourceCounts.Capture();
        NativeCudaAllocationTelemetry before =
            NativeCudaRuntime.AllocationTelemetry;

        for (int iteration = 0; iteration < 8; iteration++)
        {
            using ExecutionSession session = CreateSession();
            ExerciseHotPath(session, seed: 31 + iteration);
        }

        device.Synchronize();
        NativeCudaAllocationTelemetry delta =
            NativeCudaRuntime.AllocationTelemetry - before;
        Assert.Equal(delta.AllocationCount, delta.FreeCount);
        Assert.Equal(delta.AllocationBytes, delta.FreeBytes);
        Assert.Equal(baseline, ResourceCounts.Capture());
    }

    [Fact]
    public void LegacyFallbackIsBoundedAndExplicitlyDisposable()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        NativeCudaRuntime.DisposeFallbackResources();
        NativeCudaDevice device = NativeCudaRuntime.GetDevice(0);
        device.Bind();
        using (NativeCudaBuffer<float> left = device.Allocate(
                   new float[] { 1f, 2f, 3f, 4f }))
        using (NativeCudaBuffer<float> right = device.Allocate(
                   new float[] { 5f, 6f, 7f, 8f }))
        using (NativeCudaBuffer<float> output = device.Allocate1D<float>(4))
        {
            CudaBlas.MatMulForward(
                device,
                0,
                left,
                right,
                output,
                batch: 1,
                m: 2,
                k: 2,
                n: 2);
            NativeCudaScalarReadback readback =
                NativeCudaScalarReadback.Rent(0);
            readback.Begin(output.NativePtr, device.DefaultStream);
            _ = readback.CompleteAndReturn();
        }

        NativeCudaFallbackResourceTelemetry populated =
            NativeCudaRuntime.FallbackResourceTelemetry;
        Assert.InRange(
            populated.CublasHandleCount,
            1,
            8);
        Assert.InRange(populated.FloatScalarPoolCount, 1, 4);
        Assert.True(populated.LiveSlotCount >= 1);

        NativeCudaRuntime.DisposeFallbackResources();
        NativeCudaFallbackResourceTelemetry cleared =
            NativeCudaRuntime.FallbackResourceTelemetry;
        Assert.Equal(0, cleared.CachedOwnerCount);
        Assert.Equal(0, cleared.LiveSlotCount);
    }

    private static ExecutionSession CreateSession()
        => new(
            new ExecutionOptions
            {
                Device = ExecutionDeviceKind.Cuda,
                CudaDevices = new DeviceSet(0),
                Precision = PrecisionPolicy.Float32,
            },
            [CudaExecutionLaneFactory.Create(0)]);

    private static void ExerciseHotPath(
        ExecutionSession session,
        int seed)
    {
        using IDisposable scope = session.Enter();
        NativeCudaDevice device = NativeCudaRuntime.GetDevice(0);
        const int width = 16;
        const int length = width * width;

        using NativeCudaBuffer<float> left = device.Allocate(
            Enumerable.Range(0, length)
                .Select(index => (index + seed) * 0.001f)
                .ToArray());
        using NativeCudaBuffer<float> right = device.Allocate(
            Enumerable.Range(0, length)
                .Select(index => (index + seed + 7) * 0.002f)
                .ToArray());
        using NativeCudaBuffer<float> output =
            device.Allocate1D<float>(length);
        CudaBlas.MatMulForward(
            device,
            0,
            left,
            right,
            output,
            batch: 1,
            m: width,
            k: width,
            n: width);

        ushort[] bf16 = new ushort[length];
        TensorStorageCodec.EncodeBFloat16(
            Enumerable.Repeat(0.25f, length).ToArray(),
            bf16);
        ushort[] biasValues = new ushort[width];
        using NativeCudaBuffer<ushort> bf16Input = device.Allocate(bf16);
        using NativeCudaBuffer<ushort> bf16Weight = device.Allocate(bf16);
        using NativeCudaBuffer<ushort> bf16Bias = device.Allocate(biasValues);
        using NativeCudaBuffer<ushort> bf16Output =
            device.Allocate1D<ushort>(length);
        _ = CudaBlasLt.TryLinearForwardBFloat16(
            device,
            0,
            bf16Input,
            bf16Weight,
            bf16Bias,
            bf16Output,
            rows: width,
            inputWidth: width,
            outputWidth: width,
            applyRelu: false);

        using NativeCudaBuffer<sbyte> int8Left =
            device.Allocate(new sbyte[length]);
        using NativeCudaBuffer<sbyte> int8Right =
            device.Allocate(new sbyte[length]);
        using NativeCudaBuffer<int> int8Output =
            device.Allocate1D<int>(length);
        _ = CudaBlasLtInt8.TryMatMul(
            device,
            0,
            int8Left,
            int8Right,
            int8Output,
            width,
            width,
            width);

        ExerciseLayerNormScratch(device);

        NativeCudaScalarReadback floatReadback =
            NativeCudaScalarReadback.Rent(0);
        floatReadback.Begin(output.NativePtr, device.DefaultStream);
        _ = floatReadback.CompleteAndReturn();
        using NativeCudaBuffer<int> intScalar = device.Allocate(new[] { seed });
        NativeCudaIntScalarReadback intReadback =
            NativeCudaIntScalarReadback.Rent(0);
        intReadback.Begin(intScalar.NativePtr, device.DefaultStream);
        _ = intReadback.CompleteAndReturn();

        float[] residentValues = [seed, seed + 1f, seed + 2f, seed + 3f];
        using CudaResidentArrayLease residentLease =
            CudaResidentArrayCache.GetOrUpload(device, residentValues);

        var parameter = new Parameter(
            new float[width],
            [width],
            $"resource.{seed}",
            WeightDecayPolicy.Apply);
        parameter.T.SetCudaGradient(
            Enumerable.Repeat(0.5f, width).ToArray(),
            0);
        _ = TensorCudaKernels.ClipGradientNormResident(
            [parameter],
            maxNorm: 100f);
        parameter.T.InvalidateCudaBuffers();
    }

    private static void ExerciseLayerNormScratch(NativeCudaDevice device)
    {
        const int rows = 2;
        const int columns = 16;
        int length = rows * columns;
        using NativeCudaBuffer<float> input =
            device.Allocate(Enumerable.Repeat(0.25f, length).ToArray());
        using NativeCudaBuffer<float> gamma =
            device.Allocate(Enumerable.Repeat(1f, columns).ToArray());
        using NativeCudaBuffer<float> means =
            device.Allocate(new float[rows]);
        using NativeCudaBuffer<float> inverses =
            device.Allocate(Enumerable.Repeat(1f, rows).ToArray());
        using NativeCudaBuffer<float> outputGradient =
            device.Allocate(Enumerable.Repeat(1f, length).ToArray());
        using NativeCudaBuffer<float> inputGradient =
            device.Allocate1D<float>(length);
        using NativeCudaBuffer<float> gammaGradient =
            device.Allocate1D<float>(columns);
        using NativeCudaBuffer<float> betaGradient =
            device.Allocate1D<float>(columns);
        CudaLayerNorm.Backward(
            device,
            input,
            gamma,
            means,
            inverses,
            outputGradient,
            inputGradient,
            gammaGradient,
            betaGradient,
            rows,
            columns);
        device.Synchronize();
    }

    private readonly record struct ResourceCounts(
        int Cublas,
        int CublasLt,
        int CublasLtInt8,
        int LayerNorm,
        int FloatScalar,
        int IntScalar,
        int GradientNorm,
        int ResidentArray)
    {
        internal static ResourceCounts Capture() => new(
            CudaBlas.ActiveLaneHandleCount,
            CudaBlasLt.ActiveLaneResourceCount,
            CudaBlasLtInt8.ActiveLaneResourceCount,
            CudaLayerNorm.ActiveLaneScratchResourceCount,
            NativeCudaScalarReadback.ActiveLanePoolCount,
            NativeCudaIntScalarReadback.ActiveLanePoolCount,
            TensorCudaKernels.ActiveLaneGradientNormScratchCount,
            CudaResidentArrayCache.ActiveLaneCacheCount);
    }

    private sealed class TrackingResource : IDisposable
    {
        private int _disposed;
        internal bool Disposed => Volatile.Read(ref _disposed) != 0;
        public void Dispose() => Interlocked.Exchange(ref _disposed, 1);
    }

    private sealed class ScriptedTrainingFailure : Exception;
}
