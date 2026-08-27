using NNtrain;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class CudaTensorStateRegressionTests
{
    [Fact]
    public void PrecisionPolicyScopeNestsAndSurvivesDeviceScopes()
    {
        Assert.Null(TensorExecutionContext.ActivePrecisionPolicy);

        IDisposable outer = TensorExecutionContext.PushPrecisionPolicy(
            PrecisionPolicy.BFloat16);
        Assert.Same(
            PrecisionPolicy.BFloat16,
            TensorExecutionContext.ActivePrecisionPolicy);
        using (TensorExecutionContext.Push(
            new TorchDevice(TensorDevice.Cpu)))
        {
            Assert.Same(
                PrecisionPolicy.BFloat16,
                TensorExecutionContext.ActivePrecisionPolicy);
            using (TensorExecutionContext.PushPrecisionPolicy(
                PrecisionPolicy.Mix16_32))
            {
                Assert.Same(
                    PrecisionPolicy.Mix16_32,
                    TensorExecutionContext.ActivePrecisionPolicy);
            }
            Assert.Same(
                PrecisionPolicy.BFloat16,
                TensorExecutionContext.ActivePrecisionPolicy);
        }

        outer.Dispose();
        outer.Dispose();
        Assert.Null(TensorExecutionContext.ActivePrecisionPolicy);
    }

    [Fact]
    public void ChangingCudaDeviceIndicesPreservesCpuExecution()
    {
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cpu;

            Tensor.CudaDeviceIndices = [0];

            Assert.Equal(TensorDevice.Cpu, Tensor.ExecutionDevice);
            Assert.Equal([0], Tensor.CudaDeviceIndices);
        }
        finally
        {
            Tensor.CudaDeviceIndices = previousIndices;
            Tensor.ExecutionDevice = previousDevice;
        }
    }

    [Fact]
    public void NestedExecutionScopesRemainStableAfterDoubleDispose()
    {
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        IDisposable? outer = null;
        IDisposable? inner = null;
        try
        {
            // These indices exercise context state only; no CUDA call is made.
            Tensor.CudaDeviceIndices = [7, 9];
            Tensor.ExecutionDevice = TensorDevice.Cpu;

            outer = TensorExecutionContext.Push(
                new TorchDevice(TensorDevice.Cuda, 3));
            inner = TensorExecutionContext.Push(
                new TorchDevice(TensorDevice.Cuda, 4));

            Assert.Equal(TensorDevice.Cuda, Tensor.ExecutionDevice);
            Assert.Equal(4, Tensor.CudaDeviceIndex);
            Assert.Equal([7, 9], Tensor.CudaDeviceIndices);

            inner.Dispose();
            inner.Dispose();

            Assert.Equal(TensorDevice.Cuda, Tensor.ExecutionDevice);
            Assert.Equal(3, Tensor.CudaDeviceIndex);
            Assert.Equal([7, 9], Tensor.CudaDeviceIndices);

            outer.Dispose();
            outer.Dispose();

            Assert.Equal(TensorDevice.Cpu, Tensor.ExecutionDevice);
            Assert.Equal([7, 9], Tensor.CudaDeviceIndices);
        }
        finally
        {
            inner?.Dispose();
            outer?.Dispose();
            Tensor.CudaDeviceIndices = previousIndices;
            Tensor.ExecutionDevice = previousDevice;
        }
    }

    [Fact]
    public void ConcurrentEnsureCopiesAuthoritativeTensorToBothCudaDevices()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = [0, 1];
            Tensor.ExecutionDevice = TensorDevice.Cuda;

            const int rounds = 16;
            const int length = 257;
            for (int round = 0; round < rounds; round++)
            {
                var tensor = Tensor.Zeros(length);
                try
                {
                    float[] expected = Enumerable.Range(0, length)
                        .Select(index => round * 1000f + index + 0.25f)
                        .ToArray();
                    // Materialize a stale secondary replica before device 0
                    // becomes authoritative. This catches global-version
                    // bookkeeping that incorrectly treats every device as
                    // current after one device is updated.
                    tensor.EnsureCudaFloat32Buffer(1);
                    NativeCudaBuffer<float> primary =
                        tensor.EnsureCudaFloat32Buffer(0);
                    primary.CopyFromCPU(expected);
                    tensor.MarkCudaDataMutated(0);

                    float[][] actual = [new float[length], new float[length]];
                    using var start = new Barrier(2);
                    Parallel.For(0, 2, deviceIndex =>
                    {
                        using IDisposable scope = TensorExecutionContext.Push(
                            new TorchDevice(TensorDevice.Cuda, deviceIndex));
                        start.SignalAndWait();
                        for (int pass = 0; pass < 8; pass++)
                        {
                            NativeCudaBuffer<float> buffer =
                                tensor.EnsureCudaFloat32Buffer(deviceIndex);
                            if (pass == 7)
                                buffer.CopyToCPU(actual[deviceIndex]);
                        }
                    });

                    Assert.Equal(expected, actual[0]);
                    Assert.Equal(expected, actual[1]);
                }
                finally
                {
                    tensor.InvalidateCudaBuffers();
                }
            }
        }
        finally
        {
            Tensor.CudaDeviceIndices = previousIndices;
            Tensor.ExecutionDevice = previousDevice;
        }
    }

    [Fact]
    public void ConcurrentBFloat16EnsureRefreshesExistingSecondaryReplica()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = [0, 1];
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            const int length = 257;
            var tensor = new Tensor(
                new float[length],
                [length],
                dtype: TensorDType.BFloat16);
            try
            {
                tensor.EnsureCudaBFloat16Buffer(1);
                float[] expectedValues = Enumerable.Range(0, length)
                    .Select(index => index * 0.03125f - 2f)
                    .ToArray();
                var expected = new ushort[length];
                TensorStorageCodec.EncodeBFloat16(expectedValues, expected);
                NativeCudaBuffer<ushort> primary =
                    tensor.EnsureCudaBFloat16Buffer(0);
                primary.CopyFromCPU(expected);
                tensor.MarkCudaDataMutated(0);

                ushort[][] actual =
                    [new ushort[length], new ushort[length]];
                using var start = new Barrier(2);
                Parallel.For(0, 2, deviceIndex =>
                {
                    using IDisposable scope = TensorExecutionContext.Push(
                        new TorchDevice(TensorDevice.Cuda, deviceIndex));
                    start.SignalAndWait();
                    tensor.EnsureCudaBFloat16Buffer(deviceIndex)
                        .CopyToCPU(actual[deviceIndex]);
                });

                Assert.Equal(expected, actual[0]);
                Assert.Equal(expected, actual[1]);
            }
            finally
            {
                tensor.InvalidateCudaBuffers();
            }
        }
        finally
        {
            Tensor.CudaDeviceIndices = previousIndices;
            Tensor.ExecutionDevice = previousDevice;
        }
    }

    [Fact]
    public void ZeroGradOnArenaSliceDoesNotClearAdjacentTensorGradient()
    {
        if (Tensor.CudaDeviceCount == 0)
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        NativeCudaArena<float>? arena = null;
        Tensor? left = null;
        Tensor? right = null;
        try
        {
            Tensor.CudaDeviceIndices = [0];
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            NativeCudaDevice accelerator =
                ForgetMemoryV2Cuda.GetAccelerator(0);
            arena = new NativeCudaArena<float>(accelerator, 8);
            left = Tensor.Zeros(4);
            right = Tensor.Zeros(4);
            left.BindCudaGradientArena(0, arena.Slice(0, 4));
            right.BindCudaGradientArena(0, arena.Slice(4, 4));
            left.SetCudaGradient([1f, 2f, 3f, 4f], 0);
            right.SetCudaGradient([5f, 6f, 7f, 8f], 0);

            left.ZeroGrad();

            Assert.Equal([0f, 0f, 0f, 0f], left.Grad);
            Assert.Equal([5f, 6f, 7f, 8f], right.Grad);
        }
        finally
        {
            if (arena is not null)
            {
                left?.UnbindCudaGradientArena(0, arena);
                right?.UnbindCudaGradientArena(0, arena);
                arena.Dispose();
            }
            Tensor.CudaDeviceIndices = previousIndices;
            Tensor.ExecutionDevice = previousDevice;
        }
    }

    [Fact]
    public void InferenceScopeContinuesCleanupAfterResourceFailure()
    {
        var first = new CallbackDisposable(
            () => throw new InvalidOperationException("expected cleanup failure"));
        var second = new CallbackDisposable();
        var third = new CallbackDisposable();
        CudaInferenceScope scope = CudaInferenceScope.Begin();
        Assert.True(CudaInferenceScope.TrackResource(first));
        Assert.True(CudaInferenceScope.TrackResource(second));
        Assert.True(CudaInferenceScope.TrackResource(third));

        Exception? exception = Record.Exception(scope.Dispose);

        Assert.NotNull(exception);
        Assert.True(first.DisposeCalled);
        Assert.True(second.DisposeCalled);
        Assert.True(third.DisposeCalled);
    }

    private sealed class CallbackDisposable(Action? onDispose = null)
        : IDisposable
    {
        private readonly Action? _onDispose = onDispose;

        internal bool DisposeCalled { get; private set; }

        public void Dispose()
        {
            DisposeCalled = true;
            _onDispose?.Invoke();
        }
    }
}
