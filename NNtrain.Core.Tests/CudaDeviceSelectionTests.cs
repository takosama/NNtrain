using NNtrain;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class CudaDeviceSelectionTests
{
    [Fact]
    public void ActiveBFloat16PolicyHonorsLossSeedsConcurrentlyOnTwoDevices()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousDevices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = [0, 1];
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            float[] values = Enumerable.Range(0, 32)
                .Select(index => (index - 16) * 0.03125f).ToArray();
            var errors = new float[2];
            Parallel.For(0, 2, deviceIndex =>
            {
                using IDisposable policyScope =
                    TensorExecutionContext.PushPrecisionPolicy(
                        PrecisionPolicy.BFloat16);
                using IDisposable deviceScope = TensorExecutionContext.Push(
                    new TorchDevice(TensorDevice.Cuda, deviceIndex));

                float[] Run(float seed)
                {
                    var logits = new Tensor(
                        values, [2, 16], dtype: TensorDType.BFloat16);
                    logits.CrossEntropyWithLogits([3, 7])
                        .BackwardAndRelease([seed]);
                    return logits.Grad.ToArray();
                }

                float[] full = Run(1f);
                float[] half = Run(0.5f);
                errors[deviceIndex] = full.Zip(
                    half,
                    (expected, actual) => MathF.Abs(expected - 2f * actual))
                    .Max();
            });

            Assert.All(errors, error => Assert.InRange(error, 0f, 1e-6f));
        }
        finally
        {
            Tensor.CudaDeviceIndices = previousDevices;
            Tensor.ExecutionDevice = previousDevice;
        }
    }

    [Theory]
    [InlineData(4, 8)]
    [InlineData(8, 16)]
    [InlineData(16, 8)]
    [InlineData(8, 32)]
    public void DirectBFloat16LinearInputGradientMatchesRowShards(
        int inputWidth,
        int outputWidth)
    {
        if (Tensor.CudaDeviceCount == 0)
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousDevices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = [0];
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            const int rows = 8;
            float[] inputs = Enumerable.Range(0, rows * inputWidth)
                .Select(index => ((index * 17) % 23 - 11) * 0.03125f)
                .ToArray();
            float[] weights = Enumerable.Range(0, outputWidth * inputWidth)
                .Select(index => ((index * 13) % 19 - 9) * 0.015625f)
                .ToArray();
            float[] upstream = Enumerable.Range(0, rows * outputWidth)
                .Select(index => ((index * 11) % 29 - 14) * 0.0078125f)
                .ToArray();

            float[] Run(bool sharded)
            {
                var weight = new Tensor(
                    weights,
                    [outputWidth, inputWidth],
                    dtype: TensorDType.BFloat16);
                var bias = new Tensor(
                    new float[outputWidth],
                    [outputWidth],
                    dtype: TensorDType.BFloat16);
                if (!sharded)
                {
                    var input = new Tensor(
                        inputs,
                        [rows, inputWidth],
                        dtype: TensorDType.BFloat16);
                    input.LinearLastDim(weight, bias, applyRelu: false)
                        .BackwardAndRelease(upstream);
                    return input.Grad.ToArray();
                }

                var result = new float[inputs.Length];
                for (int shard = 0; shard < 2; shard++)
                {
                    int inputOffset = shard * rows / 2 * inputWidth;
                    int gradientOffset = shard * rows / 2 * outputWidth;
                    var input = new Tensor(
                        inputs.AsSpan(
                            inputOffset,
                            rows / 2 * inputWidth).ToArray(),
                        [rows / 2, inputWidth],
                        dtype: TensorDType.BFloat16);
                    input.LinearLastDim(weight, bias, applyRelu: false)
                        .BackwardAndRelease(
                            upstream.AsSpan(
                                gradientOffset,
                                rows / 2 * outputWidth).ToArray());
                    input.Grad.ToArray().CopyTo(
                        result.AsSpan(inputOffset, rows / 2 * inputWidth));
                }
                return result;
            }

            float[] whole = Run(sharded: false);
            float[] split = Run(sharded: true);
            for (int index = 0; index < whole.Length; index++)
            {
                Assert.True(
                    MathF.Abs(whole[index] - split[index]) <= 1e-6f,
                    $"index={index}, whole={whole[index]:R}, " +
                    $"split={split[index]:R}");
            }
        }
        finally
        {
            Tensor.CudaDeviceIndices = previousDevices;
            Tensor.ExecutionDevice = previousDevice;
        }
    }

    [Fact]
    public void BFloat16CrossEntropyBackwardHonorsScalarSeed()
    {
        if (Tensor.CudaDeviceCount == 0)
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousDevices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = [0];
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            float[] values = Enumerable.Range(0, 32)
                .Select(index => (index - 16) * 0.03125f).ToArray();

            float[] Run(float seed)
            {
                var logits = new Tensor(
                    values, [2, 16], dtype: TensorDType.BFloat16);
                Tensor loss = logits.CrossEntropyWithLogits([3, 7]);
                loss.BackwardAndRelease([seed]);
                return logits.Grad.ToArray();
            }

            float[] full = Run(1f);
            float[] half = Run(0.5f);
            for (int index = 0; index < full.Length; index++)
                Assert.InRange(MathF.Abs(half[index] * 2f - full[index]), 0f, 1e-6f);
        }
        finally
        {
            Tensor.CudaDeviceIndices = previousDevices;
            Tensor.ExecutionDevice = previousDevice;
        }
    }

    [Fact]
    public void BFloat16LinearCrossEntropyShardSeedsMatchWholeBatch()
    {
        if (Tensor.CudaDeviceCount == 0)
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousDevices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = [0];
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            float[] inputs = Enumerable.Range(0, 64)
                .Select(index => ((index * 17) % 31 - 15) * 0.03125f)
                .ToArray();
            float[] weights = Enumerable.Range(0, 256)
                .Select(index => ((index * 13) % 29 - 14) * 0.015625f)
                .ToArray();
            int[] labels = [2, 3, 4, 5, 6, 7, 8, 9];

            float[] Run(bool sharded)
            {
                var weight = new Tensor(
                    weights, [32, 8], dtype: TensorDType.BFloat16);
                var bias = new Tensor(
                    new float[32], [32], dtype: TensorDType.BFloat16);
                if (!sharded)
                {
                    var input = new Tensor(
                        inputs, [8, 8], dtype: TensorDType.BFloat16);
                    input.LinearLastDim(weight, bias, applyRelu: false)
                        .CrossEntropyWithLogits(labels)
                        .BackwardAndRelease();
                }
                else
                {
                    for (int shard = 0; shard < 2; shard++)
                    {
                        var input = new Tensor(
                            inputs.AsSpan(shard * 32, 32).ToArray(), [4, 8],
                            dtype: TensorDType.BFloat16);
                        input.LinearLastDim(weight, bias, applyRelu: false)
                            .CrossEntropyWithLogits(
                                labels.AsSpan(shard * 4, 4).ToArray())
                            .BackwardAndRelease([0.5f]);
                    }
                }
                return weight.Grad.ToArray();
            }

            float[] whole = Run(sharded: false);
            float[] split = Run(sharded: true);
            for (int index = 0; index < whole.Length; index++)
                Assert.InRange(MathF.Abs(split[index] - whole[index]), 0f, 3e-3f);
        }
        finally
        {
            Tensor.CudaDeviceIndices = previousDevices;
            Tensor.ExecutionDevice = previousDevice;
        }
    }

    [Fact]
    public void BFloat16GradientPlanReducesResidentArenaSlices()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousDevices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = [0, 1];
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            var parameter = new Parameter(
                new float[16], [2, 8], "P", WeightDecayPolicy.Apply,
                TensorDType.BFloat16);
            var plan = new CudaBFloat16GradientAllReducePlan(
                [parameter], [0, 1]);
            float[] first;
            float[] second;
            try
            {
                long stepId = plan.BeginStep();

                first = Enumerable.Range(1, 16)
                    .Select(value => (float)value).ToArray();
                second = Enumerable.Range(1, 16)
                    .Select(value => value * 2f).ToArray();
                plan.BeginDeviceStep(stepId, 0);
                parameter.T.SetCudaGradient(first, 0);
                plan.NotifyGradientReady(parameter.T, 0, stepId);
                plan.BeginDeviceStep(stepId, 1);
                parameter.T.SetCudaGradient(second, 1);
                plan.NotifyGradientReady(parameter.T, 1, stepId);
                plan.Complete(stepId);
            }
            finally
            {
                plan.Dispose();
            }

            Assert.Equal(
                first.Zip(second, (left, right) => left + right),
                parameter.T.Grad);
        }
        finally
        {
            Tensor.CudaDeviceIndices = previousDevices;
            Tensor.ExecutionDevice = previousDevice;
        }
    }

    [Fact]
    public void HostGradientPipelineKeepsNativeDeviceSelectionCacheCoherent()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        const int sourceDevice = 1;
        const int destinationDevice = 0;
        const int length = 256;

        NativeCudaDevice source =
            ForgetMemoryV2Cuda.GetAccelerator(sourceDevice);
        NativeCudaDevice destination =
            ForgetMemoryV2Cuda.GetAccelerator(destinationDevice);
        using NativeCudaBuffer<float> localSource = destination.Allocate1D(
            Enumerable.Repeat(1f, length).ToArray());
        using NativeCudaBuffer<ushort> local =
            destination.Allocate1D<ushort>(length);
        using NativeCudaBuffer<float> remoteSource = source.Allocate1D(
            Enumerable.Repeat(2f, length).ToArray());
        using NativeCudaBuffer<ushort> remote =
            source.Allocate1D<ushort>(length);
        using NativeCudaBuffer<float> reduced =
            destination.Allocate1D<float>(length);
        using NativeCudaBuffer<float> addend = source.Allocate1D(
            Enumerable.Repeat(3f, length).ToArray());
        using NativeCudaBuffer<float> result =
            source.Allocate1D<float>(length);

        nint localReady = 0;
        nint remoteReady = 0;
        nint pipeline = 0;
        try
        {
            localReady = CudaGradientBuckets.CreateReadyEvent(
                destination, destinationDevice);
            remoteReady = CudaGradientBuckets.CreateReadyEvent(
                source, sourceDevice);
            CudaGradientBuckets.Pack(
                destinationDevice, destination, localSource, local, 0, length);
            CudaGradientBuckets.RecordReady(
                destinationDevice, destination, localReady);
            CudaGradientBuckets.Pack(
                sourceDevice, source, remoteSource, remote, 0, length);
            CudaGradientBuckets.RecordReady(
                sourceDevice, source, remoteReady);
            pipeline = CudaGradientBuckets.CreateHostPipeline(
                sourceDevice, destinationDevice, length);

            // Seed both device-selection caches with sourceDevice. Before the
            // regression fix, the native host pipeline used direct
            // cudaSetDevice calls and returned on destinationDevice without
            // updating cuda_runtime_bridge's thread-local selected device.
            source.Bind();
            CudaTensorNative.Add(
                sourceDevice,
                remoteSource.NativePtr,
                addend.NativePtr,
                result.NativePtr,
                length,
                bfloat16: false);
            source.Synchronize();

            CudaGradientBuckets.HostPipelineExchange(
                destination,
                pipeline,
                local,
                remote,
                reduced,
                length,
                squaredSum: 0,
                localReady,
                remoteReady);

            // This must bind sourceDevice again on the same managed/native
            // thread. A stale native selection cache launches this kernel on
            // destinationDevice with sourceDevice pointers and poisons the
            // context with cudaErrorIllegalAddress (700).
            CudaTensorNative.Add(
                sourceDevice,
                remoteSource.NativePtr,
                addend.NativePtr,
                result.NativePtr,
                length,
                bfloat16: false);
            source.Synchronize();
            destination.Synchronize();

            var actual = new float[length];
            result.CopyToCPU(actual);
            Assert.All(actual, value => Assert.Equal(5f, value));

            var reducedValues = new float[length];
            reduced.CopyToCPU(reducedValues);
            Assert.All(reducedValues, value => Assert.Equal(3f, value));
        }
        finally
        {
            CudaGradientBuckets.DestroyHostPipeline(destination, pipeline);
            CudaGradientBuckets.DestroyEvent(
                destination, destinationDevice, localReady);
            CudaGradientBuckets.DestroyEvent(source, sourceDevice, remoteReady);
        }
    }
}
