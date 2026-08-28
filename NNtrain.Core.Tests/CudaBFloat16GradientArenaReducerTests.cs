using NNtrain;
using NNtrain.Cuda.Execution;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class CudaBFloat16GradientArenaReducerTests
{
    [Fact]
    public void ExecutionSessionCommunicationStreamsAreBorrowedAndRemainUsable()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndices = [0, 1];
            using var execution = new ExecutionSession(
                new ExecutionOptions
                {
                    Device = ExecutionDeviceKind.Cuda,
                    CudaDevices = new DeviceSet(0, 1),
                    Precision = PrecisionPolicy.BFloat16,
                },
                [
                    CudaExecutionLaneFactory.Create(0),
                    CudaExecutionLaneFactory.Create(1),
                ]);
            using IDisposable executionScope = execution.Enter();
            var parameter = new Parameter(
                [1f, -2f, 3f, -4f],
                [4],
                "bf16.borrowed-stream",
                WeightDecayPolicy.Apply,
                dtype: TensorDType.BFloat16);
            parameter.T.to(new TorchDevice(TensorDevice.Cuda, 0));

            using (var reducer = new CudaBFloat16GradientAllReducePlan(
                [parameter],
                [0, 1],
                useBFloat16GradientStorage: true))
            {
                Assert.False(reducer.OwnsCommunicationStream(0));
                Assert.False(reducer.OwnsCommunicationStream(1));
            }

            foreach (int device in new[] { 0, 1 })
            {
                var lane = Assert.IsType<CudaExecutionLane>(
                    execution.GetRequiredLane(
                        ExecutionDeviceKind.Cuda,
                        device));
                Assert.NotEqual(nint.Zero, lane.CommunicationStreamHandle);
                lane.SynchronizeCommunicationStream();
            }
            parameter.T.InvalidateCudaBuffers();
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }

    [Fact]
    public void DirectArenaPreservesAdjacentSliceAndRecoversAfterAbort()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndices = [0, 1];
            using var execution = new ExecutionSession(
                new ExecutionOptions
                {
                    Device = ExecutionDeviceKind.Cuda,
                    CudaDevices = new DeviceSet(0, 1),
                    Precision = PrecisionPolicy.BFloat16,
                },
                [
                    CudaExecutionLaneFactory.Create(0),
                    CudaExecutionLaneFactory.Create(1),
                ]);
            using IDisposable executionScope = execution.Enter();
            Parameter[] parameters =
            [
                new Parameter(
                    [0f, 0f, 0f],
                    [3],
                    "bf16.arena.left",
                    WeightDecayPolicy.Apply,
                    dtype: TensorDType.BFloat16),
                new Parameter(
                    [0f, 0f],
                    [2],
                    "bf16.arena.right",
                    WeightDecayPolicy.Apply,
                    dtype: TensorDType.BFloat16),
            ];
            foreach (Parameter parameter in parameters)
                parameter.T.to(new TorchDevice(TensorDevice.Cuda, 0));
            using var reducer = new CudaBFloat16GradientAllReducePlan(
                parameters,
                [0, 1],
                new CudaDispatchPolicy { GradientBucketElements = 32 },
                useBFloat16GradientStorage: true);

            RunStep(
                reducer,
                parameters,
                [
                    [[1f, 2f, 3f], [4f, 5f]],
                    [[0.5f, 1f, 1.5f], [2f, 3f]],
                ]);
            Assert.Equal(
                [1.5f, 3f, 4.5f],
                parameters[0].T.Grad.ToArray());
            Assert.Equal([6f, 8f], parameters[1].T.Grad.ToArray());
            Assert.Equal(0, reducer.ManagedLocalPackSubmissionCount);

            Assert.True(parameters[1].T.TryGetCudaBFloat16GradientBuffer(
                0,
                out NativeCudaBuffer<ushort>? adjacent));
            var before = new ushort[2];
            adjacent!.CopyToCPU(before);
            parameters[0].ZeroGrad();
            var after = new ushort[2];
            adjacent.CopyToCPU(after);
            Assert.Equal(before, after);

            foreach (Parameter parameter in parameters)
                parameter.ZeroGrad();
            long failedStep = reducer.BeginStep();
            reducer.BeginDeviceStep(failedStep, 0);
            Publish(
                reducer,
                failedStep,
                0,
                parameters[0],
                [7f, 8f, 9f]);
            Assert.Throws<InvalidOperationException>(() =>
                reducer.NotifyGradientReady(
                    parameters[0].T,
                    0,
                    failedStep));
            reducer.Abort(failedStep);
            Assert.All(
                parameters,
                parameter => Assert.False(
                    parameter.T.GetCudaGradientCoherenceSnapshot()
                        .PendingStamp.IsValid));

            foreach (Parameter parameter in parameters)
                parameter.ZeroGrad();
            RunStep(
                reducer,
                parameters,
                [
                    [[2f, 2f, 2f], [1f, 1f]],
                    [[3f, 3f, 3f], [4f, 4f]],
                ]);
            Assert.Equal([5f, 5f, 5f], parameters[0].T.Grad.ToArray());
            Assert.Equal([5f, 5f], parameters[1].T.Grad.ToArray());
            Assert.All(
                parameters,
                parameter => Assert.Equal(
                    CudaGradientCoherenceKind.Reduced,
                    parameter.T.GetCudaGradientCoherenceSnapshot().Kind));
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }

    private static void RunStep(
        CudaBFloat16GradientAllReducePlan reducer,
        IReadOnlyList<Parameter> parameters,
        float[][][] gradients)
    {
        long stepId = reducer.BeginStep();
        try
        {
            for (int device = 0; device < 2; device++)
            {
                reducer.BeginDeviceStep(stepId, device);
                for (int parameter = 0;
                    parameter < parameters.Count;
                    parameter++)
                {
                    Publish(
                        reducer,
                        stepId,
                        device,
                        parameters[parameter],
                        gradients[device][parameter]);
                }
            }
            reducer.Complete(stepId);
        }
        catch
        {
            reducer.Abort(stepId);
            throw;
        }
    }

    private static void Publish(
        CudaBFloat16GradientAllReducePlan reducer,
        long stepId,
        int deviceIndex,
        Parameter parameter,
        float[] values)
    {
        Assert.True(parameter.T.TryGetCudaBFloat16GradientBuffer(
            deviceIndex,
            out NativeCudaBuffer<ushort>? buffer));
        ushort[] encoded = values
            .Select(TensorStorageCodec.EncodeBFloat16)
            .ToArray();
        buffer!.CopyFromCPU(encoded);
        buffer.MarkGradientStorageDirty();
        reducer.NotifyGradientReady(parameter.T, deviceIndex, stepId);
    }
}
