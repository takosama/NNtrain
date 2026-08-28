using NNtrain;
using NNtrain.Cuda.Execution;
using NNtrain.Cuda.Memory;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class CudaGraphBatchInputsTests
{
    [Fact]
    public void UpdatesUseTwoCopiesStablePointersAndReleaseEveryResource()
    {
        WithCuda((lane, unusedSession) =>
        {
            const int length = 8;
            NativeCudaPinnedUploadTelemetry pinnedBefore =
                NativeCudaPinnedUploadTracker.Telemetry;
            var inputs = CudaGraphBatchInputs.Create(
                lane,
                length,
                vocabularySize: 32);
            nint inputPointer = inputs.InputPointer;
            nint targetPointer = inputs.TargetPointer;
            NativeCudaPinnedUploadTelemetry pinnedActive =
                NativeCudaPinnedUploadTracker.Telemetry - pinnedBefore;
            Assert.Equal(2, pinnedActive.ActiveSlotCount);
            Assert.Equal(2, pinnedActive.ActiveEventCount);
            Assert.Equal(2L * length * sizeof(int),
                pinnedActive.ActivePinnedBytes);

            int[] firstInput = Enumerable.Range(0, length).ToArray();
            int[] firstTarget = Enumerable.Range(1, length).ToArray();
            NativeCudaTransferTelemetry transferBefore =
                NativeCudaRuntime.TransferTelemetry;
            inputs.Update(
                firstInput,
                firstTarget,
                lane.ComputeStreamHandle);
            lane.SynchronizeComputeStream();
            NativeCudaTransferTelemetry firstTransfer =
                NativeCudaRuntime.TransferTelemetry - transferBefore;
            Assert.Equal(2, firstTransfer.HostToDeviceCopyCount);
            Assert.Equal(2L * length * sizeof(int),
                firstTransfer.HostToDeviceBytes);
            Assert.Equal(0, firstTransfer.DeviceToHostCopyCount);
            Assert.Equal(0, firstTransfer.DeviceToHostBytes);
            Assert.Equal(inputPointer, inputs.InputPointer);
            Assert.Equal(targetPointer, inputs.TargetPointer);

            var copiedInput = new int[length];
            var copiedTarget = new int[length];
            inputs.InputBuffer.CopyToCPU(copiedInput);
            inputs.TargetBuffer.CopyToCPU(copiedTarget);
            Assert.Equal(firstInput, copiedInput);
            Assert.Equal(firstTarget, copiedTarget);

            int[] secondInput = Enumerable.Range(8, length).ToArray();
            int[] secondTarget = Enumerable.Range(9, length).ToArray();
            transferBefore = NativeCudaRuntime.TransferTelemetry;
            inputs.Update(
                secondInput,
                secondTarget,
                lane.ComputeStreamHandle);
            lane.SynchronizeComputeStream();
            NativeCudaTransferTelemetry secondTransfer =
                NativeCudaRuntime.TransferTelemetry - transferBefore;
            Assert.Equal(2, secondTransfer.HostToDeviceCopyCount);
            Assert.Equal(2L * length * sizeof(int),
                secondTransfer.HostToDeviceBytes);
            Assert.Equal(0, secondTransfer.DeviceToHostCopyCount);
            Assert.Equal(inputPointer, inputs.InputPointer);
            Assert.Equal(targetPointer, inputs.TargetPointer);

            Array.Clear(copiedInput);
            Array.Clear(copiedTarget);
            inputs.InputBuffer.CopyToCPU(copiedInput);
            inputs.TargetBuffer.CopyToCPU(copiedTarget);
            Assert.Equal(secondInput, copiedInput);
            Assert.Equal(secondTarget, copiedTarget);
            Assert.Equal(
                new CudaGraphBatchInputsTelemetry(
                    UpdateCount: 2,
                    HostToDeviceCopyCount: 4,
                    HostToDeviceBytes: 4L * length * sizeof(int),
                    BorrowCount: 0,
                    ReturnCount: 0),
                inputs.Telemetry);

            inputs.Dispose();
            Assert.Equal(0, lane.Memory.AllocationCount);
            NativeCudaPinnedUploadTelemetry pinnedReleased =
                NativeCudaPinnedUploadTracker.Telemetry - pinnedBefore;
            Assert.Equal(0, pinnedReleased.ActiveSlotCount);
            Assert.Equal(0, pinnedReleased.ActiveEventCount);
            Assert.Equal(0, pinnedReleased.ActivePinnedBytes);
            Assert.Equal(2, pinnedReleased.CreatedSlotCount);
            Assert.Equal(2, pinnedReleased.DisposedSlotCount);
            Assert.Equal(2, pinnedReleased.HostAllocationCount);
            Assert.Equal(2, pinnedReleased.HostFreeCount);
            Assert.Equal(2, pinnedReleased.EventCreateCount);
            Assert.Equal(2, pinnedReleased.EventDestroyCount);
        });
    }

    [Fact]
    public void CaptureBorrowReturnsFixedBuffersWithoutTransferOrFree()
    {
        WithCuda((lane, unusedSession) =>
        {
            int[] input = [1, 2, 3, 4];
            int[] target = [2, 3, 4, 5];
            using CudaGraphBatchInputs inputs = CudaGraphBatchInputs.Create(
                lane,
                input.Length,
                vocabularySize: 16);
            inputs.Update(input, target, lane.ComputeStreamHandle);
            lane.SynchronizeComputeStream();
            NativeCudaTransferTelemetry transferBefore =
                NativeCudaRuntime.TransferTelemetry;
            CudaMemoryManagerTelemetry memoryBefore = lane.Memory.Telemetry;

            using (CudaGraphBatchInputs.CudaGraphBatchInputCaptureScope capture =
                   inputs.PushCaptureScope())
            {
                Assert.Same(input, CudaGraphBatchInputs.RetainOrClone(input));
                Assert.Same(target, CudaGraphBatchInputs.RetainOrClone(target));
                int[] unrelated = [9, 9, 9, 9];
                Assert.NotSame(
                    unrelated,
                    CudaGraphBatchInputs.RetainOrClone(unrelated));

                NativeCudaBuffer<int> inputBuffer =
                    Tensor.RentCudaIntBuffer(0, input);
                NativeCudaBuffer<int> targetBuffer =
                    Tensor.RentCudaIntBuffer(0, target);
                Assert.Same(inputs.InputBuffer, inputBuffer);
                Assert.Same(inputs.TargetBuffer, targetBuffer);
                Assert.Equal(inputs.InputPointer, inputBuffer.NativePtr);
                Assert.Equal(inputs.TargetPointer, targetBuffer.NativePtr);

                NativeCudaDevice device = NativeCudaRuntime.GetDevice(0);
                Tensor.ReturnCudaIntBuffer(device, targetBuffer);
                Tensor.ReturnCudaIntBuffer(device, inputBuffer);
            }

            Assert.Equal(memoryBefore, lane.Memory.Telemetry);
            NativeCudaTransferTelemetry transfer =
                NativeCudaRuntime.TransferTelemetry - transferBefore;
            Assert.Equal(0, transfer.HostToDeviceCopyCount);
            Assert.Equal(0, transfer.HostToDeviceBytes);
            Assert.Equal(0, transfer.DeviceToHostCopyCount);
            Assert.Equal(2, inputs.Telemetry.BorrowCount);
            Assert.Equal(2, inputs.Telemetry.ReturnCount);

            int[] ordinaryRetained = CudaGraphBatchInputs.RetainOrClone(input);
            Assert.NotSame(input, ordinaryRetained);
            NativeCudaBuffer<int> ordinary =
                Tensor.RentCudaIntBuffer(0, ordinaryRetained);
            Assert.NotSame(inputs.InputBuffer, ordinary);
            Tensor.ReturnCudaIntBuffer(NativeCudaRuntime.GetDevice(0), ordinary);
        });
    }

    [Fact]
    public void ValidationRejectsBadBatchesBeforeAnyTransfer()
    {
        WithCuda((lane, unusedSession) =>
        {
            using CudaGraphBatchInputs inputs = CudaGraphBatchInputs.Create(
                lane,
                length: 4,
                vocabularySize: 8,
                ignoreIndex: -7);
            NativeCudaTransferTelemetry before =
                NativeCudaRuntime.TransferTelemetry;

            Assert.Throws<ArgumentException>(() => inputs.Update(
                [0, 1, 2],
                [1, 2, 3, 4],
                lane.ComputeStreamHandle));
            Assert.Throws<ArgumentOutOfRangeException>(() => inputs.Update(
                [0, 1, 2, 8],
                [1, 2, 3, 4],
                lane.ComputeStreamHandle));
            Assert.Throws<ArgumentOutOfRangeException>(() => inputs.Update(
                [0, 1, 2, 3],
                [1, 2, -1, 4],
                lane.ComputeStreamHandle));
            Assert.Throws<ArgumentException>(() => inputs.Update(
                [0, 1, 2, 3],
                [-7, -7, -7, -7],
                lane.ComputeStreamHandle));
            int[] aliased = [0, 1, 2, 3];
            Assert.Throws<ArgumentException>(() => inputs.Update(
                aliased,
                aliased,
                lane.ComputeStreamHandle));
            Assert.Throws<ArgumentException>(() => inputs.Update(
                [0, 1, 2, 3],
                [1, 2, 3, 4],
                lane.CommunicationStreamHandle));

            Assert.Equal(
                before,
                NativeCudaRuntime.TransferTelemetry);
            Assert.Equal(default, inputs.Telemetry);

            int[] validInput = [0, 1, 2, 3];
            int[] validTarget = [1, 2, -7, 4];
            inputs.Update(validInput, validTarget, lane.ComputeStreamHandle);
            using CudaGraphBatchInputs.CudaGraphBatchInputCaptureScope capture =
                inputs.PushCaptureScope();
            Assert.Throws<InvalidOperationException>(() => inputs.Update(
                validInput,
                validTarget,
                lane.ComputeStreamHandle));
        });
    }

    [Fact]
    public void CaptureScopeRejectsNestedWrongThreadAndDoubleBorrow()
    {
        WithCuda((lane, unusedSession) =>
        {
            int[] input = [1, 2, 3, 4];
            int[] target = [2, 3, 4, 5];
            using CudaGraphBatchInputs inputs = CudaGraphBatchInputs.Create(
                lane,
                input.Length,
                vocabularySize: 16);
            inputs.Update(input, target, lane.ComputeStreamHandle);
            NativeCudaDevice device = NativeCudaRuntime.GetDevice(0);
            var capture = inputs.PushCaptureScope();
            Assert.Throws<InvalidOperationException>(inputs.PushCaptureScope);

            NativeCudaBuffer<int> borrowed =
                Tensor.RentCudaIntBuffer(0, input);
            Assert.Throws<InvalidOperationException>(() =>
                Tensor.RentCudaIntBuffer(0, input));
            Exception? wrongThread = null;
            var worker = new Thread(() =>
            {
                wrongThread = Record.Exception(capture.Dispose);
            });
            worker.Start();
            Assert.True(worker.Join(TimeSpan.FromSeconds(5)));
            Assert.IsType<InvalidOperationException>(wrongThread);

            Tensor.ReturnCudaIntBuffer(device, borrowed);
            Assert.Throws<InvalidOperationException>(() =>
                Tensor.ReturnCudaIntBuffer(device, borrowed));
            capture.Dispose();
            capture.Dispose();

            var outstanding = inputs.PushCaptureScope();
            _ = Tensor.RentCudaIntBuffer(0, target);
            Assert.Throws<InvalidOperationException>(outstanding.Dispose);

            using CudaGraphBatchInputs.CudaGraphBatchInputCaptureScope recovered =
                inputs.PushCaptureScope();
            NativeCudaBuffer<int> targetBuffer =
                Tensor.RentCudaIntBuffer(0, target);
            Tensor.ReturnCudaIntBuffer(device, targetBuffer);
        });
    }

    [Fact]
    public void EmbeddingAndCrossEntropyUseUpdatedFixedSources()
    {
        WithCuda((lane, unusedSession) =>
        {
            const int vocabulary = 8;
            const int length = 4;
            const int width = 3;
            float[] tableValues = Enumerable.Range(0, vocabulary)
                .SelectMany(row => Enumerable.Repeat((float)row, width))
                .ToArray();
            var table = new Tensor(
                tableValues,
                [vocabulary, width],
                dtype: TensorDType.BFloat16);
            var logits = new Tensor(
                Enumerable.Range(0, length * vocabulary)
                    .Select(index => (index % vocabulary) * 0.1f)
                    .ToArray(),
                [length, vocabulary],
                dtype: TensorDType.BFloat16);
            table.to(new TorchDevice(TensorDevice.Cuda, 0));
            logits.to(new TorchDevice(TensorDevice.Cuda, 0));
            _ = table.EnsureCudaBFloat16Buffer(0);
            _ = logits.EnsureCudaBFloat16Buffer(0);
            using CudaGraphBatchInputs inputs = CudaGraphBatchInputs.Create(
                lane,
                length,
                vocabulary);

            float[] first = RunForward(
                inputs,
                table,
                logits,
                input: [0, 1, 2, 3],
                target: [1, 2, 3, 4],
                lane);
            nint stableInput = inputs.InputPointer;
            nint stableTarget = inputs.TargetPointer;
            float[] second = RunForward(
                inputs,
                table,
                logits,
                input: [4, 5, 6, 7],
                target: [5, 6, 7, 0],
                lane);

            Assert.NotEqual(first, second);
            Assert.Equal(
                Enumerable.Range(0, 4)
                    .SelectMany(row => Enumerable.Repeat((float)row, width)),
                first,
                new FloatToleranceComparer(0.02f));
            Assert.Equal(
                Enumerable.Range(4, 4)
                    .SelectMany(row => Enumerable.Repeat((float)row, width)),
                second,
                new FloatToleranceComparer(0.02f));
            Assert.Equal(stableInput, inputs.InputPointer);
            Assert.Equal(stableTarget, inputs.TargetPointer);
            Assert.Equal(4, inputs.Telemetry.BorrowCount);
            Assert.Equal(4, inputs.Telemetry.ReturnCount);

            table.InvalidateCudaBuffers();
            logits.InvalidateCudaBuffers();
        });
    }

    [Fact]
    public void CapturedGraphReplaysAgainstUpdatedStableInputPointer()
    {
        WithCuda((lane, unusedSession) =>
        {
            if (!lane.CudaCapabilities.Supports(CudaKernelFeature.CudaGraphs))
                return;

            const int vocabulary = 8;
            const int length = 4;
            const int width = 2;
            float[] tableValues = Enumerable.Range(0, vocabulary)
                .SelectMany(row => Enumerable.Repeat((float)row, width))
                .ToArray();
            var table = new Tensor(
                tableValues,
                [vocabulary, width],
                dtype: TensorDType.BFloat16);
            table.to(new TorchDevice(TensorDevice.Cuda, 0));
            NativeCudaBuffer<ushort> tableBuffer =
                table.EnsureCudaBFloat16Buffer(0);
            NativeCudaDevice device = NativeCudaRuntime.GetDevice(0);
            using NativeCudaBuffer<ushort> output = device.Allocate1D<ushort>(
                length * width,
                CudaMemoryKind.Persistent);
            using CudaGraphBatchInputs inputs = CudaGraphBatchInputs.Create(
                lane,
                length,
                vocabulary);
            int[] captureInput = [0, 1, 2, 3];
            int[] captureTarget = [1, 2, 3, 4];
            inputs.Update(
                captureInput,
                captureTarget,
                lane.ComputeStreamHandle);
            lane.SynchronizeComputeStream();

            CudaGraphExecutable graph;
            using (CudaGraphBatchInputs.CudaGraphBatchInputCaptureScope scope =
                   inputs.PushCaptureScope())
            {
                int[] retained = CudaGraphBatchInputs.RetainOrClone(
                    captureInput);
                NativeCudaBuffer<int> fixedInput =
                    Tensor.RentCudaIntBuffer(0, retained);
                Assert.Equal(inputs.InputPointer, fixedInput.NativePtr);
                graph = CudaGraphExecutable.Capture(lane, () =>
                    CudaTensorNative.Embedding(
                        0,
                        tableBuffer.NativePtr,
                        fixedInput.NativePtr,
                        output.NativePtr,
                        length * width,
                        width,
                        bfloat16: true));
                Tensor.ReturnCudaIntBuffer(device, fixedInput);
            }

            float[] first = LaunchWith(
                inputs,
                graph,
                output,
                input: [0, 1, 2, 3],
                target: [1, 2, 3, 4],
                lane);
            nint stablePointer = inputs.InputPointer;
            float[] second = LaunchWith(
                inputs,
                graph,
                output,
                input: [4, 5, 6, 7],
                target: [5, 6, 7, 0],
                lane);

            Assert.Equal(stablePointer, inputs.InputPointer);
            Assert.Equal(
                Enumerable.Range(0, 4)
                    .SelectMany(row => Enumerable.Repeat((float)row, width)),
                first,
                new FloatToleranceComparer(0.02f));
            Assert.Equal(
                Enumerable.Range(4, 4)
                    .SelectMany(row => Enumerable.Repeat((float)row, width)),
                second,
                new FloatToleranceComparer(0.02f));
            graph.DisposeChecked();
            table.InvalidateCudaBuffers();
        });
    }

    private static float[] RunForward(
        CudaGraphBatchInputs inputs,
        Tensor table,
        Tensor logits,
        int[] input,
        int[] target,
        IStreamExecutionLane lane)
    {
        NativeCudaTransferTelemetry before =
            NativeCudaRuntime.TransferTelemetry;
        inputs.Update(input, target, lane.ComputeStreamHandle);
        Tensor embedding;
        Tensor loss;
        using (CudaGraphBatchInputs.CudaGraphBatchInputCaptureScope capture =
               inputs.PushCaptureScope())
        using (AutogradContext.NoGrad())
        {
            embedding = table.EmbeddingLookup(input, input.Length);
            loss = logits.CrossEntropyWithLogits(target);
        }
        lane.SynchronizeComputeStream();
        NativeCudaTransferTelemetry transfer =
            NativeCudaRuntime.TransferTelemetry - before;
        Assert.Equal(2, transfer.HostToDeviceCopyCount);
        Assert.Equal(2L * input.Length * sizeof(int),
            transfer.HostToDeviceBytes);
        Assert.Equal(0, transfer.DeviceToHostCopyCount);

        float[] result = embedding.Data.ToArray();
        _ = loss.Data[0];
        embedding.InvalidateCudaBuffers();
        loss.InvalidateCudaBuffers();
        return result;
    }

    private static float[] LaunchWith(
        CudaGraphBatchInputs inputs,
        CudaGraphExecutable graph,
        NativeCudaBuffer<ushort> output,
        int[] input,
        int[] target,
        IStreamExecutionLane lane)
    {
        NativeCudaTransferTelemetry before =
            NativeCudaRuntime.TransferTelemetry;
        inputs.Update(input, target, lane.ComputeStreamHandle);
        graph.Launch();
        lane.SynchronizeComputeStream();
        NativeCudaTransferTelemetry transfer =
            NativeCudaRuntime.TransferTelemetry - before;
        Assert.Equal(2, transfer.HostToDeviceCopyCount);
        Assert.Equal(2L * input.Length * sizeof(int),
            transfer.HostToDeviceBytes);
        Assert.Equal(0, transfer.DeviceToHostCopyCount);

        var encoded = new ushort[output.Length];
        output.CopyToCPU(encoded);
        return encoded.Select(static bits =>
            BitConverter.Int32BitsToSingle(bits << 16)).ToArray();
    }

    private static void WithCuda(
        Action<CudaExecutionLane, ExecutionSession> action)
    {
        if (Tensor.CudaDeviceCount == 0)
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousDevices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = [0];
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            var lane = CudaExecutionLaneFactory.Create(0);
            using var session = new ExecutionSession(
                new ExecutionOptions
                {
                    Device = ExecutionDeviceKind.Cuda,
                    CudaDevices = new DeviceSet(0),
                    Precision = PrecisionPolicy.BFloat16,
                },
                [lane]);
            using IDisposable scope = session.Enter();
            action(lane, session);
        }
        finally
        {
            Tensor.CudaDeviceIndices = previousDevices;
            Tensor.ExecutionDevice = previousDevice;
        }
    }

    private sealed class FloatToleranceComparer(float tolerance)
        : IEqualityComparer<float>
    {
        public bool Equals(float left, float right)
            => MathF.Abs(left - right) <= tolerance;

        public int GetHashCode(float value) => 0;
    }
}
