using NNtrain;
using NNtrain.Cuda.Execution;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class CudaBfp8TransientStorageTests
{
    [Theory]
    [InlineData(TensorPrecisionMode.Bfp8)]
    [InlineData(TensorPrecisionMode.Mix8_32)]
    public void WarmedTransformerStepReusesLaneActivationStorage(
        TensorPrecisionMode mode)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousDevices = Tensor.CudaDeviceIndices.ToArray();
        Parameter[] parameters = [];
        AdamW? optimizer = null;
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndices = [0];
            PrecisionPolicy precision = mode == TensorPrecisionMode.Bfp8
                ? PrecisionPolicy.Bfp8
                : PrecisionPolicy.Mix8_32;
            using var execution = new ExecutionSession(new ExecutionOptions
            {
                Device = ExecutionDeviceKind.Cuda,
                CudaDevices = new DeviceSet(0),
                Precision = precision,
            });
            execution.AttachLane(CudaExecutionLaneFactory.Create(0));
            using IDisposable executionScope = execution.Enter();

            var model = new GptRinWikiJp(
                vocabularySize: 64,
                contextLength: 4,
                dModel: 16,
                numHeads: 2,
                dHidden: 32,
                numLayers: 1,
                rng: new Random(1701 + (int)mode),
                initializationScale: 0.02f,
                dropout: 0.125f,
                dtype: TensorDType.Float32);
            model.to(mode, bfp8_block_size: 32);
            model.to(new TorchDevice(TensorDevice.Cuda, 0));
            parameters = model.Parameters().ToArray();
            optimizer = new AdamW(
                parameters,
                new AdamWOptions
                {
                    LearningRate = 1e-3f,
                    WeightDecay = 0f,
                });
            optimizer.prepare();

            int[] tokens = [1, 2, 3, 4, 5, 6, 7, 8];
            int[] targets = [2, 3, 4, 5, 6, 7, 8, 9];

            float RunStep()
            {
                optimizer.zero_grad();
                Tensor loss = model.forward_loss(
                    tokens,
                    targets,
                    batch_size: 2,
                    sequence_length: 4);
                float value = loss.item();
                loss.BackwardAndRelease();
                optimizer.step();
                ForgetMemoryV2Cuda.GetAccelerator(0).Synchronize();
                return value;
            }

            // The first graphs establish the live high-water mark and fill
            // every exact-size lane cache bucket used by forward, backward,
            // gradient publication, and the optimizer.
            float first = float.NaN;
            for (int warmup = 0; warmup < 4; warmup++)
                first = RunStep();

            NativeCudaDevice device = NativeCudaRuntime.GetDevice(0);
            long freeBefore = device.GetFreeMemory();
            NativeCudaAllocationTelemetry allocationBefore =
                NativeCudaRuntime.AllocationTelemetry;

            float last = first;
            for (int step = 0; step < 6; step++)
                last = RunStep();

            NativeCudaAllocationTelemetry allocations =
                NativeCudaRuntime.AllocationTelemetry - allocationBefore;
            long freeAfter = device.GetFreeMemory();

            Assert.True(float.IsFinite(first));
            Assert.True(float.IsFinite(last));
            Assert.Equal(0, allocations.AllocationCount);
            Assert.Equal(0, allocations.AllocationBytes);
            Assert.Equal(0, allocations.FreeCount);
            Assert.Equal(0, allocations.FreeBytes);
            Assert.True(
                freeAfter >= freeBefore - 1024 * 1024,
                $"Steady BFP8 training retained " +
                $"{freeBefore - freeAfter:N0} unexpected bytes.");
        }
        finally
        {
            optimizer?.DisposeCudaResources();
            foreach (Parameter parameter in parameters)
                parameter.T.InvalidateCudaBuffers();
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousDevices;
        }
    }
}
