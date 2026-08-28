using System.Runtime.InteropServices;
using NNtrain;
using NNtrain.Cuda.Execution;
using NNtrain.Cuda.Interop;
using NNtrain.Cuda.Memory;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class CudaGraphTests
{
    [Fact]
    public void CapturedDropoutForwardAndBackwardShareMaskAcrossTailAndChangePerReplay()
    {
        if (CudaNativeGateway.DeviceCount(out int deviceCount) != 0
            || deviceCount == 0)
        {
            return;
        }

        const int length = 4099;
        const float probability = 0.37f;
        const ulong operationSeed = 0xf17e_cafe_1234_9876UL;
        using CudaExecutionLane lane = CudaExecutionLaneFactory.Create(0);
        CudaMemoryLease input = PersistentFloatBuffer(lane, length);
        CudaMemoryLease output = PersistentFloatBuffer(lane, length);
        CudaMemoryLease outputGradient = PersistentFloatBuffer(lane, length);
        CudaMemoryLease inputGradient = PersistentFloatBuffer(lane, length);
        Upload(lane, input, Enumerable.Repeat(1f, length).ToArray());
        Upload(lane, outputGradient, Enumerable.Repeat(1f, length).ToArray());

        CudaGraphRngState rng = CudaGraphRngState.Create(
            lane,
            initialCounter: 41);
        CudaGraphExecutable graph = CudaGraphExecutable.Capture(
            lane,
            () =>
            {
                Assert.Equal(
                    0,
                    CudaNativeGateway.MemsetAsync(
                        lane.DeviceIndex,
                        inputGradient.Pointer,
                        0,
                        checked((nuint)length * sizeof(float)),
                        lane.ComputeStreamHandle));
                rng.EnqueueAdvance();
                rng.EnqueueDropoutForwardFloat32(
                    input.Pointer,
                    output.Pointer,
                    length,
                    probability,
                    operationSeed);
                rng.EnqueueDropoutBackwardFloat32(
                    outputGradient.Pointer,
                    inputGradient.Pointer,
                    length,
                    probability,
                    operationSeed);
            });

        float[] firstForward = ReplayAndRead(
            lane, graph, output, length);
        float[] firstBackward = Read(lane, inputGradient, length);
        float[] secondForward = ReplayAndRead(
            lane, graph, output, length);
        float[] secondBackward = Read(lane, inputGradient, length);

        Assert.Equal(firstForward, firstBackward);
        Assert.Equal(secondForward, secondBackward);
        Assert.All(
            firstForward,
            value => Assert.True(value == 0f || value > 1f));
        int changed = firstForward.Zip(secondForward)
            .Count(pair => pair.First != pair.Second);
        Assert.InRange(changed, length / 4, length * 3 / 4);
    }

    [Fact]
    public void CapturedFusedLayerNormSharesMaskAndChangesItPerReplayAt512Columns()
    {
        if (CudaNativeGateway.DeviceCount(out int deviceCount) != 0
            || deviceCount == 0)
        {
            return;
        }

        const int rows = 16;
        const int columns = 512;
        const int length = rows * columns;
        const float probability = 0.25f;
        const float keepScale = 1f / (1f - probability);
        const uint threshold = (uint)(
            probability * ((double)uint.MaxValue + 1d));
        const ulong operationSeed = 0x4c4e_4655_5345_4431UL;
        using CudaExecutionLane lane = CudaExecutionLaneFactory.Create(0);
        CudaMemoryLease residual = PersistentFloatBuffer(lane, length);
        CudaMemoryLease branch = PersistentFloatBuffer(lane, length);
        CudaMemoryLease gamma = PersistentFloatBuffer(lane, columns);
        CudaMemoryLease beta = PersistentFloatBuffer(lane, columns);
        CudaMemoryLease output = PersistentFloatBuffer(lane, length);
        CudaMemoryLease means = PersistentFloatBuffer(lane, rows);
        CudaMemoryLease inverses = PersistentFloatBuffer(lane, rows);
        CudaMemoryLease outputGradient = PersistentFloatBuffer(lane, length);
        CudaMemoryLease residualGradient = PersistentFloatBuffer(lane, length);
        CudaMemoryLease branchGradient = PersistentFloatBuffer(lane, length);
        CudaMemoryLease gammaGradient = PersistentFloatBuffer(lane, columns);
        CudaMemoryLease betaGradient = PersistentFloatBuffer(lane, columns);
        CudaMemoryLease parameterScratch = PersistentFloatBuffer(
            lane,
            2 * columns);
        Upload(lane, residual, Enumerable.Repeat(0.25f, length).ToArray());
        Upload(lane, branch, Enumerable.Repeat(1f, length).ToArray());
        Upload(lane, gamma, Enumerable.Repeat(1f, columns).ToArray());
        Upload(lane, beta, new float[columns]);
        Upload(
            lane,
            outputGradient,
            Enumerable.Range(0, length)
                .Select(index => ((index * 17) % 53 - 26) * 0.03125f)
                .ToArray());

        CudaGraphRngState rng = CudaGraphRngState.Create(lane, 91);
        CudaGraphExecutable graph = CudaGraphExecutable.Capture(
            lane,
            () =>
            {
                Zero(lane, residualGradient, length);
                Zero(lane, branchGradient, length);
                Zero(lane, gammaGradient, columns);
                Zero(lane, betaGradient, columns);
                rng.EnqueueAdvance();
                rng.EnqueueResidualDropoutLayerNormForwardFloat32(
                    residual.Pointer,
                    branch.Pointer,
                    gamma.Pointer,
                    beta.Pointer,
                    output.Pointer,
                    means.Pointer,
                    inverses.Pointer,
                    rows,
                    columns,
                    threshold,
                    keepScale,
                    1e-5f,
                    operationSeed);
                rng.EnqueueResidualDropoutLayerNormBackwardFloat32(
                    residual.Pointer,
                    branch.Pointer,
                    gamma.Pointer,
                    means.Pointer,
                    inverses.Pointer,
                    outputGradient.Pointer,
                    residualGradient.Pointer,
                    branchGradient.Pointer,
                    gammaGradient.Pointer,
                    betaGradient.Pointer,
                    parameterScratch.Pointer,
                    rows,
                    columns,
                    sameParent: false,
                    threshold,
                    keepScale,
                    operationSeed);
            });

        bool[] firstMask = ReplayAndValidateMask();
        bool[] secondMask = ReplayAndValidateMask();
        int changed = firstMask.Zip(secondMask)
            .Count(pair => pair.First != pair.Second);
        Assert.InRange(changed, length / 4, length * 3 / 4);

        bool[] ReplayAndValidateMask()
        {
            graph.Launch();
            lane.SynchronizeComputeStream();
            float[] normalized = Read(lane, output, length);
            float[] rowMeans = Read(lane, means, rows);
            float[] rowInverses = Read(lane, inverses, rows);
            float[] residualGradients = Read(
                lane, residualGradient, length);
            float[] branchGradients = Read(lane, branchGradient, length);
            var mask = new bool[length];
            int checkedGradients = 0;
            for (int index = 0; index < length; index++)
            {
                int row = index / columns;
                float fusedInput = normalized[index] / rowInverses[row]
                    + rowMeans[row];
                bool kept = MathF.Abs(fusedInput - (0.25f + keepScale))
                    < MathF.Abs(fusedInput - 0.25f);
                mask[index] = kept;
                float residualValue = residualGradients[index];
                if (MathF.Abs(residualValue) < 1e-5f)
                    continue;
                checkedGradients++;
                float expectedBranch = kept
                    ? residualValue * keepScale
                    : 0f;
                Assert.InRange(
                    MathF.Abs(branchGradients[index] - expectedBranch),
                    0f,
                    3e-5f);
            }
            Assert.InRange(checkedGradients, length / 2, length);
            return mask;
        }
    }

    [Theory]
    [InlineData(TensorPrecisionMode.Float32)]
    [InlineData(TensorPrecisionMode.BFloat16)]
    [InlineData(TensorPrecisionMode.Bfp8)]
    [InlineData(TensorPrecisionMode.Mix8_32)]
    public void TensorGraphDropoutIsFiniteAndBackwardRetainsForwardToken(
        TensorPrecisionMode mode)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        const int length = 1024;
        const float probability = 0.25f;
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        Tensor? input = null;
        Tensor? output = null;
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndices = [0];
            using IDisposable precision = TensorExecutionContext
                .PushPrecisionPolicy(Policy(mode));
            using CudaExecutionLane lane =
                CudaExecutionLaneFactory.Create(0);
            CudaGraphRngState rng = CudaGraphRngState.Create(
                lane,
                initialCounter: 9000);
            input = CreateTensor(mode, Enumerable.Repeat(1f, length).ToArray());
            input.to(new TorchDevice(TensorDevice.Cuda, 0));
            NativeCudaTransferTelemetry before =
                NativeCudaRuntime.TransferTelemetry;

            using (CudaGraphDropoutCaptureScope.Begin(
                       rng,
                       baseSeed: 0x4e4e_5452_4752_4150UL))
            {
                output = input.Dropout(
                    probability,
                    new Random(1701));
            }
            lane.SynchronizeComputeStream();
            NativeCudaTransferTelemetry forwardTransfers =
                NativeCudaRuntime.TransferTelemetry - before;
            Assert.Equal(0, forwardTransfers.HostToDeviceBytes);
            Assert.Equal(0, forwardTransfers.DeviceToHostBytes);

            float[] forward = output.Data.ToArray();
            output.BackwardAndRelease(Enumerable.Repeat(1f, length).ToArray());
            lane.SynchronizeComputeStream();
            float[] backward = input.Grad.ToArray();

            Assert.All(forward, static value => Assert.True(float.IsFinite(value)));
            Assert.All(backward, static value => Assert.True(float.IsFinite(value)));
            for (int index = 0; index < length; index++)
            {
                Assert.Equal(forward[index] == 0f, backward[index] == 0f);
            }
        }
        finally
        {
            output?.InvalidateCudaBuffers();
            input?.InvalidateCudaBuffers();
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }

    [Theory]
    [InlineData(TensorPrecisionMode.Float32)]
    [InlineData(TensorPrecisionMode.BFloat16)]
    [InlineData(TensorPrecisionMode.Bfp8)]
    [InlineData(TensorPrecisionMode.Mix8_32)]
    public void CapturedFusedResidualDropoutLayerNormStaysOnCudaAndIsFinite(
        TensorPrecisionMode mode)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        const int rows = 4;
        const int columns = 32;
        const int length = rows * columns;
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        var tensors = new List<Tensor>();
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndices = [0];
            using IDisposable precision = TensorExecutionContext
                .PushPrecisionPolicy(Policy(mode));
            using CudaExecutionLane lane =
                CudaExecutionLaneFactory.Create(0);
            CudaGraphRngState rng = CudaGraphRngState.Create(lane, 112);
            int[] activationShape = [rows, columns];
            Tensor residual = CreateTensor(
                mode,
                Enumerable.Range(0, length)
                    .Select(index => MathF.Sin(index * 0.13f) * 0.25f)
                    .ToArray(),
                activationShape);
            Tensor branch = CreateTensor(
                mode,
                Enumerable.Range(0, length)
                    .Select(index => MathF.Cos(index * 0.07f) * 0.5f)
                    .ToArray(),
                activationShape);
            Tensor gamma = CreateTensor(
                mode,
                Enumerable.Repeat(1f, columns).ToArray(),
                [columns]);
            Tensor beta = CreateTensor(
                mode,
                Enumerable.Repeat(0f, columns).ToArray(),
                [columns]);
            tensors.AddRange([residual, branch, gamma, beta]);
            foreach (Tensor tensor in tensors)
                tensor.to(new TorchDevice(TensorDevice.Cuda, 0));
            NativeCudaTransferTelemetry before =
                NativeCudaRuntime.TransferTelemetry;

            Tensor output;
            IReadOnlyList<CudaOperationProfileSample> operations;
            NativeCudaTransferTelemetry transfers;
            using (CudaOperationProfiler.Begin())
            {
                using (CudaGraphDropoutCaptureScope.Begin(
                           rng,
                           baseSeed: 0x5253_444f_5554_4c4eUL))
                {
                    output = residual.AddDropoutLayerNormLastDim(
                        branch,
                        gamma,
                        beta,
                        probability: 0.2f,
                        random: new Random(73));
                }
                lane.SynchronizeComputeStream();
                transfers = NativeCudaRuntime.TransferTelemetry - before;
                output.BackwardAndRelease(
                    Enumerable.Repeat(1f, length).ToArray());
                lane.SynchronizeComputeStream();
                operations = CudaOperationProfiler.Snapshot();
            }
            tensors.Add(output);
            lane.SynchronizeComputeStream();
            Assert.Equal(0, transfers.HostToDeviceBytes);
            Assert.Equal(0, transfers.DeviceToHostBytes);
            Assert.All(output.Data, static value => Assert.True(float.IsFinite(value)));

            Assert.Contains(
                operations,
                static sample => sample.Operation
                    == "forward.residual_dropout_layer_norm");
            Assert.Contains(
                operations,
                static sample => sample.Operation
                    == "backward.residual_dropout_layer_norm");
            Assert.DoesNotContain(
                operations,
                static sample => sample.Operation
                    is "forward.residual_dropout" or "forward.layer_norm");
            Assert.All(residual.Grad, static value => Assert.True(float.IsFinite(value)));
            Assert.All(branch.Grad, static value => Assert.True(float.IsFinite(value)));
            Assert.All(gamma.Grad, static value => Assert.True(float.IsFinite(value)));
            Assert.All(beta.Grad, static value => Assert.True(float.IsFinite(value)));
        }
        finally
        {
            foreach (Tensor tensor in tensors.Distinct())
                tensor.InvalidateCudaBuffers();
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }

    [Fact]
    public void ConsecutiveReplaysAdvanceDeviceDropoutCounterAndLaneOwnsGraph()
    {
        if (CudaNativeGateway.DeviceCount(out int deviceCount) != 0
            || deviceCount == 0)
        {
            return;
        }

        const int length = 4096;
        CudaExecutionLane lane = CudaExecutionLaneFactory.Create(0);
        CudaMemoryLease output = ExecutionLaneResources.Attach(
            lane,
            lane.Memory.Allocate(
                checked((nuint)length * sizeof(float)),
                CudaMemoryKind.Persistent));
        CudaGraphRngState rng = CudaGraphRngState.Create(lane);
        CudaGraphExecutable graph = CudaGraphExecutable.Capture(
            lane,
            () => rng.EnqueueDropoutMask(
                output.Pointer,
                length,
                dropoutProbability: 0.5f,
                seed: 0x4e4e5452u));

        float[] first = ReplayAndRead(lane, graph, output, length);
        float[] second = ReplayAndRead(lane, graph, output, length);

        Assert.All(first, value => Assert.True(value is 0f or 2f));
        Assert.All(second, value => Assert.True(value is 0f or 2f));
        int changed = first.Zip(second).Count(pair => pair.First != pair.Second);
        Assert.InRange(changed, length / 4, length * 3 / 4);
        Assert.False(graph.IsClosed);

        lane.Dispose();
        Assert.True(graph.IsClosed);
        Assert.Null(graph.ReleaseFailure);
    }

    [Fact]
    public void FailedRecordingEndsCaptureAndLeavesLaneReusable()
    {
        if (CudaNativeGateway.DeviceCount(out int deviceCount) != 0
            || deviceCount == 0)
        {
            return;
        }

        const int length = 64;
        using CudaExecutionLane lane = CudaExecutionLaneFactory.Create(0);
        CudaMemoryLease output = ExecutionLaneResources.Attach(
            lane,
            lane.Memory.Allocate(
                checked((nuint)length * sizeof(float)),
                CudaMemoryKind.Persistent));
        CudaGraphRngState rng = CudaGraphRngState.Create(lane);

        InvalidOperationException failure = Assert.Throws<InvalidOperationException>(
            () => CudaGraphExecutable.Capture(
                lane,
                () => throw new InvalidOperationException("recording failed")));
        Assert.Equal("recording failed", failure.Message);

        CudaGraphExecutable graph = CudaGraphExecutable.Capture(
            lane,
            () => rng.EnqueueDropoutMask(
                output.Pointer,
                length,
                dropoutProbability: 0.25f,
                seed: 17));
        float[] values = ReplayAndRead(lane, graph, output, length);
        Assert.Contains(values, static value => value == 0f);
        Assert.Contains(values, static value => value > 1f);
    }

    private static float[] ReplayAndRead(
        CudaExecutionLane lane,
        CudaGraphExecutable graph,
        CudaMemoryLease output,
        int length)
    {
        graph.Launch();
        lane.SynchronizeComputeStream();
        return Read(lane, output, length);
    }

    private static float[] Read(
        CudaExecutionLane lane,
        CudaMemoryLease output,
        int length)
    {
        nint host = Marshal.AllocHGlobal(checked(length * sizeof(float)));
        try
        {
            int status = CudaNativeGateway.CopyDeviceToHost(
                lane.DeviceIndex,
                host,
                output.Pointer,
                checked((nuint)length * sizeof(float)));
            Assert.Equal(0, status);
            var values = new float[length];
            Marshal.Copy(host, values, 0, length);
            return values;
        }
        finally
        {
            Marshal.FreeHGlobal(host);
        }
    }

    private static void Zero(
        CudaExecutionLane lane,
        CudaMemoryLease destination,
        int length)
    {
        Assert.Equal(
            0,
            CudaNativeGateway.MemsetAsync(
                lane.DeviceIndex,
                destination.Pointer,
                0,
                checked((nuint)length * sizeof(float)),
                lane.ComputeStreamHandle));
    }

    private static CudaMemoryLease PersistentFloatBuffer(
        CudaExecutionLane lane,
        int length)
        => ExecutionLaneResources.Attach(
            lane,
            lane.Memory.Allocate(
                checked((nuint)length * sizeof(float)),
                CudaMemoryKind.Persistent));

    private static void Upload(
        CudaExecutionLane lane,
        CudaMemoryLease destination,
        float[] values)
    {
        int byteLength = checked(values.Length * sizeof(float));
        nint host = Marshal.AllocHGlobal(byteLength);
        try
        {
            Marshal.Copy(values, 0, host, values.Length);
            Assert.Equal(
                0,
                CudaNativeGateway.CopyHostToDevice(
                    lane.DeviceIndex,
                    destination.Pointer,
                    host,
                    checked((nuint)byteLength)));
        }
        finally
        {
            Marshal.FreeHGlobal(host);
        }
    }

    private static PrecisionPolicy Policy(TensorPrecisionMode mode)
        => mode switch
        {
            TensorPrecisionMode.Float32 => PrecisionPolicy.Float32,
            TensorPrecisionMode.BFloat16 => PrecisionPolicy.BFloat16,
            TensorPrecisionMode.Bfp8 => PrecisionPolicy.Bfp8,
            TensorPrecisionMode.Mix8_32 => PrecisionPolicy.Mix8_32,
            _ => throw new ArgumentOutOfRangeException(nameof(mode)),
        };

    private static Tensor CreateTensor(
        TensorPrecisionMode mode,
        float[] values,
        int[]? shape = null)
    {
        int[] tensorShape = shape ?? [values.Length];
        return mode switch
        {
            TensorPrecisionMode.Float32 => new Tensor(
                values, tensorShape, dtype: TensorDType.Float32),
            TensorPrecisionMode.BFloat16 => new Tensor(
                values, tensorShape, dtype: TensorDType.BFloat16),
            TensorPrecisionMode.Bfp8 => Tensor.FromBfp8(
                values,
                tensorShape,
                Bfp8QuantizationDescriptor.TensorWide),
            TensorPrecisionMode.Mix8_32 => Tensor.FromBfp8(
                values,
                tensorShape,
                Bfp8QuantizationDescriptor.Mix8_32),
            _ => throw new ArgumentOutOfRangeException(nameof(mode)),
        };
    }
}
