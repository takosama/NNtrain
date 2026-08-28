using NNtrain;
using Xunit;

public sealed class CudaGradientCoherenceTests
{
    [Fact]
    public void TwoGpuOptimizerRejectsUnreducedLocalGradientWithoutAdvancing()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        WithTwoCudaDevices(() =>
        {
            Parameter parameter = CreateParameter();
            var optimizer = CreateOptimizer(parameter);
            try
            {
                parameter.T.SetCudaGradient(Values(1f), 0);
                parameter.T.SetCudaGradient(Values(2f), 1);

                InvalidOperationException failure =
                    Assert.Throws<InvalidOperationException>(optimizer.step);

                Assert.Contains("has not been reduced", failure.Message);
                Assert.Equal(0, optimizer.CaptureState().Step);
            }
            finally
            {
                optimizer.DisposeCudaResources();
                parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void PartialReductionFailureCannotBeConsumedByOptimizer()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        WithTwoCudaDevices(() =>
        {
            Parameter parameter = CreateParameter();
            var optimizer = CreateOptimizer(parameter);
            using var reducer = new CudaBFloat16GradientAllReducePlan(
                [parameter], [0, 1]);
            try
            {
                long stepId = reducer.BeginStep();
                reducer.BeginDeviceStep(stepId, 0);
                reducer.BeginDeviceStep(stepId, 1);
                parameter.T.SetCudaGradient(Values(1f), 0);
                reducer.NotifyGradientReady(parameter.T, 0, stepId);

                InvalidOperationException reductionFailure =
                    Assert.Throws<InvalidOperationException>(
                        () => reducer.Complete(stepId));
                Assert.Contains("not completed", reductionFailure.Message);

                InvalidOperationException optimizerFailure =
                    Assert.Throws<InvalidOperationException>(optimizer.step);
                Assert.Contains("has not been reduced", optimizerFailure.Message);
                Assert.Equal(0, optimizer.CaptureState().Step);
            }
            finally
            {
                reducer.Dispose();
                optimizer.DisposeCudaResources();
                parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void DuplicateBf16NotificationIsRejectedAndAbortReopensPlan()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        WithTwoCudaDevices(() =>
        {
            Parameter parameter = CreateParameter();
            using var reducer = new CudaBFloat16GradientAllReducePlan(
                [parameter], [0, 1]);
            try
            {
                long failedStep = reducer.BeginStep();
                reducer.BeginDeviceStep(failedStep, 0);
                reducer.BeginDeviceStep(failedStep, 1);
                parameter.T.SetCudaGradient(Values(1f), 0);
                reducer.NotifyGradientReady(parameter.T, 0, failedStep);

                InvalidOperationException duplicate =
                    Assert.Throws<InvalidOperationException>(() =>
                        reducer.NotifyGradientReady(
                            parameter.T, 0, failedStep));
                Assert.Contains("notified twice", duplicate.Message);
                reducer.Abort(failedStep);

                Reduce(reducer, parameter, Values(2f), Values(3f));
                CudaGradientCoherenceSnapshot snapshot =
                    parameter.T.GetCudaGradientCoherenceSnapshot();
                Assert.Equal(
                    CudaGradientCoherenceKind.Reduced,
                    snapshot.Kind);
            }
            finally
            {
                reducer.Dispose();
                parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void RebuiltReducerInvalidatesOldUnconsumedStamp()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        WithTwoCudaDevices(() =>
        {
            Parameter parameter = CreateParameter();
            var optimizer = CreateOptimizer(parameter);
            try
            {
                using (var first = new CudaBFloat16GradientAllReducePlan(
                    [parameter], [0, 1]))
                {
                    Reduce(first, parameter, Values(1f), Values(2f));
                }

                using var rebuilt = new CudaBFloat16GradientAllReducePlan(
                    [parameter], [0, 1]);
                InvalidOperationException failure =
                    Assert.Throws<InvalidOperationException>(optimizer.step);

                Assert.Contains("has not been reduced", failure.Message);
                Assert.Equal(0, optimizer.CaptureState().Step);
            }
            finally
            {
                optimizer.DisposeCudaResources();
                parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void OptimizerRejectsParametersFromDifferentCompletedSteps()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        WithTwoCudaDevices(() =>
        {
            Parameter first = CreateParameter(name: "first");
            Parameter second = CreateParameter(name: "second");
            var optimizer = new AdamW(
                [first, second],
                new AdamWOptions
                {
                    LearningRate = 1e-3f,
                    WeightDecay = 0f,
                });
            try
            {
                using (var firstReducer =
                    new CudaBFloat16GradientAllReducePlan(
                        [first], [0, 1]))
                {
                    Reduce(firstReducer, first, Values(1f), Values(2f));
                }
                using (var secondReducer =
                    new CudaBFloat16GradientAllReducePlan(
                        [second], [0, 1]))
                {
                    Reduce(secondReducer, second, Values(3f), Values(4f));
                }

                InvalidOperationException failure =
                    Assert.Throws<InvalidOperationException>(optimizer.step);
                Assert.Contains("do not share", failure.Message);
                Assert.Equal(0, optimizer.CaptureState().Step);
            }
            finally
            {
                optimizer.DisposeCudaResources();
                first.T.InvalidateCudaBuffers();
                second.T.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void CompletedTwoGpuReductionIsExactAndConsumedOnce()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        WithTwoCudaDevices(() =>
        {
            Parameter parameter = CreateParameter();
            var optimizer = CreateOptimizer(parameter);
            using var reducer = new CudaBFloat16GradientAllReducePlan(
                [parameter], [0, 1]);
            try
            {
                float[] first = Values(1f);
                float[] second = Values(2f);
                Reduce(reducer, parameter, first, second);

                Assert.Equal(
                    first.Zip(second, static (left, right) => left + right),
                    parameter.T.Grad);
                optimizer.step();
                Assert.Equal(1, optimizer.CaptureState().Step);

                float[] primary = Read(
                    parameter.T.EnsureCudaBFloat16Buffer(0));
                float[] secondary = Read(
                    parameter.T.EnsureCudaBFloat16Buffer(1));
                Assert.Equal(primary, secondary);

                InvalidOperationException duplicate =
                    Assert.Throws<InvalidOperationException>(optimizer.step);
                Assert.Contains("already consumed", duplicate.Message);
                Assert.Equal(1, optimizer.CaptureState().Step);
            }
            finally
            {
                reducer.Dispose();
                optimizer.DisposeCudaResources();
                parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void OneGpuLocalGradientIsAcceptedButStillConsumedOnce()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousDevices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = [0];
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Parameter parameter = CreateParameter(includeSecondReplica: false);
            var optimizer = CreateOptimizer(parameter);
            try
            {
                parameter.T.SetCudaGradient(Values(1f), 0);
                optimizer.step();
                Assert.Equal(1, optimizer.CaptureState().Step);

                Assert.Throws<InvalidOperationException>(optimizer.step);
                Assert.Equal(1, optimizer.CaptureState().Step);
            }
            finally
            {
                optimizer.DisposeCudaResources();
                parameter.T.InvalidateCudaBuffers();
            }
        }
        finally
        {
            Tensor.CudaDeviceIndices = previousDevices;
            Tensor.ExecutionDevice = previousDevice;
        }
    }

    private static void Reduce(
        CudaBFloat16GradientAllReducePlan reducer,
        Parameter parameter,
        float[] first,
        float[] second)
    {
        long stepId = reducer.BeginStep();
        reducer.BeginDeviceStep(stepId, 0);
        reducer.BeginDeviceStep(stepId, 1);
        parameter.T.SetCudaGradient(first, 0);
        reducer.NotifyGradientReady(parameter.T, 0, stepId);
        parameter.T.SetCudaGradient(second, 1);
        reducer.NotifyGradientReady(parameter.T, 1, stepId);
        reducer.Complete(stepId);
    }

    private static Parameter CreateParameter(
        bool includeSecondReplica = true,
        string name = "weight")
    {
        var parameter = new Parameter(
            Enumerable.Repeat(0.5f, 16).ToArray(),
            [2, 8],
            name,
            WeightDecayPolicy.Apply,
            TensorDType.BFloat16);
        parameter.T.to(new TorchDevice(TensorDevice.Cuda, 0));
        if (includeSecondReplica)
            parameter.T.EnsureCudaBFloat16Buffer(1);
        return parameter;
    }

    private static AdamW CreateOptimizer(Parameter parameter)
        => new(
            [parameter],
            new AdamWOptions
            {
                LearningRate = 1e-3f,
                WeightDecay = 0f,
            });

    private static float[] Values(float multiplier)
        => Enumerable.Range(1, 16)
            .Select(value => value * multiplier)
            .ToArray();

    private static float[] Read(NativeCudaBuffer<ushort> buffer)
    {
        var encoded = new ushort[buffer.Length];
        var decoded = new float[buffer.Length];
        buffer.CopyToCPU(encoded);
        TensorStorageCodec.DecodeBFloat16(encoded, decoded);
        return decoded;
    }

    private static void WithTwoCudaDevices(Action action)
    {
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousDevices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = [0, 1];
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            action();
        }
        finally
        {
            Tensor.CudaDeviceIndices = previousDevices;
            Tensor.ExecutionDevice = previousDevice;
        }
    }
}
