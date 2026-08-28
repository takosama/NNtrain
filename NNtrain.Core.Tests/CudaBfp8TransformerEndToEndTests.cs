using NNtrain;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class CudaBfp8TransformerEndToEndTests
{
    [Theory]
    [InlineData(TensorPrecisionMode.Bfp8)]
    [InlineData(TensorPrecisionMode.Mix8_32)]
    public void CudaThenPrecisionConversionPreservesAllConfiguredReplicas(
        TensorPrecisionMode mode)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousDevices = Tensor.CudaDeviceIndices.ToArray();
        Parameter[] parameters = [];
        try
        {
            int[] devices = Enumerable.Range(
                    0,
                    Math.Min(2, Tensor.CudaDeviceCount))
                .ToArray();
            Tensor.CudaDeviceIndices = devices;
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            var model = new GptRinWikiJp(
                vocabularySize: 32,
                contextLength: 4,
                dModel: 16,
                numHeads: 2,
                dHidden: 32,
                numLayers: 1,
                rng: new Random(47),
                dropout: 0f);
            parameters = model.parameters().ToArray();

            model.to(TensorDevice.Cuda);
            model.to(mode, bfp8_block_size: 32);
            Bfp8QuantizationDescriptor expectedDescriptor =
                mode == TensorPrecisionMode.Bfp8
                    ? Bfp8QuantizationDescriptor.TensorWide
                    : Bfp8QuantizationDescriptor.Block(32);

            Assert.All(
                parameters,
                parameter =>
                {
                    Assert.Equal(TensorDevice.Cuda, parameter.T.Device);
                    Assert.Equal(devices[0], parameter.T.device.Index);
                    Assert.Equal(TensorDType.Bfp8, parameter.T.DType);
                    Assert.Equal(
                        expectedDescriptor,
                        parameter.T.Bfp8Quantization);
                    Assert.Equal(
                        devices,
                        parameter.T.GetResidentCudaDeviceIndices());
                    foreach (int deviceIndex in devices)
                    {
                        Assert.Equal(
                            expectedDescriptor,
                            parameter.T
                                .EnsureCudaBfp8Buffer(deviceIndex)
                                .Descriptor);
                    }
                });
        }
        finally
        {
            foreach (Parameter parameter in parameters)
                parameter.T.InvalidateCudaBuffers();
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousDevices;
        }
    }

    [Fact]
    public void PureBfp8NekoMuonAndAdamWConvergeForTwelveCudaSteps()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousDevices = Tensor.CudaDeviceIndices.ToArray();
        Parameter[] parameters = [];
        NekoMuon? neko = null;
        AdamW? adam = null;
        try
        {
            Tensor.CudaDeviceIndices = [0];
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            using IDisposable policy =
                TensorExecutionContext.PushPrecisionPolicy(
                    PrecisionPolicy.Bfp8);
            var model = new GptRinWikiJp(
                vocabularySize: 256,
                contextLength: 16,
                dModel: 32,
                numHeads: 4,
                dHidden: 64,
                numLayers: 1,
                rng: new Random(991),
                initializationScale: 0.02f,
                dropout: 0f);
            model.to(TensorPrecisionMode.Bfp8);
            model.to(TensorDevice.Cuda);
            parameters = model.parameters().ToArray();
            neko = new NekoMuon(
                model.HiddenWeightParameters,
                new NekoMuonOptions
                {
                    LearningRate = 3e-4f,
                    WeightDecay = 0.01f,
                    MaxNewtonSchulzSteps = 5,
                    NewtonSchulzInterval = 1,
                    NewtonSchulzDepthMode =
                        NekoMuonNewtonSchulzDepthMode.Fixed,
                    NewtonSchulzDepth = 5f,
                });
            adam = new AdamW(
                model.AuxiliaryParameters,
                new AdamWOptions
                {
                    LearningRate = 3e-4f,
                    Beta1 = 0.9f,
                    Beta2 = 0.95f,
                    Epsilon = 1e-8f,
                    WeightDecay = 0.01f,
                });
            var optimizer = new CompositeOptimizer(neko, adam);
            int[] input = Enumerable.Range(0, 4 * 16)
                .Select(index => (index * 37 + 11) % 256)
                .ToArray();
            int[] target = Enumerable.Range(0, 4 * 16)
                .Select(index => (index * 53 + 7) % 256)
                .ToArray();
            var losses = new float[12];
            var norms = new float[12];
            using var dataParallel = new CudaDataParallelEngine(model, [0]);
            dataParallel.PrepareForTraining(batchSize: 4);
            optimizer.prepare();

            for (int step = 0; step < losses.Length; step++)
            {
                optimizer.zero_grad();
                losses[step] = dataParallel.ForwardBackward(
                    input,
                    target,
                    batchSize: 4,
                    sequenceLength: 16);
                norms[step] = nn.utils.clip_grad_norm_(
                    parameters,
                    max_norm: 1f);
                optimizer.step();
            }

            Assert.All(losses, value => Assert.True(float.IsFinite(value)));
            Assert.All(norms, value => Assert.True(float.IsFinite(value)));
            Assert.True(losses[^1] < losses[0] - 0.1f,
                $"Expected pure BFP8 loss to fall, got {losses[0]} -> " +
                $"{losses[^1]}.");
            NekoMuonDiagnostics diagnostics = neko.GetDiagnostics();
            Assert.InRange(diagnostics.MeanConfidence, 0f, 1f);
            Assert.All(parameters, parameter =>
            {
                Assert.Equal(TensorDType.Bfp8, parameter.T.DType);
                Assert.False(parameter.T.HasCudaMasterFloat32Buffer(0));
                Assert.Equal(
                    Bfp8QuantizationDescriptor.TensorWide,
                    parameter.T.EnsureCudaBfp8Buffer(0).Descriptor);
            });
        }
        finally
        {
            adam?.DisposeCudaResources();
            neko?.DisposeCudaResources();
            foreach (Parameter parameter in parameters)
                parameter.T.InvalidateCudaBuffers();
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousDevices;
        }
    }

    [Theory]
    [InlineData(TensorPrecisionMode.Bfp8)]
    [InlineData(TensorPrecisionMode.Mix8_32)]
    public void OneGpuTransformerForwardBackwardAndAdamWStayResident(
        TensorPrecisionMode mode)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousDevices = Tensor.CudaDeviceIndices.ToArray();
        Parameter[] parameters = [];
        try
        {
            Tensor.CudaDeviceIndices = [0];
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            using IDisposable policy =
                TensorExecutionContext.PushPrecisionPolicy(
                    mode == TensorPrecisionMode.Bfp8
                        ? PrecisionPolicy.Bfp8
                        : PrecisionPolicy.Mix8_32);
            var model = new GptRinWikiJp(
                vocabularySize: 32,
                contextLength: 4,
                dModel: 16,
                numHeads: 2,
                dHidden: 32,
                numLayers: 1,
                rng: new Random(53),
                dropout: 0.125f);
            model.to(mode, bfp8_block_size: 32);
            model.to(TensorDevice.Cuda);
            parameters = model.parameters().ToArray();
            var optimizer = new AdamW(
                parameters,
                new AdamWOptions
                {
                    LearningRate = 1e-3f,
                    WeightDecay = 0f,
                });

            optimizer.zero_grad();
            Tensor logits = model.forward(
                [1, 2, 3, 4, 5, 6, 7, 8],
                batch_size: 2,
                sequence_length: 4);
            Assert.Equal(TensorDType.Bfp8, logits.DType);
            Assert.Equal(
                mode == TensorPrecisionMode.Bfp8
                    ? Bfp8QuantizationDescriptor.TensorWide
                    : Bfp8QuantizationDescriptor.Block(32),
                logits.Bfp8Quantization);

            Tensor loss = logits.CrossEntropyWithLogits(
                [2, 3, 4, 5, 6, 7, 8, 9]);
            Assert.True(float.IsFinite(loss.item()));
            loss.BackwardAndRelease();
            Assert.All(
                parameters,
                parameter =>
                {
                    Assert.True(parameter.T.HasGradientBuffer);
                    if (mode == TensorPrecisionMode.Bfp8)
                        Assert.True(parameter.T.HasAuthoritativeCudaBfp8Gradient);
                });

            optimizer.step();
            Assert.All(
                parameters,
                parameter =>
                {
                    Assert.Equal(TensorDevice.Cuda, parameter.T.Device);
                    Assert.Equal(TensorDType.Bfp8, parameter.T.DType);
                    Assert.Equal(
                        logits.Bfp8Quantization,
                        parameter.T.Bfp8Quantization);
                });
        }
        finally
        {
            foreach (Parameter parameter in parameters)
                parameter.T.InvalidateCudaBuffers();
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousDevices;
        }
    }

    [Theory]
    [InlineData(TensorPrecisionMode.Bfp8)]
    [InlineData(TensorPrecisionMode.Mix8_32)]
    public void TrainingLossHeadPublishesBFloat16DirectlyToCrossEntropy(
        TensorPrecisionMode mode)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousDevices = Tensor.CudaDeviceIndices.ToArray();
        Parameter[] parameters = [];
        try
        {
            Tensor.CudaDeviceIndices = [0];
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            using IDisposable policy =
                TensorExecutionContext.PushPrecisionPolicy(
                    mode == TensorPrecisionMode.Bfp8
                        ? PrecisionPolicy.Bfp8
                        : PrecisionPolicy.Mix8_32);
            var model = new GptRinWikiJp(
                vocabularySize: 32,
                contextLength: 4,
                dModel: 16,
                numHeads: 2,
                dHidden: 32,
                numLayers: 1,
                rng: new Random(67),
                dropout: 0f);
            model.to(mode, bfp8_block_size: 32);
            model.to(TensorDevice.Cuda);
            parameters = model.parameters().ToArray();
            CudaBfp8GemmTelemetrySnapshot before =
                CudaBfp8GemmTelemetry.Snapshot;

            Tensor loss = model.forward_loss(
                [1, 2, 3, 4, 5, 6, 7, 8],
                [2, 3, 4, 5, 6, 7, 8, 9],
                batch_size: 2,
                sequence_length: 4);
            Assert.True(float.IsFinite(loss.item()));
            loss.BackwardAndRelease();

            CudaBfp8GemmTelemetrySnapshot delta =
                CudaBfp8GemmTelemetry.Snapshot - before;
            Assert.Equal(1, delta.DirectBFloat16LossHeadExecutions);
            Assert.All(parameters, parameter =>
                Assert.True(parameter.T.HasGradientBuffer));
        }
        finally
        {
            foreach (Parameter parameter in parameters)
                parameter.T.InvalidateCudaBuffers();
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousDevices;
        }
    }

    [Fact]
    public void DataParallelGraphAndDetailedProfileUseProductionLossHead()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousDevices = Tensor.CudaDeviceIndices.ToArray();
        Parameter[] parameters = [];
        try
        {
            Tensor.CudaDeviceIndices = [0];
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            using IDisposable policy =
                TensorExecutionContext.PushPrecisionPolicy(
                    PrecisionPolicy.Mix8_32);
            var model = new GptRinWikiJp(
                vocabularySize: 32,
                contextLength: 4,
                dModel: 16,
                numHeads: 2,
                dHidden: 32,
                numLayers: 1,
                rng: new Random(71),
                dropout: 0f);
            model.to(TensorPrecisionMode.Mix8_32, bfp8_block_size: 32);
            model.to(TensorDevice.Cuda);
            parameters = model.parameters().ToArray();
            int[] input = [1, 2, 3, 4, 5, 6, 7, 8];
            int[] target = [2, 3, 4, 5, 6, 7, 8, 9];
            using var engine = new CudaDataParallelEngine(model, [0]);
            engine.PrepareForTraining(batchSize: 2);

            CudaBfp8GemmTelemetrySnapshot graphBefore =
                CudaBfp8GemmTelemetry.Snapshot;
            _ = engine.ForwardBackward(
                input,
                target,
                batchSize: 2,
                sequenceLength: 4);
            CudaBfp8GemmTelemetrySnapshot graphDelta =
                CudaBfp8GemmTelemetry.Snapshot - graphBefore;
            Assert.True(graphDelta.DirectBFloat16LossHeadExecutions >= 1);

            foreach (Parameter parameter in parameters)
                parameter.T.ZeroGrad();
            CudaBfp8GemmTelemetrySnapshot profileBefore =
                CudaBfp8GemmTelemetry.Snapshot;
            CudaDataParallelProfile profile = engine.ForwardBackwardProfiled(
                input,
                target,
                batchSize: 2,
                sequenceLength: 4);
            CudaBfp8GemmTelemetrySnapshot profileDelta =
                CudaBfp8GemmTelemetry.Snapshot - profileBefore;

            Assert.Equal(1, profileDelta.DirectBFloat16LossHeadExecutions);
            Assert.True(float.IsFinite(profile.Loss));
        }
        finally
        {
            foreach (Parameter parameter in parameters)
                parameter.T.InvalidateCudaBuffers();
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousDevices;
        }
    }

    [Theory]
    [InlineData(TensorPrecisionMode.Bfp8)]
    [InlineData(TensorPrecisionMode.Mix8_32)]
    public void CudaShapeGluePreservesPayloadDescriptorAndBackward(
        TensorPrecisionMode mode)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousDevices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = [0];
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Bfp8QuantizationDescriptor descriptor =
                mode == TensorPrecisionMode.Bfp8
                    ? Bfp8QuantizationDescriptor.TensorWide
                    : Bfp8QuantizationDescriptor.Block(32);
            using IDisposable policy =
                TensorExecutionContext.PushPrecisionPolicy(
                    mode == TensorPrecisionMode.Bfp8
                        ? PrecisionPolicy.Bfp8
                        : PrecisionPolicy.Mix8_32);
            Tensor source = Tensor.FromBfp8(
                Enumerable.Range(0, 128)
                    .Select(index => MathF.Sin(index * 0.17f))
                    .ToArray(),
                [2, 4, 16],
                descriptor);
            source.to(new TorchDevice(TensorDevice.Cuda, 0));
            _ = source.EnsureCudaBfp8Buffer(0);
            NativeCudaTransferTelemetry before =
                NativeCudaRuntime.TransferTelemetry;

            Tensor reshaped = source.Reshape(8, 16);
            Tensor selected = source.Reshape(8, 16)
                .SelectLastSequenceToken();
            ForgetMemoryV2Cuda.GetAccelerator(0).Synchronize();

            NativeCudaTransferTelemetry transfers =
                NativeCudaRuntime.TransferTelemetry - before;
            Assert.Equal(0, transfers.HostToDeviceBytes);
            Assert.Equal(0, transfers.DeviceToHostBytes);
            Assert.Equal(descriptor, reshaped.Bfp8Quantization);
            Assert.Equal(descriptor, selected.Bfp8Quantization);
            selected.BackwardAndRelease(Enumerable.Repeat(1f, 16).ToArray());
            Assert.True(source.HasGradientBuffer);

            reshaped.InvalidateCudaBuffers();
            selected.InvalidateCudaBuffers();
            source.InvalidateCudaBuffers();
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousDevices;
        }
    }
}
