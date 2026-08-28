using NNtrain;
using Xunit;

public sealed class CudaBfp8GenerationTests
{
    [Theory]
    [InlineData(TensorPrecisionMode.Bfp8)]
    [InlineData(TensorPrecisionMode.Mix8_32)]
    public void GenerationKeepsBfp8ParametersButRunsBf16KvCache(
        TensorPrecisionMode precisionMode)
    {
        if (Tensor.CudaDeviceCount == 0)
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            using IDisposable dispatch = CudaDispatchPolicy.Push(
                CudaDispatchPolicy.Defaults);
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndices = [0];
            var model = new GptRinWikiJp(
                vocabularySize: 128,
                contextLength: 4,
                dModel: 16,
                numHeads: 2,
                dHidden: 32,
                numLayers: 1,
                rng: new Random(193),
                dropout: 0.1f);
            model.to(precisionMode, bfp8_block_size: 16);
            model.to(TensorDevice.Cuda);
            Parameter[] parameters = model.Parameters().ToArray();
            CudaBfp8BufferView[] resident = parameters
                .Select(parameter => parameter.T.EnsureCudaBfp8Buffer(0))
                .ToArray();
            CudaBfp8InferenceTelemetrySnapshot before =
                CudaBfp8InferenceTelemetry.Snapshot;

            int[] generated = model.GenerateTokenIds(
                [1, 2],
                maxNewTokens: 2,
                // Unrestricted sampling keeps this test independent of the
                // separately-versioned CUDA top-K native ABI.
                temperature: 0.8f,
                topK: 0,
                stopTokenId: null,
                random: new Random(197));

            CudaBfp8InferenceTelemetrySnapshot inference =
                CudaBfp8InferenceTelemetry.Snapshot - before;
            Assert.Equal([1, 2], generated[..2]);
            Assert.Equal(4, generated.Length);
            Assert.All(generated, token => Assert.InRange(token, 0, 127));
            Assert.True(model.IsTraining);
            Assert.True(inference.EmbeddingWithPositionsExecutions >= 1);
            Assert.True(inference.EmbeddingExecutions >= 2);
            Assert.True(inference.MixedLinearExecutions >= 10);
            Assert.True(inference.MixedLayerNormExecutions >= 2);
            Assert.True(inference.MixedResidualLayerNormExecutions >= 4);
            Assert.Equal(1, inference.KvCachePrefillExecutions);
            Assert.Equal(1, inference.KvCacheIncrementalExecutions);
            Assert.All(parameters, parameter =>
            {
                Assert.Equal(TensorDType.Bfp8, parameter.T.DType);
                Assert.Equal(
                    precisionMode == TensorPrecisionMode.Bfp8
                        ? Bfp8ScaleGranularity.Tensor
                        : Bfp8ScaleGranularity.Block,
                    parameter.T.Bfp8Quantization!.Granularity);
            });
            Assert.Equal(
                resident.Select(static view => view.Payload.NativePtr),
                parameters.Select(parameter => parameter.T
                    .EnsureCudaBfp8Buffer(0).Payload.NativePtr));

            // Leaving no-grad generation must restore the all-BFP8 training
            // graph instead of silently changing its activation contract.
            CudaBfp8InferenceTelemetrySnapshot beforeRecordedForward =
                CudaBfp8InferenceTelemetry.Snapshot;
            Tensor logits = model.Forward([1, 2, 3, 4], 1, 4);
            Assert.Equal(TensorDType.Bfp8, logits.DType);
            CudaBfp8InferenceTelemetrySnapshot recordedForward =
                CudaBfp8InferenceTelemetry.Snapshot - beforeRecordedForward;
            Assert.Equal(0, recordedForward.EmbeddingExecutions);
            Assert.Equal(0, recordedForward.EmbeddingWithPositionsExecutions);
            Assert.Equal(0, recordedForward.MixedLinearExecutions);
            Assert.Equal(0, recordedForward.MixedLayerNormExecutions);
            Assert.Equal(0, recordedForward.MixedResidualLayerNormExecutions);
            Tensor loss = logits.CrossEntropyWithLogits([2, 3, 4, 5]);
            loss.BackwardAndRelease();
            ForgetMemoryV2Cuda.GetAccelerator(0).Synchronize();
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }
}
