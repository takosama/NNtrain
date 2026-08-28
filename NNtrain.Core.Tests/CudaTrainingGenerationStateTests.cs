using NNtrain;
using Xunit;

public sealed class CudaTrainingGenerationStateTests
{
    [Fact]
    public void InTrainingGenerationKeepsParametersResidentAndRestoresTraining()
    {
        if (Tensor.CudaDeviceCount == 0)
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndices = [0];
            var model = new GptRinWikiJp(
                vocabularySize: 32,
                contextLength: 4,
                dModel: 8,
                numHeads: 2,
                dHidden: 16,
                numLayers: 1,
                rng: new Random(73),
                dropout: 0.1f,
                dtype: TensorDType.BFloat16);
            model.SetPrecisionMode(TensorPrecisionMode.Mix16_32);
            model.to(TensorDevice.Cuda);
            model.train();

            Parameter[] parameters = model.Parameters().ToArray();
            nint[] residentPointers = parameters
                .Select(parameter => parameter.T
                    .EnsureCudaBFloat16Buffer(0).NativePtr)
                .ToArray();
            nint[] residentMasterPointers = parameters
                .Select(parameter => parameter.T
                    .EnsureCudaMasterFloat32Buffer(0).NativePtr)
                .ToArray();

            int[] generated = model.GenerateTokenIds(
                [1, 2],
                maxNewTokens: 2,
                temperature: 0f,
                topK: 1,
                stopTokenId: null,
                random: new Random(79));

            Assert.Equal(4, generated.Length);
            Assert.True(model.IsTraining);
            Assert.Equal(
                residentPointers,
                parameters.Select(parameter => parameter.T
                    .EnsureCudaBFloat16Buffer(0).NativePtr).ToArray());
            Assert.Equal(
                residentMasterPointers,
                parameters.Select(parameter => parameter.T
                    .EnsureCudaMasterFloat32Buffer(0).NativePtr).ToArray());

            // A normal recorded backward must still work immediately after
            // the no-grad/eval generation scope has restored training state.
            model.ZeroGrad();
            Tensor logits = model.Forward([1, 2, 3, 4], 1, 4);
            Tensor loss = logits.CrossEntropyWithLogits([2, 3, 4, 5]);
            float lossValue = loss.item();
            loss.BackwardAndRelease();
            ForgetMemoryV2Cuda.GetAccelerator(0).Synchronize();

            Assert.True(float.IsFinite(lossValue));
            Assert.True(model.IsTraining);
            Assert.Equal(
                residentPointers,
                parameters.Select(parameter => parameter.T
                    .EnsureCudaBFloat16Buffer(0).NativePtr).ToArray());
            Assert.Equal(
                residentMasterPointers,
                parameters.Select(parameter => parameter.T
                    .EnsureCudaMasterFloat32Buffer(0).NativePtr).ToArray());
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }
}
