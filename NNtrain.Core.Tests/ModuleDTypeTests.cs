using NNtrain;
using Xunit;

public sealed class ModuleDTypeTests
{
    [Fact]
    public void GeneralPurposeModulesRemainFloat32ByDefault()
    {
        var linear = new Linear(4, 3, new Random(11));
        var layerNorm = new LayerNorm(4);
        var feedForward = new FeedForward(4, 8, new Random(13));
        var dropout = new Dropout(0.25f, new Random(17));

        Assert.Equal(TensorDType.Float32, linear.DType);
        Assert.Equal(TensorDType.Float32, layerNorm.DType);
        Assert.Equal(TensorDType.Float32, feedForward.DType);
        Assert.Equal(TensorDType.Float32, dropout.DType);
        Assert.All(
            linear.Parameters()
                .Concat(layerNorm.Parameters())
                .Concat(feedForward.Parameters()),
            parameter =>
                Assert.Equal(TensorDType.Float32, parameter.T.DType));
    }

    [Fact]
    public void GeneralPurposeModulesPropagateExplicitFloat16()
    {
        var linear = new Linear(
            4,
            3,
            new Random(19),
            dtype: TensorDType.Float16);
        var layerNorm = new LayerNorm(
            4,
            dtype: TensorDType.Float16);
        var feedForward = new FeedForward(
            4,
            8,
            new Random(23),
            dtype: TensorDType.Float16);

        Assert.Equal(TensorDType.Float16, linear.DType);
        Assert.Equal(TensorDType.Float16, layerNorm.DType);
        Assert.Equal(TensorDType.Float16, feedForward.DType);
        Assert.Equal(TensorDType.Float16, feedForward.Fc1.DType);
        Assert.Equal(TensorDType.Float16, feedForward.Fc2.DType);
        Assert.All(
            linear.Parameters()
                .Concat(layerNorm.Parameters())
                .Concat(feedForward.Parameters()),
            parameter =>
                Assert.Equal(TensorDType.Float16, parameter.T.DType));
    }

    [Fact]
    public void ExistingNonV2ModelsRemainFloat32ByDefault()
    {
        Module[] models =
        [
            new TransformerClassifier(
                seqLen: 2,
                dModel: 4,
                numHeads: 1,
                dHidden: 8,
                numLayers: 1,
                numClasses: 3,
                rng: new Random(37)),
            new GptRinWikiJp(
                vocabularySize: 7,
                contextLength: 2,
                dModel: 4,
                numHeads: 1,
                dHidden: 8,
                numLayers: 1,
                rng: new Random(41)),
            new HyenaGpt(
                vocabularySize: 7,
                contextLength: 2,
                modelWidth: 4,
                hiddenWidth: 8,
                numLayers: 1,
                random: new Random(43),
                filterWidth: 4),
            new ForgetScanGpt(
                vocabularySize: 7,
                contextLength: 2,
                modelWidth: 4,
                hiddenWidth: 8,
                numLayers: 1,
                random: new Random(47)),
        ];

        Assert.All(
            models,
            model =>
            {
                Assert.Equal(TensorDType.Float32, model.DType);
                Assert.All(
                    model.Parameters(),
                    parameter => Assert.Equal(
                        TensorDType.Float32,
                        parameter.T.DType));
            });
    }

    [Fact]
    public void ForgetMemoryV2DefaultsToFloat16ThroughoutModelAndForward()
    {
        ForgetMemoryV2Gpt model = CreateModel();

        Assert.Equal(TensorDType.Float16, model.DType);
        Assert.All(
            model.Parameters(),
            parameter =>
                Assert.Equal(TensorDType.Float16, parameter.T.DType));
        Assert.All(model.Layers, AssertFloat16Layer);

        Tensor logits = model.Forward([1, 2, 3], 1, 3);

        Assert.Equal(TensorDType.Float16, logits.DType);
        Assert.Equal(3 * 7, logits.Numel);
        Assert.All(logits.Data, value => Assert.True(float.IsFinite(value)));
    }

    [Fact]
    public void ForgetMemoryV2ExplicitFloat32UsesLegacyStoragePath()
    {
        ForgetMemoryV2Gpt model = CreateModel(TensorDType.Float32);

        Assert.Equal(TensorDType.Float32, model.DType);
        Assert.All(
            model.Parameters(),
            parameter =>
                Assert.Equal(TensorDType.Float32, parameter.T.DType));
        Assert.All(
            model.Layers,
            layer => Assert.Equal(TensorDType.Float32, layer.DType));

        Tensor logits = model.Forward([1, 2, 3], 1, 3);

        Assert.Equal(TensorDType.Float32, logits.DType);
        Assert.All(logits.Data, value => Assert.True(float.IsFinite(value)));
    }

    [Fact]
    public void TorchFactoryDefaultsToFloat16AndAllowsFloat32Override()
    {
        ForgetMemoryV2Gpt defaultModel = nn.forget_memory_v2_lm(
            vocab_size: 7,
            context_length: 3,
            d_model: 4,
            dim_feedforward: 8,
            num_layers: 1,
            key_width: 2,
            value_width: 2,
            generator: new Random(31));
        ForgetMemoryV2Gpt float32Model = nn.forget_memory_v2_lm(
            vocab_size: 7,
            context_length: 3,
            d_model: 4,
            dim_feedforward: 8,
            num_layers: 1,
            key_width: 2,
            value_width: 2,
            generator: new Random(31),
            dtype: TensorDType.Float32);

        Assert.Equal(TensorDType.Float16, defaultModel.DType);
        Assert.Equal(TensorDType.Float32, float32Model.DType);
        Assert.All(
            defaultModel.Parameters(),
            parameter =>
                Assert.Equal(TensorDType.Float16, parameter.T.DType));
        Assert.All(
            float32Model.Parameters(),
            parameter =>
                Assert.Equal(TensorDType.Float32, parameter.T.DType));
    }

    private static ForgetMemoryV2Gpt CreateModel(
        TensorDType dtype = TensorDType.Float16)
        => new(
            vocabularySize: 7,
            contextLength: 3,
            modelWidth: 4,
            hiddenWidth: 8,
            numLayers: 1,
            keyWidth: 2,
            valueWidth: 2,
            random: new Random(29),
            dtype: dtype);

    private static void AssertFloat16Layer(ForgetMemoryV2Layer layer)
    {
        Assert.Equal(TensorDType.Float16, layer.DType);
        Assert.Equal(TensorDType.Float16, layer.Ln1.DType);
        Assert.Equal(TensorDType.Float16, layer.MemoryDropout.DType);
        Assert.Equal(TensorDType.Float16, layer.Ln2.DType);
        Assert.Equal(TensorDType.Float16, layer.Ffn.DType);
        Assert.Equal(TensorDType.Float16, layer.FfnDropout.DType);
    }
}
