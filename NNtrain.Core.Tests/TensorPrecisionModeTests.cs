using NNtrain;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class TensorPrecisionModeTests
{
    [Fact]
    public void PureBfp8ParameterHasNoFloat32MasterWhileMix8Does()
    {
        var pure = new Parameter(
            [0.25f, -0.5f],
            [2],
            "pure",
            WeightDecayPolicy.Apply,
            TensorDType.Bfp8);
        Assert.False(pure.T.HasFloat32MasterData);

        pure.T.ConvertStorageInPlace(
            TensorDType.Bfp8,
            Bfp8QuantizationDescriptor.Block(64),
            preserveFloat32Master: true);
        Assert.True(pure.T.HasFloat32MasterData);
    }

    [Fact]
    public void BFloat16StorageUsesFloat32Accumulation()
    {
        var tensor = new Tensor([1f, -2f], [2], dtype: TensorDType.BFloat16);

        Assert.Equal(TensorDType.BFloat16, tensor.ComputeDType);
        Assert.Equal(TensorDType.Float32, tensor.AccumulationDType);
    }

    [Theory]
    [InlineData("float32", TensorPrecisionMode.Float32, TensorDType.Float32)]
    [InlineData("bfloat16", TensorPrecisionMode.BFloat16, TensorDType.BFloat16)]
    [InlineData("mix16_32", TensorPrecisionMode.Mix16_32, TensorDType.BFloat16)]
    [InlineData("bfp8", TensorPrecisionMode.Bfp8, TensorDType.Bfp8)]
    [InlineData("mix8_32", TensorPrecisionMode.Mix8_32, TensorDType.Bfp8)]
    [InlineData("mix8_16", TensorPrecisionMode.Mix8_16, TensorDType.Bfp8)]
    public void CanonicalNamesMapToOneStorageContract(
        string name,
        TensorPrecisionMode expectedMode,
        TensorDType expectedStorage)
    {
        TensorPrecisionMode mode = TensorPrecisionModeNames.Parse(name);

        Assert.Equal(expectedMode, mode);
        Assert.Equal(expectedStorage, mode.ToStorageDType());
        Assert.Equal(name, TensorPrecisionModeNames.Format(mode));
    }

    [Theory]
    [InlineData(TensorDType.Float32, TensorPrecisionMode.Float32)]
    [InlineData(TensorDType.BFloat16, TensorPrecisionMode.BFloat16)]
    [InlineData(TensorDType.Float16, TensorPrecisionMode.Mix16_32)]
    [InlineData(TensorDType.Bfp8, TensorPrecisionMode.Bfp8)]
    public void RawStorageDTypesHaveStableDefaultModes(
        TensorDType dtype,
        TensorPrecisionMode expectedMode)
        => Assert.Equal(expectedMode, dtype.ToPrecisionMode());

    [Fact]
    public void PrecisionModeParserRejectsStorageAliases()
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(
            () => TensorPrecisionModeNames.Parse("float16"));

        Assert.Contains("mix16_32", exception.Message);
    }

    [Fact]
    public void PrecisionModeParserAcceptsFp16_32CompatibilityAlias()
    {
        Assert.Equal(
            TensorPrecisionMode.Mix16_32,
            TensorPrecisionModeNames.Parse("fp16_32"));
        Assert.Equal(
            "mix16_32",
            TensorPrecisionModeNames.Format(TensorPrecisionMode.Mix16_32));
    }

    [Fact]
    public void ModulePrecisionModePropagatesAndRejectsWrongStorage()
    {
        var model = new GptRinWikiJp(
            vocabularySize: 32,
            contextLength: 2,
            dModel: 4,
            numHeads: 1,
            dHidden: 8,
            numLayers: 1,
            rng: new Random(7),
            dtype: TensorDType.BFloat16);

        model.SetPrecisionMode(TensorPrecisionMode.Mix16_32);

        Assert.Equal(TensorPrecisionMode.Mix16_32, model.PrecisionMode);
        Assert.All(
            model.Parameters(),
            parameter => Assert.Equal(
                TensorDType.BFloat16,
                parameter.T.DType));
        Assert.Throws<InvalidOperationException>(
            () => model.SetPrecisionMode(TensorPrecisionMode.Float32));
    }

    [Fact]
    public void ModuleToConvertsStorageWithoutReplacingParametersOrTensors()
    {
        var model = new Linear(4, 3, new Random(17));
        Parameter[] parameters = model.parameters().ToArray();
        Tensor[] tensors = parameters.Select(parameter => parameter.T).ToArray();
        float[][] original = tensors
            .Select(tensor => tensor.Data.ToArray())
            .ToArray();
        var optimizer = new AdamW(
            parameters,
            new AdamWOptions { LearningRate = 1e-3f, WeightDecay = 0f });

        Assert.Same(model, model.to(TensorPrecisionMode.Mix16_32));

        Assert.Equal(TensorPrecisionMode.Mix16_32, model.PrecisionMode);
        Assert.Equal(TensorDType.BFloat16, model.DType);
        Assert.Equal(parameters, model.parameters().ToArray());
        for (int index = 0; index < tensors.Length; index++)
        {
            Assert.Same(tensors[index], parameters[index].T);
            Assert.Equal(TensorDType.BFloat16, tensors[index].DType);
            Assert.Equal(
                original[index],
                tensors[index].Data,
                new BFloat16ToleranceComparer());
            tensors[index].MutableGrad.Fill(0.125f);
        }

        optimizer.Step();

        Assert.Equal(parameters, optimizer.Parameters);
        Assert.All(tensors, tensor => Assert.All(tensor.Data, AssertFinite));

        model.to("float32");

        Assert.Equal(TensorPrecisionMode.Float32, model.PrecisionMode);
        Assert.Equal(TensorDType.Float32, model.DType);
        for (int index = 0; index < tensors.Length; index++)
        {
            Assert.Same(tensors[index], parameters[index].T);
            Assert.Equal(TensorDType.Float32, tensors[index].DType);
        }
    }

    [Fact]
    public void ModuleToSupportsBfp8AndConfigurableMix8BlockScales()
    {
        var model = new Linear(16, 4, new Random(23));
        Tensor[] tensors = model.parameters()
            .Select(parameter => parameter.T)
            .ToArray();

        model.to(TensorPrecisionMode.Mix8_32, bfp8_block_size: 32);

        Assert.Equal(TensorPrecisionMode.Mix8_32, model.PrecisionMode);
        Assert.All(
            tensors,
            tensor =>
            {
                Assert.Equal(TensorDType.Bfp8, tensor.DType);
                Assert.Equal(
                    Bfp8ScaleGranularity.Block,
                    tensor.Bfp8Quantization!.Granularity);
                Assert.Equal(32, tensor.Bfp8Quantization.BlockSize);
            });

        model.to("bfp8");

        Assert.Equal(TensorPrecisionMode.Bfp8, model.PrecisionMode);
        Assert.All(
            tensors,
            tensor => Assert.Equal(
                Bfp8ScaleGranularity.Tensor,
                tensor.Bfp8Quantization!.Granularity));
    }

    [Theory]
    [InlineData("fp16_32", TensorPrecisionMode.Mix16_32)]
    [InlineData("mix16_32", TensorPrecisionMode.Mix16_32)]
    [InlineData("bfloat16", TensorPrecisionMode.BFloat16)]
    [InlineData("mix8_32", TensorPrecisionMode.Mix8_32)]
    [InlineData("mix8_16", TensorPrecisionMode.Mix8_16)]
    public void ModuleToAcceptsCanonicalPrecisionStringsAndFpAlias(
        string target,
        TensorPrecisionMode expected)
    {
        var model = new Linear(2, 2, new Random(29));

        Assert.Same(model, model.to(target));

        Assert.Equal(expected, model.PrecisionMode);
    }

    [Fact]
    public void ModuleToRejectsUnknownStringWithoutMutatingModel()
    {
        var model = new Linear(2, 2, new Random(31));
        Tensor[] tensors = model.parameters()
            .Select(parameter => parameter.T)
            .ToArray();

        Assert.Throws<ArgumentException>(() => model.to("fp8_magic"));

        Assert.Equal(TensorPrecisionMode.Float32, model.PrecisionMode);
        Assert.All(tensors, tensor => Assert.Equal(TensorDType.Float32, tensor.DType));
    }

    [Theory]
    [InlineData(TensorPrecisionMode.Bfp8)]
    [InlineData(TensorPrecisionMode.Mix8_32)]
    [InlineData(TensorPrecisionMode.Mix8_16)]
    public void ModuleStateRoundTripsBfp8WithoutChangingItsNumericContract(
        TensorPrecisionMode mode)
    {
        var source = new Linear(16, 4, new Random(37));
        source.to(mode, bfp8_block_size: 32);
        ModuleState state = source.state_dict();
        var destination = new Linear(16, 4, new Random(41));
        destination.to(mode, bfp8_block_size: 32);

        destination.load_state_dict(state);

        Assert.Equal(mode, destination.PrecisionMode);
        Tensor[] expected = source.parameters()
            .Select(parameter => parameter.T)
            .ToArray();
        Tensor[] actual = destination.parameters()
            .Select(parameter => parameter.T)
            .ToArray();
        Assert.Equal(expected.Length, actual.Length);
        for (int index = 0; index < expected.Length; index++)
        {
            Assert.Equal(TensorDType.Bfp8, actual[index].DType);
            Assert.Equal(
                expected[index].Bfp8Quantization,
                actual[index].Bfp8Quantization);
            Assert.Equal(expected[index].Data, actual[index].Data);
            Assert.Equal(
                mode == TensorPrecisionMode.Bfp8,
                actual[index].RequiresTwoPassBfp8CheckpointRestore);
        }
    }

    [Fact]
    public void Mix8_16HasBlockStorageAndBFloat16RetainedStateContract()
    {
        PrecisionPolicy policy = PrecisionPolicy.Parse("mix8_16");
        Assert.Equal(PrecisionMode.Mix8_16, policy.Mode);
        Assert.Equal("mix8_16", policy.ToString());
        Assert.Same(policy, PrecisionPolicy.For(PrecisionMode.Mix8_16));
        Assert.Equal(NumericFormat.Bfp8, policy.ParameterStorage);
        Assert.Equal(NumericFormat.Bfp8, policy.ActivationStorage);
        Assert.Equal(NumericFormat.Bfp8, policy.ElementwiseCompute);
        Assert.Equal(NumericFormat.Bfp8, policy.MatrixOperand);
        Assert.Equal(
            GemmExecutionFormat.Int8 | GemmExecutionFormat.BFloat16,
            policy.GemmExecutionFormats);
        Assert.True(policy.PreferInt8WeightGemm);
        Assert.Equal(
            NumericFormatSet.Bfp8 | NumericFormatSet.BFloat16,
            policy.AllowedActivationStorageFormats);
        Assert.Equal(
            NumericFormatSet.Bfp8 | NumericFormatSet.BFloat16
                | NumericFormatSet.Float32,
            policy.AllowedNonWeightComputeFormats);
        Assert.Equal(NonWeightSelectionPolicy.FastestAvailable,
            policy.NonWeightSelection);
        Assert.True(policy.AllowNonWeightReassociation);
        Assert.Equal(NumericFormat.Float32, policy.Accumulation);
        Assert.Equal(NumericFormat.Float32, policy.Reduction);
        Assert.Equal(NumericFormat.Float32, policy.Normalization);
        Assert.Equal(NumericFormat.Float32, policy.Loss);
        Assert.Equal(NumericFormat.BFloat16, policy.Gradient);
        Assert.Equal(NumericFormat.BFloat16, policy.OptimizerState);
        Assert.Equal(NumericFormat.BFloat16, policy.MasterWeight);

        PrecisionPolicy mix8_32 = PrecisionPolicy.Mix8_32;
        Assert.Equal(GemmExecutionFormat.BFloat16,
            mix8_32.GemmExecutionFormats);
        Assert.Equal(NumericFormatSet.Bfp8,
            mix8_32.AllowedActivationStorageFormats);
        Assert.Equal(NonWeightSelectionPolicy.Fixed,
            mix8_32.NonWeightSelection);
        Assert.False(mix8_32.AllowNonWeightReassociation);

        var model = new Linear(16, 4, new Random(47));
        model.to(TensorPrecisionMode.Mix8_16, bfp8_block_size: 32);
        Assert.Equal(TensorPrecisionMode.Mix8_16, model.PrecisionMode);
        Assert.Equal(TensorDType.Bfp8, model.DType);
        Assert.All(model.parameters(), parameter =>
        {
            Assert.Equal(Bfp8ScaleGranularity.Block,
                parameter.T.Bfp8Quantization!.Granularity);
            Assert.Equal(32, parameter.T.Bfp8Quantization.BlockSize);
            Assert.True(parameter.T.HasBFloat16HostMaster);
            Assert.False(parameter.T.HasFloat32HostMaster);
        });
    }

    private sealed class BFloat16ToleranceComparer : IEqualityComparer<float>
    {
        public bool Equals(float left, float right)
            => MathF.Abs(left - right) <= MathF.Max(1e-5f, MathF.Abs(left) / 128f);

        public int GetHashCode(float value) => value.GetHashCode();
    }

    private static void AssertFinite(float value)
        => Assert.True(float.IsFinite(value));
}
