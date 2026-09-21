using System.Reflection;
using NNtrain;
using Xunit;

public sealed class TensorFloat16OperationManifestTests
{
    [Fact]
    public void PublicTensorReturningMembersAreCompletelyRepresentedInTheManifest()
    {
        string[] reflected = typeof(Tensor)
            .GetMethods(
                BindingFlags.DeclaredOnly
                | BindingFlags.Public
                | BindingFlags.Instance
                | BindingFlags.Static)
            .Where(static method => method.ReturnType == typeof(Tensor))
            .Select(ToMemberId)
            .Order()
            .ToArray();
        TensorFloat16OperationManifestEntry[] manifested =
            TensorFloat16OperationManifest.PublicTensorReturningMembers
                .OrderBy(static entry => entry.MemberId)
                .ToArray();

        Assert.Equal(
            manifested.Length,
            manifested.Select(static entry => entry.MemberId).Distinct().Count());
        AssertCompleteInventory(reflected, manifested);
        Assert.Equal(
            reflected,
            manifested.Select(static entry => entry.MemberId).ToArray());
        AssertVerificationTargetsExist(manifested);
    }

    [Fact]
    public void InternalTensorReturningKernelsAreCompletelyRepresentedInTheManifest()
    {
        string[] reflected = typeof(Tensor)
            .GetMethods(
                BindingFlags.DeclaredOnly
                | BindingFlags.NonPublic
                | BindingFlags.Instance
                | BindingFlags.Static)
            .Where(static method => method.IsAssembly)
            .Where(static method => method.ReturnType == typeof(Tensor))
            .Select(ToMemberId)
            .Order()
            .ToArray();
        TensorFloat16OperationManifestEntry[] manifested =
            TensorFloat16OperationManifest.InternalTensorReturningMembers
                .OrderBy(static entry => entry.MemberId)
                .ToArray();

        Assert.Equal(
            manifested.Length,
            manifested.Select(static entry => entry.MemberId).Distinct().Count());
        AssertCompleteInventory(reflected, manifested);
        Assert.Equal(
            reflected,
            manifested.Select(static entry => entry.MemberId).ToArray());
        AssertVerificationTargetsExist(manifested);
    }

    [Fact]
    public void BackendOnlyOperationsHaveExplicitFloat16Restrictions()
    {
        string[] backendMembers =
        [
            "ArcCheckpoint(Func`2,IReadOnlyList`1)",
            "ArcLinearCrossEntropy(Tensor,Tensor,Int32[],Int32)",
            "LinearLastDimFrozen(Tensor,Tensor,Boolean)",
        ];
        foreach (string member in backendMembers)
        {
            TensorFloat16OperationManifestEntry entry = Assert.Single(
                TensorFloat16OperationManifest.InternalTensorReturningMembers,
                candidate => candidate.MemberId == member);
            Assert.Equal(TensorFloat16ResultPolicy.BackendWithoutFloat16, entry.ResultPolicy);
            Assert.False(string.IsNullOrWhiteSpace(entry.Float16Restriction));
            Assert.Contains("Float16", entry.Float16Restriction);
        }
    }

    [Fact]
    public void ArcCheckpointCpuPassthroughPreservesDelegateContract()
    {
        using var execution = TensorExecutionContext.Push(new TorchDevice(TensorDevice.Cpu));
        Tensor input = Half([-1f, .5f, -2f, 1f], 4);
        Tensor output = input.ArcCheckpoint(static value => value.Relu(), []);
        AssertFloat16StorageContract(output);
        output.Sum().BackwardAndRelease();
        Assert.Equal(new float[] { 0, 1, 0, 1 }, input.Grad);
    }

    [Fact]
    public void FrozenLinearRejectsLegacyFloat16()
    {
        using var execution = TensorExecutionContext.Push(new TorchDevice(TensorDevice.Cpu));
        Tensor input = Half([1f, 2f], 1, 2);
        Tensor weight = Half([.25f, .5f], 1, 2);
        Tensor bias = Half([0f], 1);
        Assert.Throws<NotSupportedException>(() => input.LinearLastDimFrozen(weight, bias, applyRelu: false));
    }

    [Fact]
    public void DpoLossReadsFloat16ScoresAndReturnsFloat32LossAndGradients()
    {
        using var execution = TensorExecutionContext.Push(new TorchDevice(TensorDevice.Cpu));
        float[] scores = [.5f, 1f, 2f, 4f];
        Tensor[] losses = scores.Select(value => Tensor.Scalar(value, dtype: TensorDType.Float16)).ToArray();
        int[] counts = [3, 5, 2, 7];
        Tensor result = Tensor.DpoLoss(losses, counts, scores, beta: .25f);

        Assert.All(losses, AssertFloat16StorageContract);
        Assert.Equal(TensorDType.Float32, result.DType);
        Assert.Equal(TensorDType.Float32, result.ComputeDType);
        Assert.Equal(TensorDType.Float32, result.AccumulationDType);
        Assert.Equal(sizeof(float), result.StorageByteLength);
        Assert.Equal((float)Math.Log(2), result.item());

        result.BackwardAndRelease([2f]);
        // Both margins are zero: sigmoid(-margin)=1/2. The batch has two
        // pairs, with a nonunit backward seed and token-count weighting.
        Assert.Equal(new float[] { .375f, -.625f, .25f, -.875f },
            losses.Select(loss => Assert.Single(loss.Grad)).ToArray());
    }

    [Fact]
    public void FactoriesAndConversionsSupportFloat16()
    {
        float[] owned = [0.25f, -0.5f, 0.75f, -1f];
        Tensor[] halfFactories =
        [
            new Tensor(owned, [4], dtype: TensorDType.Float16),
            Tensor.FromOwnedData(
                [0.25f, -0.5f, 0.75f, -1f],
                [4],
                dtype: TensorDType.Float16),
            Tensor.Scalar(0.25f, dtype: TensorDType.Float16),
            Tensor.tensor(owned, [4], dtype: TensorDType.Float16),
            Tensor.Zeros(TensorDType.Float16, 2, 2),
            Tensor.From1D(owned, dtype: TensorDType.Float16),
            Tensor.From2D(
                new float[,] { { 0.25f, -0.5f }, { 0.75f, -1f } },
                dtype: TensorDType.Float16),
        ];

        Assert.All(halfFactories, AssertFloat16StorageContract);
        Assert.Equal(TensorDType.Float32, Tensor.Zeros(2, 2).DType);

        var float32 = new Tensor(owned, [4]);
        Tensor[] convertedToHalf =
        [
            float32.To(TensorDType.Float16),
            float32.to(TensorDType.Float16),
            float32.Half(),
            float32.half(),
        ];
        Assert.All(convertedToHalf, AssertFloat16StorageContract);
        Assert.All(
            convertedToHalf,
            static tensor => Assert.Equal(
                TensorDType.Float32,
                tensor.ToFloat32().DType));
    }

    [Fact]
    public void Float16AuxiliaryApisReadAndClearGradients()
    {
        Tensor scalar = Tensor.Scalar(1.25f, dtype: TensorDType.Float16);
        Assert.Equal(1.25f, scalar.item());

        Tensor input = Half(Pattern(8), 8);
        Tensor output = input.Relu();
        output.Backward(Enumerable.Repeat(0.5f, output.Numel).ToArray());

        Assert.NotEmpty(input.DataString());
        Assert.NotEmpty(input.GradString());
        Assert.Contains(input.Grad, static value => value != 0f);
        input.ZeroGrad();
        Assert.All(input.Grad, static value => Assert.Equal(0f, value));
        AssertVerificationTargetsExist(
            TensorFloat16OperationManifest.AuxiliaryMembers);
    }

    [Fact]
    public void MatrixAndBatchedOperationsPreserveFloat16StorageAndFloat32Gradients()
    {
        Tensor vectorLeft = Half(Pattern(8), 8);
        Tensor vectorRight = Half(Pattern(8, offset: 0.1f), 8);
        AssertHalfOutput(vectorLeft.MatMul(vectorRight), vectorLeft, vectorRight);

        Tensor matrixVectorLeft = Half(Pattern(16), 2, 8);
        Tensor matrixVectorRight = Half(Pattern(8, offset: -0.15f), 8);
        AssertHalfOutput(
            matrixVectorLeft.MatMul(matrixVectorRight),
            matrixVectorLeft,
            matrixVectorRight);

        Tensor matrixLeft = Half(Pattern(16), 2, 8);
        Tensor matrixRight = Half(Pattern(32, offset: 0.2f), 8, 4);
        AssertHalfOutput(matrixLeft.MatMul(matrixRight), matrixLeft, matrixRight);

        Tensor transposedLeft = Half(Pattern(16), 2, 8);
        Tensor transposedRight = Half(Pattern(32, offset: -0.1f), 4, 8);
        AssertHalfOutput(
            transposedLeft.MatMulTransposedRight(transposedRight),
            transposedLeft,
            transposedRight);

        Tensor biasedLeft = Half(Pattern(16), 2, 8);
        Tensor biasedRight = Half(Pattern(32, offset: 0.1f), 4, 8);
        Tensor bias = Half(Enumerable.Repeat(0.5f, 4).ToArray(), 4);
        AssertHalfOutput(
            biasedLeft.MatMulTransposedRightAddRow(biasedRight, bias),
            biasedLeft,
            biasedRight,
            bias);

        Tensor reluLeft = Half(Pattern(16), 2, 8);
        Tensor reluRight = Half(Pattern(32, offset: 0.1f), 4, 8);
        Tensor reluBias = Half(Enumerable.Repeat(0.5f, 4).ToArray(), 4);
        AssertHalfOutput(
            reluLeft.MatMulTransposedRightAddRowRelu(reluRight, reluBias),
            reluLeft,
            reluRight,
            reluBias);

        Tensor batchLeft = Half(Pattern(2 * 2 * 8), 2, 2, 8);
        Tensor batchRight = Half(Pattern(2 * 8 * 4, offset: 0.15f), 2, 8, 4);
        AssertHalfOutput(
            batchLeft.BatchedMatMul(batchRight),
            batchLeft,
            batchRight);

        Tensor batchTransposedLeft = Half(Pattern(2 * 2 * 8), 2, 2, 8);
        Tensor batchTransposedRight = Half(
            Pattern(2 * 4 * 8, offset: -0.2f),
            2,
            4,
            8);
        AssertHalfOutput(
            batchTransposedLeft.BatchedMatMulTransposedRight(
                batchTransposedRight),
            batchTransposedLeft,
            batchTransposedRight);
    }

    private static void AssertHalfOutput(Tensor output, params Tensor[] parents)
    {
        AssertFloat16StorageContract(output);
        Assert.All(output.Data, static value => Assert.True(float.IsFinite(value)));

        output.Sum().Backward();
        Assert.All(
            parents,
            static parent => Assert.All(
                parent.Grad,
                static value => Assert.True(float.IsFinite(value))));
    }

    private static void AssertFloat16StorageContract(Tensor tensor)
    {
        Assert.Equal(TensorDType.Float16, tensor.DType);
        Assert.Equal(TensorDType.Float32, tensor.ComputeDType);
        Assert.Equal(TensorDType.Float32, tensor.AccumulationDType);
        Assert.Equal(tensor.Numel * sizeof(ushort), tensor.StorageByteLength);
    }

    private static Tensor Half(float[] values, params int[] shape)
        => new(values, shape, dtype: TensorDType.Float16);

    private static float[] Pattern(int count, float offset = 0f)
        => Enumerable.Range(0, count)
            .Select(index => offset + (((index * 17) % 23) - 11) * 0.03125f)
            .ToArray();

    private static string ToMemberId(MethodInfo method)
        => $"{method.Name}({string.Join(",", method.GetParameters()
            .Select(static parameter => parameter.ParameterType.Name))})";

    private static void AssertCompleteInventory(
        string[] reflected,
        IReadOnlyList<TensorFloat16OperationManifestEntry> manifested)
    {
        string[] ids = manifested.Select(static entry => entry.MemberId).ToArray();
        Assert.True(reflected.SequenceEqual(ids),
            "Tensor operation inventory mismatch.\nMissing manifest entries:\n"
            + string.Join("\n", reflected.Except(ids))
            + "\nUnexpected manifest entries:\n"
            + string.Join("\n", ids.Except(reflected)));
    }

    private static void AssertVerificationTargetsExist(
        IEnumerable<TensorFloat16OperationManifestEntry> entries)
    {
        Assembly testAssembly = typeof(TensorFloat16OperationManifestTests)
            .Assembly;
        foreach (TensorFloat16OperationManifestEntry entry in entries)
        {
            if (entry.ResultPolicy == TensorFloat16ResultPolicy.BackendWithoutFloat16)
            {
                Assert.False(string.IsNullOrWhiteSpace(entry.Float16Restriction),
                    $"Backend-only operation '{entry.MemberId}' must explain its Float16 restriction.");
            }
            int separator = entry.Verification.LastIndexOf('.');
            Assert.True(
                separator > 0 && separator < entry.Verification.Length - 1,
                $"Invalid verification target '{entry.Verification}' for " +
                $"'{entry.MemberId}'.");
            string typeName = entry.Verification[..separator];
            string methodName = entry.Verification[(separator + 1)..];
            Type? type = testAssembly.GetTypes().SingleOrDefault(
                candidate => candidate.Name == typeName);
            MethodInfo? method = type?.GetMethod(
                methodName,
                BindingFlags.DeclaredOnly
                | BindingFlags.Public
                | BindingFlags.Instance
                | BindingFlags.Static);

            Assert.True(
                method is not null,
                $"Verification target '{entry.Verification}' for " +
                $"'{entry.MemberId}' does not exist.");
        }
    }
}
