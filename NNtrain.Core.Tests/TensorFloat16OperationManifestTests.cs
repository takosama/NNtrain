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
        Assert.Equal(
            reflected,
            manifested.Select(static entry => entry.MemberId).ToArray());
        AssertVerificationTargetsExist(manifested);
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

    private static void AssertVerificationTargetsExist(
        IEnumerable<TensorFloat16OperationManifestEntry> entries)
    {
        Assembly testAssembly = typeof(TensorFloat16OperationManifestTests)
            .Assembly;
        foreach (TensorFloat16OperationManifestEntry entry in entries)
        {
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
