using NNtrain;
using Xunit;
using static TensorCharacterizationTests;

public sealed class AutogradTraversalTests
{
    [Fact]
    public void BackwardHandlesFiftyThousandDeepSharedGraph()
    {
        const int depth = 50_000;
        var input = Tensor.Scalar(1f);
        Tensor shared = input;

        for (int index = 0; index < depth; index++)
            shared = -shared;

        Tensor output = shared + shared;
        output.Backward();

        AssertClose([2f], shared.Grad);
        AssertClose([2f], input.Grad);
    }

    [Fact]
    public void DeepGraphCanRunBackwardRepeatedly()
    {
        const int depth = 10_001;
        var input = Tensor.Scalar(1f);
        Tensor output = input;

        for (int index = 0; index < depth; index++)
            output = -output;

        output.Backward();
        output.Backward();

        AssertClose([-2f], input.Grad);
        AssertClose([1f], output.Grad);
    }
}
