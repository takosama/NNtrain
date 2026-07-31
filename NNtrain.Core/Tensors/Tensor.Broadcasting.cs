namespace NNtrain;

partial class Tensor
{
    private delegate float BinaryForward(float left, float right);

    private delegate (float Left, float Right) BinaryDerivative(
        float left,
        float right);

    private static Tensor ApplyBinaryElementwise(
        Tensor left,
        Tensor right,
        BinaryForward forward,
        BinaryDerivative derivative)
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);
        ArgumentNullException.ThrowIfNull(forward);
        ArgumentNullException.ThrowIfNull(derivative);

        BinaryBroadcastPlan plan = BinaryBroadcastPlan.Create(left, right);
        float[] resultData = new float[plan.ElementCount];

        for (int index = 0; index < plan.ElementCount; index++)
        {
            resultData[index] = forward(
                plan.LeftValue(left, index),
                plan.RightValue(right, index));
        }

        var result = new Tensor(
            resultData,
            plan.ResultShape,
            [left, right]);

        result.Node.BackwardAction = () =>
            AccumulateBinaryGradients(left, right, result, plan, derivative);

        return result;
    }

    private static void AccumulateBinaryGradients(
        Tensor left,
        Tensor right,
        Tensor result,
        BinaryBroadcastPlan plan,
        BinaryDerivative derivative)
    {
        float reducedLeftGradient = 0f;
        float reducedRightGradient = 0f;

        for (int index = 0; index < plan.ElementCount; index++)
        {
            float leftValue = plan.LeftValue(left, index);
            float rightValue = plan.RightValue(right, index);
            (float leftDerivative, float rightDerivative) =
                derivative(leftValue, rightValue);
            float upstreamGradient = result._grad[index];

            float leftContribution = leftDerivative * upstreamGradient;
            if (plan.LeftIsScalar)
                reducedLeftGradient += leftContribution;
            else
                left._grad[index] += leftContribution;

            float rightContribution = rightDerivative * upstreamGradient;
            if (plan.RightIsScalar)
                reducedRightGradient += rightContribution;
            else
                right._grad[index] += rightContribution;
        }

        if (plan.LeftIsScalar)
            left._grad[0] += reducedLeftGradient;
        if (plan.RightIsScalar)
            right._grad[0] += reducedRightGradient;
    }

    private readonly struct BinaryBroadcastPlan
    {
        private BinaryBroadcastPlan(
            int elementCount,
            int[] resultShape,
            bool leftIsScalar,
            bool rightIsScalar)
        {
            ElementCount = elementCount;
            ResultShape = resultShape;
            LeftIsScalar = leftIsScalar;
            RightIsScalar = rightIsScalar;
        }

        internal int ElementCount { get; }
        internal int[] ResultShape { get; }
        internal bool LeftIsScalar { get; }
        internal bool RightIsScalar { get; }

        internal static BinaryBroadcastPlan Create(Tensor left, Tensor right)
        {
            bool leftIsScalar = left.Numel == 1;
            bool rightIsScalar = right.Numel == 1;

            if (left._shape.AsSpan().SequenceEqual(right._shape))
            {
                return new BinaryBroadcastPlan(
                    left.Numel,
                    (int[])left._shape.Clone(),
                    leftIsScalar,
                    rightIsScalar);
            }

            if (leftIsScalar)
            {
                return new BinaryBroadcastPlan(
                    right.Numel,
                    (int[])right._shape.Clone(),
                    leftIsScalar: true,
                    rightIsScalar);
            }

            if (rightIsScalar)
            {
                return new BinaryBroadcastPlan(
                    left.Numel,
                    (int[])left._shape.Clone(),
                    leftIsScalar,
                    rightIsScalar: true);
            }

            throw ShapeMismatch(left, right, "Element-wise broadcasting");
        }

        internal float LeftValue(Tensor left, int index)
            => left._data[LeftIsScalar ? 0 : index];

        internal float RightValue(Tensor right, int index)
            => right._data[RightIsScalar ? 0 : index];
    }
}
