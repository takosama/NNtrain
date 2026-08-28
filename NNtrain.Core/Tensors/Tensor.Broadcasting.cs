using System.Runtime.Intrinsics;

namespace NNtrain;

partial class Tensor
{
    private enum BinaryOperation
    {
        Add,
        Subtract,
        Multiply,
        Divide,
    }

    private delegate float BinaryForward(float left, float right);

    private delegate (float Left, float Right) BinaryDerivative(
        float left,
        float right);

    private static Tensor ApplyBinaryElementwise(
        Tensor left,
        Tensor right,
        BinaryOperation operation,
        BinaryForward forward,
        BinaryDerivative derivative)
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);
        ArgumentNullException.ThrowIfNull(forward);
        ArgumentNullException.ThrowIfNull(derivative);

        BinaryBroadcastPlan plan = BinaryBroadcastPlan.Create(left, right);
        if (ExecutionDevice == TensorDevice.Cuda
            && left.DType is TensorDType.Float32
                or TensorDType.BFloat16
                or TensorDType.Bfp8
            && right.DType is TensorDType.Float32
                or TensorDType.BFloat16
                or TensorDType.Bfp8)
        {
            return ApplyBinaryElementwiseCuda(
                left,
                right,
                plan,
                operation);
        }
        if (ExecutionDevice == TensorDevice.Cuda
            && operation == BinaryOperation.Add
            && (left.DType == TensorDType.Bfp8
                || right.DType == TensorDType.Bfp8))
        {
            if (left.DType != TensorDType.Bfp8
                || right.DType != TensorDType.Bfp8)
            {
                throw new InvalidOperationException(
                    "CUDA BFP8 addition requires both operands to use BFP8 " +
                    "storage; implicit host fallback is forbidden.");
            }
            if (plan.LeftIsScalar
                || plan.RightIsScalar
                || left.Numel != plan.ElementCount
                || right.Numel != plan.ElementCount)
            {
                throw new PlatformNotSupportedException(
                    "Broadcast BFP8 addition has no resident CUDA kernel. " +
                    "CPU fallback is forbidden.");
            }
            return AddBfp8Cuda(left, right, plan.ResultShape);
        }
        if (ExecutionDevice == TensorDevice.Cuda
            && operation == BinaryOperation.Add
            && !plan.LeftIsScalar
            && !plan.RightIsScalar)
        {
            if (left.DType == TensorDType.BFloat16
                && right.DType == TensorDType.BFloat16)
            {
                var bfloat16Buffer =
                    TensorCudaKernels.AddForwardBFloat16Resident(left, right);
                Tensor bfloat16Result = FromCudaResult(
                    bfloat16Buffer,
                    CudaDeviceIndex,
                    plan.ResultShape,
                    [left, right],
                    TensorDType.BFloat16);
                bfloat16Result.Node.BackwardAction = () =>
                    TensorCudaKernels.AddBackwardResident(
                        bfloat16Result,
                        left,
                        right);
                return bfloat16Result;
            }
            var cudaBuffer = TensorCudaKernels.AddForwardResident(
                left,
                right,
                left.DType == TensorDType.BFloat16
                    && right.DType == TensorDType.BFloat16);
            Tensor cudaResult = FromCudaResult(
                cudaBuffer,
                CudaDeviceIndex,
                plan.ResultShape,
                [left, right]);
            cudaResult.Node.BackwardAction = () =>
                TensorCudaKernels.AddBackwardResident(
                    cudaResult,
                    left,
                    right);
            return cudaResult;
        }
        ThrowIfCudaHostFallback($"Elementwise {operation}");
        float[] resultData = new float[plan.ElementCount];

        if (!TryApplyBinaryForwardSimd(
            left,
            right,
            plan,
            operation,
            resultData))
        {
            for (int index = 0; index < plan.ElementCount; index++)
            {
                resultData[index] = forward(
                    plan.LeftValue(left, index),
                    plan.RightValue(right, index));
            }
        }

        var result = new Tensor(
            resultData,
            plan.ResultShape,
            [left, right]);

        result.Node.BackwardAction = () =>
            AccumulateBinaryGradients(
                left,
                right,
                result,
                plan,
                operation,
                derivative);

        return result;
    }

    private static void AccumulateBinaryGradients(
        Tensor left,
        Tensor right,
        Tensor result,
        BinaryBroadcastPlan plan,
        BinaryOperation operation,
        BinaryDerivative derivative)
    {
        if (TryAccumulateBinaryGradientsSimd(
            left,
            right,
            result,
            plan,
            operation))
        {
            return;
        }

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

    private static bool TryApplyBinaryForwardSimd(
        Tensor left,
        Tensor right,
        BinaryBroadcastPlan plan,
        BinaryOperation operation,
        float[] destination)
    {
        if (!CanUseSimd(plan.ElementCount))
            return false;

        int vectorWidth = Vector256<float>.Count;
        int vectorizedLength =
            plan.ElementCount - plan.ElementCount % vectorWidth;
        int index = 0;
        var leftScalar = plan.LeftIsScalar
            ? Vector256.Create(left._data[0])
            : default;
        var rightScalar = plan.RightIsScalar
            ? Vector256.Create(right._data[0])
            : default;

        for (; index < vectorizedLength; index += vectorWidth)
        {
            var leftVector = plan.LeftIsScalar
                ? leftScalar
                : LoadVector256(left._data, index);
            var rightVector = plan.RightIsScalar
                ? rightScalar
                : LoadVector256(right._data, index);

            Vector256<float> resultVector = operation switch
            {
                BinaryOperation.Add => leftVector + rightVector,
                BinaryOperation.Subtract => leftVector - rightVector,
                BinaryOperation.Multiply => leftVector * rightVector,
                BinaryOperation.Divide => leftVector / rightVector,
                _ => throw new InvalidOperationException(
                    $"Unknown binary operation '{operation}'."),
            };
            StoreVector256(resultVector, destination, index);
        }

        for (; index < plan.ElementCount; index++)
        {
            float leftValue = plan.LeftValue(left, index);
            float rightValue = plan.RightValue(right, index);
            destination[index] = operation switch
            {
                BinaryOperation.Add => leftValue + rightValue,
                BinaryOperation.Subtract => leftValue - rightValue,
                BinaryOperation.Multiply => leftValue * rightValue,
                BinaryOperation.Divide => leftValue / rightValue,
                _ => throw new InvalidOperationException(
                    $"Unknown binary operation '{operation}'."),
            };
        }

        return true;
    }

    private static bool TryAccumulateBinaryGradientsSimd(
        Tensor left,
        Tensor right,
        Tensor result,
        BinaryBroadcastPlan plan,
        BinaryOperation operation)
    {
        if (!CanUseSimd(plan.ElementCount))
            return false;

        if (!plan.LeftIsScalar && !plan.RightIsScalar)
        {
            switch (operation)
            {
                case BinaryOperation.Add:
                    AddScaledValues(
                        left._grad, 0, result._grad, 0, 1f, plan.ElementCount);
                    AddScaledValues(
                        right._grad, 0, result._grad, 0, 1f, plan.ElementCount);
                    return true;
                case BinaryOperation.Subtract:
                    AddScaledValues(
                        left._grad, 0, result._grad, 0, 1f, plan.ElementCount);
                    AddScaledValues(
                        right._grad, 0, result._grad, 0, -1f, plan.ElementCount);
                    return true;
                case BinaryOperation.Multiply:
                    AddProductValues(
                        left._grad,
                        0,
                        right._data,
                        0,
                        result._grad,
                        0,
                        1f,
                        plan.ElementCount);
                    AddProductValues(
                        right._grad,
                        0,
                        left._data,
                        0,
                        result._grad,
                        0,
                        1f,
                        plan.ElementCount);
                    return true;
                case BinaryOperation.Divide:
                    AccumulateDivisionGradientsSimd(left, right, result);
                    return true;
            }
        }

        if (plan.RightIsScalar && !plan.LeftIsScalar)
        {
            float scalar = right._data[0];
            switch (operation)
            {
                case BinaryOperation.Add:
                    AddScaledValues(
                        left._grad, 0, result._grad, 0, 1f, plan.ElementCount);
                    right._grad[0] += SumValues(
                        result._grad, 0, plan.ElementCount);
                    return true;
                case BinaryOperation.Subtract:
                    AddScaledValues(
                        left._grad, 0, result._grad, 0, 1f, plan.ElementCount);
                    right._grad[0] -= SumValues(
                        result._grad, 0, plan.ElementCount);
                    return true;
                case BinaryOperation.Multiply:
                    AddScaledValues(
                        left._grad,
                        0,
                        result._grad,
                        0,
                        scalar,
                        plan.ElementCount);
                    right._grad[0] += DotProduct(
                        left._data,
                        0,
                        result._grad,
                        0,
                        plan.ElementCount);
                    return true;
                case BinaryOperation.Divide:
                    AddScaledValues(
                        left._grad,
                        0,
                        result._grad,
                        0,
                        1f / scalar,
                        plan.ElementCount);
                    right._grad[0] -= DotProduct(
                            left._data,
                            0,
                            result._grad,
                            0,
                            plan.ElementCount)
                        / (scalar * scalar);
                    return true;
            }
        }

        if (plan.LeftIsScalar && !plan.RightIsScalar)
        {
            float scalar = left._data[0];
            switch (operation)
            {
                case BinaryOperation.Add:
                    left._grad[0] += SumValues(
                        result._grad, 0, plan.ElementCount);
                    AddScaledValues(
                        right._grad, 0, result._grad, 0, 1f, plan.ElementCount);
                    return true;
                case BinaryOperation.Subtract:
                    left._grad[0] += SumValues(
                        result._grad, 0, plan.ElementCount);
                    AddScaledValues(
                        right._grad, 0, result._grad, 0, -1f, plan.ElementCount);
                    return true;
                case BinaryOperation.Multiply:
                    left._grad[0] += DotProduct(
                        right._data,
                        0,
                        result._grad,
                        0,
                        plan.ElementCount);
                    AddScaledValues(
                        right._grad,
                        0,
                        result._grad,
                        0,
                        scalar,
                        plan.ElementCount);
                    return true;
            }
        }

        return false;
    }

    private static void AccumulateDivisionGradientsSimd(
        Tensor left,
        Tensor right,
        Tensor result)
    {
        int length = result.Numel;
        int vectorWidth = Vector256<float>.Count;
        int vectorizedLength = length - length % vectorWidth;
        int index = 0;
        Vector256<float> minusOne = Vector256.Create(-1f);

        for (; index < vectorizedLength; index += vectorWidth)
        {
            Vector256<float> leftValue = LoadVector256(
                left._data,
                index);
            Vector256<float> rightValue = LoadVector256(
                right._data,
                index);
            Vector256<float> gradient = LoadVector256(
                result._grad,
                index);
            Vector256<float> leftGradient = LoadVector256(
                left._grad,
                index);
            Vector256<float> rightGradient = LoadVector256(
                right._grad,
                index);

            StoreVector256(
                leftGradient + gradient / rightValue,
                left._grad,
                index);
            StoreVector256(
                rightGradient
                    + minusOne
                        * leftValue
                        * gradient
                        / (rightValue * rightValue),
                        right._grad,
                        index);
        }

        for (; index < length; index++)
        {
            float rightValue = right._data[index];
            float gradient = result._grad[index];
            left._grad[index] += gradient / rightValue;
            right._grad[index] +=
                -left._data[index]
                * gradient
                / (rightValue * rightValue);
        }
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
