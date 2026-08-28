namespace NNtrain;

partial class Tensor
{
    public static Tensor operator +(Tensor left, Tensor right)
        => ApplyBinaryElementwise(
            left,
            right,
            BinaryOperation.Add,
            static (leftValue, rightValue) => leftValue + rightValue,
            static (_, _) => (1f, 1f));

    public static Tensor operator -(Tensor left, Tensor right)
        => ApplyBinaryElementwise(
            left,
            right,
            BinaryOperation.Subtract,
            static (leftValue, rightValue) => leftValue - rightValue,
            static (_, _) => (1f, -1f));

    public static Tensor operator -(Tensor value)
    {
        ArgumentNullException.ThrowIfNull(value);
        if (ExecutionDevice == TensorDevice.Cuda
            && value.DType is TensorDType.Float32
                or TensorDType.BFloat16
                or TensorDType.Bfp8)
        {
            return value.ApplyUnaryCuda(CudaPublicUnaryOperation.Negate);
        }
        ThrowIfCudaHostFallback("Unary negation");

        float[] resultData = new float[value.Numel];
        MultiplyValues(
            value._data,
            0,
            -1f,
            resultData,
            0,
            value.Numel);

        var result = new Tensor(resultData, value._shape, [value]);

        result.Node.BackwardAction = () =>
        {
            AddScaledValues(
                value._grad,
                0,
                result._grad,
                0,
                -1f,
                value.Numel);
        };

        return result;
    }

    public static Tensor operator *(Tensor left, Tensor right)
        => ApplyBinaryElementwise(
            left,
            right,
            BinaryOperation.Multiply,
            static (leftValue, rightValue) => leftValue * rightValue,
            static (leftValue, rightValue) => (rightValue, leftValue));

    public static Tensor operator /(Tensor left, Tensor right)
        => ApplyBinaryElementwise(
            left,
            right,
            BinaryOperation.Divide,
            static (leftValue, rightValue) => leftValue / rightValue,
            static (leftValue, rightValue) =>
                (1f / rightValue, -leftValue / (rightValue * rightValue)));

    public Tensor Pow(float exponent)
    {
        if (ExecutionDevice == TensorDevice.Cuda
            && DType is TensorDType.Float32
                or TensorDType.BFloat16
                or TensorDType.Bfp8)
        {
            return ApplyUnaryCuda(CudaPublicUnaryOperation.Pow, exponent);
        }
        ThrowIfCudaHostFallback(nameof(Pow));
        float[] resultData = new float[Numel];
        for (int index = 0; index < Numel; index++)
            resultData[index] = MathF.Pow(_data[index], exponent);

        var result = new Tensor(resultData, _shape, [this]);

        result.Node.BackwardAction = () =>
        {
            for (int index = 0; index < Numel; index++)
            {
                _grad[index] += exponent
                    * MathF.Pow(_data[index], exponent - 1f)
                    * result._grad[index];
            }
        };

        return result;
    }

    public Tensor Sum()
    {
        if (ExecutionDevice == TensorDevice.Cuda
            && DType is TensorDType.Float32
                or TensorDType.BFloat16
                or TensorDType.Bfp8)
        {
            return ReduceCuda(CudaPublicReductionOperation.Sum);
        }
        ThrowIfCudaHostFallback(nameof(Sum));
        float sum = SumValues(_data, 0, Numel);

        var result = new Tensor(
            [sum],
            [1],
            [this],
            dtype: TensorDType.Float32);

        result.Node.BackwardAction = () =>
        {
            float gradient = result._grad[0];
            for (int index = 0; index < Numel; index++)
                _grad[index] += gradient;
        };

        return result;
    }

    public Tensor Mean()
    {
        if (ExecutionDevice == TensorDevice.Cuda
            && DType is TensorDType.Float32
                or TensorDType.BFloat16
                or TensorDType.Bfp8)
        {
            return ReduceCuda(CudaPublicReductionOperation.Mean);
        }
        return Sum() / Scalar(Numel);
    }

    public Tensor Max()
    {
        if (ExecutionDevice == TensorDevice.Cuda
            && DType is TensorDType.Float32
                or TensorDType.BFloat16
                or TensorDType.Bfp8)
        {
            return ReduceCuda(CudaPublicReductionOperation.Max);
        }

        ThrowIfCudaHostFallback(nameof(Max));
        float maximum = MaxValues(_data, 0, Numel);
        var result = new Tensor(
            [maximum],
            [1],
            [this],
            dtype: TensorDType.Float32);
        result.Node.BackwardAction = () =>
        {
            float gradient = result._grad[0];
            for (int index = 0; index < Numel; index++)
            {
                if (_data[index] == maximum)
                    _grad[index] += gradient;
            }
        };
        return result;
    }
}
