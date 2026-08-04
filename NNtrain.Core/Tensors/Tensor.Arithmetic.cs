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
        float sum = SumValues(_data, 0, Numel);

        var result = new Tensor([sum], [1], [this]);

        result.Node.BackwardAction = () =>
        {
            float gradient = result._grad[0];
            for (int index = 0; index < Numel; index++)
                _grad[index] += gradient;
        };

        return result;
    }

    public Tensor Mean()
        => Sum() / Scalar(Numel);
}
