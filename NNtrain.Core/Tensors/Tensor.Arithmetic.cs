namespace NNtrain;

partial class Tensor
{
    public static Tensor operator +(Tensor left, Tensor right)
        => ApplyBinaryElementwise(
            left,
            right,
            static (leftValue, rightValue) => leftValue + rightValue,
            static (_, _) => (1f, 1f));

    public static Tensor operator -(Tensor left, Tensor right)
        => ApplyBinaryElementwise(
            left,
            right,
            static (leftValue, rightValue) => leftValue - rightValue,
            static (_, _) => (1f, -1f));

    public static Tensor operator -(Tensor value)
    {
        ArgumentNullException.ThrowIfNull(value);

        float[] resultData = new float[value.Numel];
        for (int index = 0; index < value.Numel; index++)
            resultData[index] = -value._data[index];

        var result = new Tensor(resultData, value._shape, [value]);

        result.Node.BackwardAction = () =>
        {
            for (int index = 0; index < value.Numel; index++)
                value._grad[index] -= result._grad[index];
        };

        return result;
    }

    public static Tensor operator *(Tensor left, Tensor right)
        => ApplyBinaryElementwise(
            left,
            right,
            static (leftValue, rightValue) => leftValue * rightValue,
            static (leftValue, rightValue) => (rightValue, leftValue));

    public static Tensor operator /(Tensor left, Tensor right)
        => ApplyBinaryElementwise(
            left,
            right,
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
        float sum = 0f;
        for (int index = 0; index < Numel; index++)
            sum += _data[index];

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
