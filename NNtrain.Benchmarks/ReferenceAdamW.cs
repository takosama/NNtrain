using System.Runtime.Intrinsics;
using NNtrain;

namespace NNtrain.Benchmarks;

/// <summary>
/// Frozen copy of the pre-optimization AdamW hot path, used only to obtain a
/// same-process benchmark ratio against the JSON model shape.
/// </summary>
internal sealed class ReferenceAdamW
{
    private readonly Parameter[] _parameters;
    private readonly AdamWOptions _options;
    private readonly float[][] _firstMoments;
    private readonly float[][] _secondMoments;
    private readonly long _totalElements;
    private int _step;

    internal ReferenceAdamW(
        Parameter[] parameters,
        AdamWOptions options)
    {
        _parameters = parameters;
        _options = options;
        _firstMoments = parameters
            .Select(parameter => new float[parameter.T.Numel])
            .ToArray();
        _secondMoments = parameters
            .Select(parameter => new float[parameter.T.Numel])
            .ToArray();
        _totalElements = parameters.Sum(
            parameter => (long)parameter.T.Numel);
    }

    internal void ZeroGrad()
    {
        if (_parameters.Length > 1 && _totalElements >= 32_768)
        {
            Tensor.RunParallel(
                0,
                _parameters.Length,
                index => _parameters[index].ZeroGrad());
            return;
        }
        foreach (Parameter parameter in _parameters)
            parameter.ZeroGrad();
    }

    internal void Step()
    {
        _step++;
        float bc1 = 1f - MathF.Pow(_options.Beta1, _step);
        float bc2 = 1f - MathF.Pow(_options.Beta2, _step);

        void UpdateParameter(int parameterIndex)
        {
            Parameter parameter = _parameters[parameterIndex];
            using Tensor.DataMutation mutation = parameter.BeginUpdate();
            Span<float> data = mutation.Values;
            float[] gradient = parameter.T.GradientBuffer;
            float[] firstMoment = _firstMoments[parameterIndex];
            float[] secondMoment = _secondMoments[parameterIndex];
            bool applyWeightDecay =
                parameter.WeightDecay == WeightDecayPolicy.Apply
                || (_options.Decay1D && parameter.T.Rank == 1);
            int index = 0;

            if (Tensor.SimdEnabled
                && Vector256.IsHardwareAccelerated
                && data.Length >= Vector256<float>.Count)
            {
                int width = Vector256<float>.Count;
                int end = data.Length - data.Length % width;
                Vector256<float> beta1 = Vector256.Create(_options.Beta1);
                Vector256<float> beta2 = Vector256.Create(_options.Beta2);
                Vector256<float> oneMinusBeta1 =
                    Vector256.Create(1f - _options.Beta1);
                Vector256<float> oneMinusBeta2 =
                    Vector256.Create(1f - _options.Beta2);
                Vector256<float> inverseBc1 = Vector256.Create(1f / bc1);
                Vector256<float> inverseBc2 = Vector256.Create(1f / bc2);
                Vector256<float> learningRate =
                    Vector256.Create(_options.LearningRate);
                Vector256<float> epsilon = Vector256.Create(_options.Epsilon);
                Vector256<float> decay = Vector256.Create(
                    _options.LearningRate * _options.WeightDecay);

                for (; index < end; index += width)
                {
                    Vector256<float> g = Vector256.LoadUnsafe(
                        ref gradient[index]);
                    Vector256<float> m = beta1 * Vector256.LoadUnsafe(
                        ref firstMoment[index]) + oneMinusBeta1 * g;
                    Vector256<float> v = beta2 * Vector256.LoadUnsafe(
                        ref secondMoment[index]) + oneMinusBeta2 * g * g;
                    m.StoreUnsafe(ref firstMoment[index]);
                    v.StoreUnsafe(ref secondMoment[index]);
                    Vector256<float> p = Vector256.LoadUnsafe(ref data[index]);
                    if (applyWeightDecay)
                        p -= decay * p;
                    p -= learningRate * (m * inverseBc1)
                        / (Vector256.Sqrt(v * inverseBc2) + epsilon);
                    p.StoreUnsafe(ref data[index]);
                }
            }

            for (; index < data.Length; index++)
            {
                float g = gradient[index];
                firstMoment[index] = _options.Beta1 * firstMoment[index]
                    + (1f - _options.Beta1) * g;
                secondMoment[index] = _options.Beta2 * secondMoment[index]
                    + (1f - _options.Beta2) * g * g;
                if (applyWeightDecay)
                {
                    data[index] -= _options.LearningRate
                        * _options.WeightDecay
                        * data[index];
                }
                data[index] -= _options.LearningRate
                    * (firstMoment[index] / bc1)
                    / (MathF.Sqrt(secondMoment[index] / bc2)
                        + _options.Epsilon);
            }
        }

        if (_parameters.Length > 1 && _totalElements >= 32_768)
            Tensor.RunParallel(0, _parameters.Length, UpdateParameter);
        else
            for (int index = 0; index < _parameters.Length; index++)
                UpdateParameter(index);
    }
}
