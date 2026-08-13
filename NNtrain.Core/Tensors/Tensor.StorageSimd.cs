using System.Runtime.Intrinsics;

namespace NNtrain;

/// <summary>
/// Storage-aware SIMD specializations. Values are decoded at vector-load
/// boundaries and all arithmetic and accumulation remains Float32.
/// </summary>
partial class Tensor
{
    private static float MaxValues(
        TensorStorage values,
        int offset,
        int length)
    {
        if (length <= 0)
            throw new ArgumentOutOfRangeException(nameof(length));

        int index = 1;
        float maximum = values[offset];
        if (CanUseSimd(length))
        {
            int width = Vector256<float>.Count;
            int end = length - length % width;
            Vector256<float> maximumVector = LoadVector256(values, offset);
            index = width;
            for (; index < end; index += width)
            {
                maximumVector = Vector256.Max(
                    maximumVector,
                    LoadVector256(values, offset + index));
            }

            for (int lane = 0; lane < width; lane++)
                maximum = MathF.Max(maximum, maximumVector.GetElement(lane));
        }

        for (; index < length; index++)
            maximum = MathF.Max(maximum, values[offset + index]);
        return maximum;
    }

    private static float ExpShiftedValues(
        TensorStorage source,
        int sourceOffset,
        float shift,
        float[] destination,
        int destinationOffset,
        int length)
    {
        int index = 0;
        float sum = 0f;
        if (CanUseSimd(length))
        {
            int width = Vector256<float>.Count;
            int end = length - length % width;
            Vector256<float> shiftVector = Vector256.Create(shift);
            Vector256<float> minimum = Vector256.Create(-87.33654f);
            Vector256<float> sumVector = Vector256<float>.Zero;
            for (; index < end; index += width)
            {
                Vector256<float> result = ExpVector(Vector256.Max(
                    LoadVector256(source, sourceOffset + index) - shiftVector,
                    minimum));
                StoreVector256(result, destination, destinationOffset + index);
                sumVector += result;
            }
            sum = Vector256.Sum(sumVector);
        }

        for (; index < length; index++)
        {
            float value = MathF.Exp(source[sourceOffset + index] - shift);
            destination[destinationOffset + index] = value;
            sum += value;
        }
        return sum;
    }

    private static float SumExpShiftedValues(
        TensorStorage source,
        int sourceOffset,
        float shift,
        int length)
    {
        int index = 0;
        float sum = 0f;
        if (CanUseSimd(length))
        {
            int width = Vector256<float>.Count;
            int end = length - length % width;
            Vector256<float> shiftVector = Vector256.Create(shift);
            Vector256<float> minimum = Vector256.Create(-87.33654f);
            Vector256<float> sumVector = Vector256<float>.Zero;
            for (; index < end; index += width)
            {
                sumVector += ExpVector(Vector256.Max(
                    LoadVector256(source, sourceOffset + index) - shiftVector,
                    minimum));
            }
            sum = Vector256.Sum(sumVector);
        }

        for (; index < length; index++)
            sum += MathF.Exp(source[sourceOffset + index] - shift);
        return sum;
    }

    private static void AccumulateNormalizedExpGradient(
        float[] destination,
        int destinationOffset,
        TensorStorage logits,
        int logitsOffset,
        float maximum,
        float probabilityScale,
        float uniformTarget,
        int length)
    {
        int index = 0;
        if (CanUseSimd(length))
        {
            int width = Vector256<float>.Count;
            int end = length - length % width;
            Vector256<float> maximumVector = Vector256.Create(maximum);
            Vector256<float> minimum = Vector256.Create(-87.33654f);
            Vector256<float> probabilityScaleVector =
                Vector256.Create(probabilityScale);
            Vector256<float> uniformTargetVector =
                Vector256.Create(uniformTarget);
            for (; index < end; index += width)
            {
                Vector256<float> probability = ExpVector(Vector256.Max(
                    LoadVector256(logits, logitsOffset + index)
                        - maximumVector,
                    minimum)) * probabilityScaleVector;
                StoreVector256(
                    LoadVector256(destination, destinationOffset + index)
                        + probability - uniformTargetVector,
                    destination,
                    destinationOffset + index);
            }
        }

        for (; index < length; index++)
        {
            destination[destinationOffset + index] +=
                MathF.Exp(logits[logitsOffset + index] - maximum)
                    * probabilityScale
                - uniformTarget;
        }
    }

    private static void SubtractShiftAndScalarValues(
        TensorStorage source,
        int sourceOffset,
        float shift,
        float scalar,
        float[] destination,
        int destinationOffset,
        int length)
    {
        int index = 0;
        if (CanUseSimd(length))
        {
            int width = Vector256<float>.Count;
            int end = length - length % width;
            Vector256<float> shiftVector = Vector256.Create(shift);
            Vector256<float> scalarVector = Vector256.Create(scalar);
            for (; index < end; index += width)
            {
                StoreVector256(
                    LoadVector256(source, sourceOffset + index)
                        - shiftVector - scalarVector,
                    destination,
                    destinationOffset + index);
            }
        }

        if (CanUseVector128(length - index))
        {
            StoreVector128(
                LoadVector128(source, sourceOffset + index)
                    - Vector128.Create(shift)
                    - Vector128.Create(scalar),
                destination,
                destinationOffset + index);
            index += Vector128<float>.Count;
        }

        for (; index < length; index++)
        {
            destination[destinationOffset + index] =
                source[sourceOffset + index] - shift - scalar;
        }
    }

    private static void AddProductValues(
        float[] destination,
        int destinationOffset,
        TensorStorage left,
        int leftOffset,
        float[] right,
        int rightOffset,
        float scale,
        int length)
    {
        int index = 0;
        if (CanUseSimd(length))
        {
            int width = Vector256<float>.Count;
            int end = length - length % width;
            Vector256<float> scaleVector = Vector256.Create(scale);
            for (; index < end; index += width)
            {
                StoreVector256(
                    LoadVector256(destination, destinationOffset + index)
                        + LoadVector256(left, leftOffset + index)
                            * LoadVector256(right, rightOffset + index)
                            * scaleVector,
                    destination,
                    destinationOffset + index);
            }
        }

        for (; index < length; index++)
        {
            destination[destinationOffset + index] +=
                left[leftOffset + index]
                * right[rightOffset + index]
                * scale;
        }
    }

    private static float SumValues(
        TensorStorage values,
        int offset,
        int length)
    {
        int index = 0;
        float sum = 0f;
        if (CanUseSimd(length))
        {
            int width = Vector256<float>.Count;
            int end = length - length % width;
            Vector256<float> sumVector = Vector256<float>.Zero;
            for (; index < end; index += width)
                sumVector += LoadVector256(values, offset + index);
            sum = Vector256.Sum(sumVector);
        }

        if (CanUseVector128(length - index))
        {
            sum += Vector128.Sum(LoadVector128(values, offset + index));
            index += Vector128<float>.Count;
        }

        for (; index < length; index++)
            sum += values[offset + index];
        return sum;
    }

    private static float SumSquaredDifferences(
        TensorStorage values,
        int offset,
        int length,
        float mean)
    {
        int index = 0;
        float sum = 0f;
        if (CanUseSimd(length))
        {
            int width = Vector256<float>.Count;
            int end = length - length % width;
            Vector256<float> meanVector = Vector256.Create(mean);
            Vector256<float> sumVector = Vector256<float>.Zero;
            for (; index < end; index += width)
            {
                Vector256<float> difference =
                    LoadVector256(values, offset + index) - meanVector;
                sumVector += difference * difference;
            }
            sum = Vector256.Sum(sumVector);
        }

        if (CanUseVector128(length - index))
        {
            Vector128<float> difference =
                LoadVector128(values, offset + index)
                - Vector128.Create(mean);
            sum += Vector128.Sum(difference * difference);
            index += Vector128<float>.Count;
        }

        for (; index < length; index++)
        {
            float difference = values[offset + index] - mean;
            sum += difference * difference;
        }
        return sum;
    }

    private static void AccumulateSoftmaxGradient(
        float[] destination,
        int destinationOffset,
        TensorStorage probability,
        int probabilityOffset,
        float[] gradient,
        int gradientOffset,
        int length,
        float dot)
    {
        int index = 0;
        if (CanUseSimd(length))
        {
            int width = Vector256<float>.Count;
            int end = length - length % width;
            Vector256<float> dotVector = Vector256.Create(dot);
            for (; index < end; index += width)
            {
                StoreVector256(
                    LoadVector256(destination, destinationOffset + index)
                        + LoadVector256(probability, probabilityOffset + index)
                            * (LoadVector256(gradient, gradientOffset + index)
                                - dotVector),
                    destination,
                    destinationOffset + index);
            }
        }

        for (; index < length; index++)
        {
            destination[destinationOffset + index] +=
                probability[probabilityOffset + index]
                * (gradient[gradientOffset + index] - dot);
        }
    }

    private static void NormalizeAffineValues(
        TensorStorage input,
        int inputOffset,
        TensorStorage gamma,
        TensorStorage beta,
        float mean,
        float inverseStandardDeviation,
        float[] normalized,
        int normalizedOffset,
        float[] output,
        int outputOffset,
        int length)
    {
        int index = 0;
        if (CanUseSimd(length))
        {
            int width = Vector256<float>.Count;
            int end = length - length % width;
            Vector256<float> meanVector = Vector256.Create(mean);
            Vector256<float> inverseVector =
                Vector256.Create(inverseStandardDeviation);
            for (; index < end; index += width)
            {
                Vector256<float> normalizedVector =
                    (LoadVector256(input, inputOffset + index) - meanVector)
                    * inverseVector;
                StoreVector256(
                    normalizedVector,
                    normalized,
                    normalizedOffset + index);
                StoreVector256(
                    normalizedVector * LoadVector256(gamma, index)
                        + LoadVector256(beta, index),
                    output,
                    outputOffset + index);
            }
        }

        for (; index < length; index++)
        {
            float normalizedValue =
                (input[inputOffset + index] - mean)
                * inverseStandardDeviation;
            normalized[normalizedOffset + index] = normalizedValue;
            output[outputOffset + index] =
                normalizedValue * gamma[index] + beta[index];
        }
    }

    private static void AccumulateLayerNormParameterGradients(
        float[] gradient,
        int gradientOffset,
        TensorStorage gamma,
        float[] normalized,
        int normalizedOffset,
        float[] gammaGradient,
        float[] betaGradient,
        int length,
        out float sumGradientToNormalized,
        out float sumGradientToNormalizedTimesNormalized)
    {
        int index = 0;
        float firstSum = 0f;
        float secondSum = 0f;
        if (CanUseSimd(length))
        {
            int width = Vector256<float>.Count;
            int end = length - length % width;
            Vector256<float> firstVector = Vector256<float>.Zero;
            Vector256<float> secondVector = Vector256<float>.Zero;
            for (; index < end; index += width)
            {
                Vector256<float> gradientVector =
                    LoadVector256(gradient, gradientOffset + index);
                Vector256<float> normalizedVector =
                    LoadVector256(normalized, normalizedOffset + index);
                Vector256<float> gradientToNormalized =
                    gradientVector * LoadVector256(gamma, index);
                StoreVector256(
                    LoadVector256(betaGradient, index) + gradientVector,
                    betaGradient,
                    index);
                StoreVector256(
                    LoadVector256(gammaGradient, index)
                        + gradientVector * normalizedVector,
                    gammaGradient,
                    index);
                firstVector += gradientToNormalized;
                secondVector += gradientToNormalized * normalizedVector;
            }
            firstSum = Vector256.Sum(firstVector);
            secondSum = Vector256.Sum(secondVector);
        }

        for (; index < length; index++)
        {
            float currentGradient = gradient[gradientOffset + index];
            float normalizedValue = normalized[normalizedOffset + index];
            betaGradient[index] += currentGradient;
            gammaGradient[index] += currentGradient * normalizedValue;
            float gradientToNormalized = currentGradient * gamma[index];
            firstSum += gradientToNormalized;
            secondSum += gradientToNormalized * normalizedValue;
        }
        sumGradientToNormalized = firstSum;
        sumGradientToNormalizedTimesNormalized = secondSum;
    }

    private static void ComputeLayerNormGradientSums(
        float[] gradient,
        int gradientOffset,
        TensorStorage gamma,
        float[] normalized,
        int normalizedOffset,
        int length,
        out float sumGradientToNormalized,
        out float sumGradientToNormalizedTimesNormalized)
    {
        int index = 0;
        float firstSum = 0f;
        float secondSum = 0f;
        if (CanUseSimd(length))
        {
            int width = Vector256<float>.Count;
            int end = length - length % width;
            Vector256<float> firstVector = Vector256<float>.Zero;
            Vector256<float> secondVector = Vector256<float>.Zero;
            for (; index < end; index += width)
            {
                Vector256<float> gradientToNormalized =
                    LoadVector256(gradient, gradientOffset + index)
                    * LoadVector256(gamma, index);
                Vector256<float> normalizedVector =
                    LoadVector256(normalized, normalizedOffset + index);
                firstVector += gradientToNormalized;
                secondVector += gradientToNormalized * normalizedVector;
            }
            firstSum = Vector256.Sum(firstVector);
            secondSum = Vector256.Sum(secondVector);
        }

        for (; index < length; index++)
        {
            float gradientToNormalized =
                gradient[gradientOffset + index] * gamma[index];
            firstSum += gradientToNormalized;
            secondSum += gradientToNormalized
                * normalized[normalizedOffset + index];
        }
        sumGradientToNormalized = firstSum;
        sumGradientToNormalizedTimesNormalized = secondSum;
    }

    private static void AccumulateLayerNormInputGradient(
        float[] destination,
        int destinationOffset,
        float[] gradient,
        int gradientOffset,
        TensorStorage gamma,
        float[] normalized,
        int normalizedOffset,
        int length,
        float inverseStandardDeviation,
        float sumGradientToNormalized,
        float sumGradientToNormalizedTimesNormalized)
    {
        int index = 0;
        float factor = inverseStandardDeviation / length;
        if (CanUseSimd(length))
        {
            int width = Vector256<float>.Count;
            int end = length - length % width;
            Vector256<float> lengthVector = Vector256.Create((float)length);
            Vector256<float> firstVector =
                Vector256.Create(sumGradientToNormalized);
            Vector256<float> secondVector =
                Vector256.Create(sumGradientToNormalizedTimesNormalized);
            Vector256<float> factorVector = Vector256.Create(factor);
            for (; index < end; index += width)
            {
                Vector256<float> gradientToNormalized =
                    LoadVector256(gradient, gradientOffset + index)
                    * LoadVector256(gamma, index);
                Vector256<float> normalizedVector =
                    LoadVector256(normalized, normalizedOffset + index);
                StoreVector256(
                    LoadVector256(destination, destinationOffset + index)
                        + factorVector
                            * (lengthVector * gradientToNormalized
                                - firstVector
                                - normalizedVector * secondVector),
                    destination,
                    destinationOffset + index);
            }
        }

        for (; index < length; index++)
        {
            float gradientToNormalized =
                gradient[gradientOffset + index] * gamma[index];
            destination[destinationOffset + index] += factor
                * (length * gradientToNormalized
                    - sumGradientToNormalized
                    - normalized[normalizedOffset + index]
                        * sumGradientToNormalizedTimesNormalized);
        }
    }
}
