using System.Runtime.Intrinsics;
using System.Threading;

namespace NNtrain;

partial class Tensor
{
    private static int _simdEnabled = 1;

    /// <summary>
    /// Gets or sets whether tensor operations may use hardware-accelerated
    /// 256-bit SIMD.
    /// </summary>
    /// <remarks>
    /// SIMD is enabled by default. When disabled, or when the current runtime
    /// does not provide hardware acceleration, tensor operations use their
    /// scalar implementation.
    /// </remarks>
    public static bool SimdEnabled
    {
        get => Volatile.Read(ref _simdEnabled) != 0;
        set => Volatile.Write(ref _simdEnabled, value ? 1 : 0);
    }

    public static bool IsSimdHardwareAccelerated
        => Vector256.IsHardwareAccelerated;

    private static bool CanUseSimd(int length)
        => SimdEnabled
            && Vector256.IsHardwareAccelerated
            && length >= Vector256<float>.Count;

    private static Vector256<float> LoadVector256(
        float[] values,
        int offset)
        => Vector256.LoadUnsafe(ref values[offset]);

    private static void StoreVector256(
        Vector256<float> vector,
        float[] destination,
        int offset)
        => vector.StoreUnsafe(ref destination[offset]);

    private static void AddScaledValues(
        float[] destination,
        int destinationOffset,
        float[] source,
        int sourceOffset,
        float scale,
        int length)
    {
        int index = 0;
        if (CanUseSimd(length))
        {
            int vectorWidth = Vector256<float>.Count;
            int vectorizedLength = length - length % vectorWidth;
            Vector256<float> scaleVector = Vector256.Create(scale);

            for (; index < vectorizedLength; index += vectorWidth)
            {
                Vector256<float> destinationVector = LoadVector256(
                    destination,
                    destinationOffset + index);
                Vector256<float> sourceVector = LoadVector256(
                    source,
                    sourceOffset + index);
                StoreVector256(
                    destinationVector + sourceVector * scaleVector,
                    destination,
                    destinationOffset + index);
            }
        }

        for (; index < length; index++)
        {
            destination[destinationOffset + index] +=
                source[sourceOffset + index] * scale;
        }
    }

    private static void MultiplyValues(
        float[] source,
        int sourceOffset,
        float scale,
        float[] destination,
        int destinationOffset,
        int length)
    {
        int index = 0;
        if (CanUseSimd(length))
        {
            int vectorWidth = Vector256<float>.Count;
            int vectorizedLength = length - length % vectorWidth;
            Vector256<float> scaleVector = Vector256.Create(scale);

            for (; index < vectorizedLength; index += vectorWidth)
            {
                Vector256<float> sourceVector = LoadVector256(
                    source,
                    sourceOffset + index);
                StoreVector256(
                    sourceVector * scaleVector,
                    destination,
                    destinationOffset + index);
            }
        }

        for (; index < length; index++)
        {
            destination[destinationOffset + index] =
                source[sourceOffset + index] * scale;
        }
    }

    private static void AddProductValues(
        float[] destination,
        int destinationOffset,
        float[] left,
        int leftOffset,
        float[] right,
        int rightOffset,
        float scale,
        int length)
    {
        int index = 0;
        if (CanUseSimd(length))
        {
            int vectorWidth = Vector256<float>.Count;
            int vectorizedLength = length - length % vectorWidth;
            Vector256<float> scaleVector = Vector256.Create(scale);

            for (; index < vectorizedLength; index += vectorWidth)
            {
                Vector256<float> destinationVector = LoadVector256(
                    destination,
                    destinationOffset + index);
                Vector256<float> leftVector = LoadVector256(
                    left,
                    leftOffset + index);
                Vector256<float> rightVector = LoadVector256(
                    right,
                    rightOffset + index);
                StoreVector256(
                    destinationVector
                        + leftVector * rightVector * scaleVector,
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
        float[] values,
        int offset,
        int length)
    {
        int index = 0;
        float sum = 0f;

        if (CanUseSimd(length))
        {
            int vectorWidth = Vector256<float>.Count;
            int vectorizedLength = length - length % vectorWidth;
            Vector256<float> sumVector = Vector256<float>.Zero;

            for (; index < vectorizedLength; index += vectorWidth)
            {
                sumVector += LoadVector256(values, offset + index);
            }

            sum += Vector256.Sum(sumVector);
        }

        for (; index < length; index++)
            sum += values[offset + index];

        return sum;
    }

    private static float SumSquaredDifferences(
        float[] values,
        int offset,
        int length,
        float mean)
    {
        int index = 0;
        float sum = 0f;

        if (CanUseSimd(length))
        {
            int vectorWidth = Vector256<float>.Count;
            int vectorizedLength = length - length % vectorWidth;
            Vector256<float> meanVector = Vector256.Create(mean);
            Vector256<float> sumVector = Vector256<float>.Zero;

            for (; index < vectorizedLength; index += vectorWidth)
            {
                Vector256<float> difference =
                    LoadVector256(values, offset + index) - meanVector;
                sumVector += difference * difference;
            }

            sum += Vector256.Sum(sumVector);
        }

        for (; index < length; index++)
        {
            float difference = values[offset + index] - mean;
            sum += difference * difference;
        }

        return sum;
    }

    private static float SumAddedValues(
        float[] left,
        float[] right,
        int offset,
        int length)
    {
        int index = 0;
        float sum = 0f;
        if (CanUseSimd(length))
        {
            int vectorWidth = Vector256<float>.Count;
            int vectorizedLength = length - length % vectorWidth;
            Vector256<float> sumVector = Vector256<float>.Zero;
            for (; index < vectorizedLength; index += vectorWidth)
            {
                sumVector += LoadVector256(left, offset + index)
                    + LoadVector256(right, offset + index);
            }

            sum += Vector256.Sum(sumVector);
        }

        for (; index < length; index++)
            sum += left[offset + index] + right[offset + index];
        return sum;
    }

    private static float SumSquaredAddedDifferences(
        float[] left,
        float[] right,
        int offset,
        int length,
        float mean)
    {
        int index = 0;
        float sum = 0f;
        if (CanUseSimd(length))
        {
            int vectorWidth = Vector256<float>.Count;
            int vectorizedLength = length - length % vectorWidth;
            Vector256<float> meanVector = Vector256.Create(mean);
            Vector256<float> sumVector = Vector256<float>.Zero;
            for (; index < vectorizedLength; index += vectorWidth)
            {
                Vector256<float> difference =
                    LoadVector256(left, offset + index)
                    + LoadVector256(right, offset + index)
                    - meanVector;
                sumVector += difference * difference;
            }

            sum += Vector256.Sum(sumVector);
        }

        for (; index < length; index++)
        {
            float difference =
                left[offset + index] + right[offset + index] - mean;
            sum += difference * difference;
        }

        return sum;
    }

    private static void NormalizeAddedAffineValues(
        float[] left,
        float[] right,
        int offset,
        float[] gamma,
        float[] beta,
        float mean,
        float inverseStandardDeviation,
        float[] normalized,
        float[] output,
        int length)
    {
        int index = 0;
        if (CanUseSimd(length))
        {
            int vectorWidth = Vector256<float>.Count;
            int vectorizedLength = length - length % vectorWidth;
            Vector256<float> meanVector = Vector256.Create(mean);
            Vector256<float> inverseVector =
                Vector256.Create(inverseStandardDeviation);
            for (; index < vectorizedLength; index += vectorWidth)
            {
                Vector256<float> normalizedVector =
                    (LoadVector256(left, offset + index)
                        + LoadVector256(right, offset + index)
                        - meanVector)
                    * inverseVector;
                StoreVector256(
                    normalizedVector,
                    normalized,
                    offset + index);
                StoreVector256(
                    normalizedVector * LoadVector256(gamma, index)
                        + LoadVector256(beta, index),
                    output,
                    offset + index);
            }
        }

        for (; index < length; index++)
        {
            float normalizedValue =
                (left[offset + index] + right[offset + index] - mean)
                * inverseStandardDeviation;
            normalized[offset + index] = normalizedValue;
            output[offset + index] =
                normalizedValue * gamma[index] + beta[index];
        }
    }

    private static void AccumulateSoftmaxGradient(
        float[] destination,
        int destinationOffset,
        float[] probability,
        int probabilityOffset,
        float[] gradient,
        int gradientOffset,
        int length,
        float dot)
    {
        int index = 0;
        if (CanUseSimd(length))
        {
            int vectorWidth = Vector256<float>.Count;
            int vectorizedLength = length - length % vectorWidth;
            Vector256<float> dotVector = Vector256.Create(dot);

            for (; index < vectorizedLength; index += vectorWidth)
            {
                Vector256<float> destinationVector = LoadVector256(
                    destination,
                    destinationOffset + index);
                Vector256<float> probabilityVector = LoadVector256(
                    probability,
                    probabilityOffset + index);
                Vector256<float> gradientVector = LoadVector256(
                    gradient,
                    gradientOffset + index);
                StoreVector256(
                    destinationVector
                        + probabilityVector
                            * (gradientVector - dotVector),
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

    private static void AccumulateLogSoftmaxGradient(
        float[] destination,
        int destinationOffset,
        float[] softmax,
        int softmaxOffset,
        float[] gradient,
        int gradientOffset,
        int length,
        float gradientSum)
    {
        int index = 0;
        if (CanUseSimd(length))
        {
            int vectorWidth = Vector256<float>.Count;
            int vectorizedLength = length - length % vectorWidth;
            Vector256<float> gradientSumVector =
                Vector256.Create(gradientSum);

            for (; index < vectorizedLength; index += vectorWidth)
            {
                Vector256<float> destinationVector = LoadVector256(
                    destination,
                    destinationOffset + index);
                Vector256<float> softmaxVector = LoadVector256(
                    softmax,
                    softmaxOffset + index);
                Vector256<float> gradientVector = LoadVector256(
                    gradient,
                    gradientOffset + index);
                StoreVector256(
                    destinationVector
                        + gradientVector
                        - softmaxVector * gradientSumVector,
                    destination,
                    destinationOffset + index);
            }
        }

        for (; index < length; index++)
        {
            destination[destinationOffset + index] +=
                gradient[gradientOffset + index]
                - softmax[softmaxOffset + index] * gradientSum;
        }
    }

    private static void NormalizeAffineValues(
        float[] input,
        int inputOffset,
        float[] gamma,
        float[] beta,
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
            int vectorWidth = Vector256<float>.Count;
            int vectorizedLength = length - length % vectorWidth;
            Vector256<float> meanVector = Vector256.Create(mean);
            Vector256<float> inverseVector =
                Vector256.Create(inverseStandardDeviation);

            for (; index < vectorizedLength; index += vectorWidth)
            {
                Vector256<float> normalizedVector =
                    (LoadVector256(input, inputOffset + index)
                        - meanVector)
                    * inverseVector;
                StoreVector256(
                    normalizedVector,
                    normalized,
                    normalizedOffset + index);
                StoreVector256(
                    normalizedVector
                        * LoadVector256(gamma, index)
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
        float[] gamma,
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
            int vectorWidth = Vector256<float>.Count;
            int vectorizedLength = length - length % vectorWidth;
            Vector256<float> firstSumVector = Vector256<float>.Zero;
            Vector256<float> secondSumVector = Vector256<float>.Zero;

            for (; index < vectorizedLength; index += vectorWidth)
            {
                Vector256<float> gradientVector = LoadVector256(
                    gradient,
                    gradientOffset + index);
                Vector256<float> gammaVector = LoadVector256(gamma, index);
                Vector256<float> normalizedVector = LoadVector256(
                    normalized,
                    normalizedOffset + index);
                Vector256<float> gradientToNormalized =
                    gradientVector * gammaVector;

                StoreVector256(
                    LoadVector256(betaGradient, index) + gradientVector,
                    betaGradient,
                    index);
                StoreVector256(
                    LoadVector256(gammaGradient, index)
                        + gradientVector * normalizedVector,
                    gammaGradient,
                    index);
                firstSumVector += gradientToNormalized;
                secondSumVector +=
                    gradientToNormalized * normalizedVector;
            }

            firstSum += Vector256.Sum(firstSumVector);
            secondSum += Vector256.Sum(secondSumVector);
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
        float[] gamma,
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
            int vectorWidth = Vector256<float>.Count;
            int vectorizedLength = length - length % vectorWidth;
            Vector256<float> firstSumVector = Vector256<float>.Zero;
            Vector256<float> secondSumVector = Vector256<float>.Zero;

            for (; index < vectorizedLength; index += vectorWidth)
            {
                Vector256<float> gradientToNormalized =
                    LoadVector256(gradient, gradientOffset + index)
                    * LoadVector256(gamma, index);
                Vector256<float> normalizedVector = LoadVector256(
                    normalized,
                    normalizedOffset + index);
                firstSumVector += gradientToNormalized;
                secondSumVector +=
                    gradientToNormalized * normalizedVector;
            }

            firstSum += Vector256.Sum(firstSumVector);
            secondSum += Vector256.Sum(secondSumVector);
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
        float[] gamma,
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
            int vectorWidth = Vector256<float>.Count;
            int vectorizedLength = length - length % vectorWidth;
            Vector256<float> lengthVector = Vector256.Create((float)length);
            Vector256<float> firstSumVector =
                Vector256.Create(sumGradientToNormalized);
            Vector256<float> secondSumVector =
                Vector256.Create(sumGradientToNormalizedTimesNormalized);
            Vector256<float> factorVector = Vector256.Create(factor);

            for (; index < vectorizedLength; index += vectorWidth)
            {
                Vector256<float> destinationVector = LoadVector256(
                    destination,
                    destinationOffset + index);
                Vector256<float> gradientToNormalized =
                    LoadVector256(gradient, gradientOffset + index)
                    * LoadVector256(gamma, index);
                Vector256<float> normalizedVector = LoadVector256(
                    normalized,
                    normalizedOffset + index);
                StoreVector256(
                    destinationVector
                        + factorVector
                            * (lengthVector * gradientToNormalized
                                - firstSumVector
                                - normalizedVector * secondSumVector),
                    destination,
                    destinationOffset + index);
            }
        }

        for (; index < length; index++)
        {
            float gradientToNormalized =
                gradient[gradientOffset + index] * gamma[index];
            destination[destinationOffset + index] +=
                factor
                * (length * gradientToNormalized
                    - sumGradientToNormalized
                    - normalized[normalizedOffset + index]
                        * sumGradientToNormalizedTimesNormalized);
        }
    }

    private static void AccumulateLayerNormInputGradientPair(
        float[] firstDestination,
        float[] secondDestination,
        int destinationOffset,
        float[] gradient,
        int gradientOffset,
        float[] gamma,
        float[] normalized,
        int normalizedOffset,
        int length,
        float inverseStandardDeviation,
        float sumGradientToNormalized,
        float sumGradientToNormalizedTimesNormalized)
    {
        int index = 0;
        float factor = inverseStandardDeviation / length;
        bool sameDestination = ReferenceEquals(
            firstDestination,
            secondDestination);

        if (CanUseSimd(length))
        {
            int vectorWidth = Vector256<float>.Count;
            int vectorizedLength = length - length % vectorWidth;
            Vector256<float> lengthVector = Vector256.Create((float)length);
            Vector256<float> firstSumVector =
                Vector256.Create(sumGradientToNormalized);
            Vector256<float> secondSumVector =
                Vector256.Create(sumGradientToNormalizedTimesNormalized);
            Vector256<float> factorVector = Vector256.Create(factor);
            Vector256<float> destinationScale =
                Vector256.Create(sameDestination ? 2f : 1f);

            for (; index < vectorizedLength; index += vectorWidth)
            {
                Vector256<float> gradientToNormalized =
                    LoadVector256(gradient, gradientOffset + index)
                    * LoadVector256(gamma, index);
                Vector256<float> contribution = factorVector
                    * (lengthVector * gradientToNormalized
                        - firstSumVector
                        - LoadVector256(normalized, normalizedOffset + index)
                            * secondSumVector);
                StoreVector256(
                    LoadVector256(
                        firstDestination,
                        destinationOffset + index)
                        + contribution * destinationScale,
                    firstDestination,
                    destinationOffset + index);
                if (!sameDestination)
                {
                    StoreVector256(
                        LoadVector256(
                            secondDestination,
                            destinationOffset + index)
                            + contribution,
                        secondDestination,
                        destinationOffset + index);
                }
            }
        }

        for (; index < length; index++)
        {
            float gradientToNormalized =
                gradient[gradientOffset + index] * gamma[index];
            float contribution = factor
                * (length * gradientToNormalized
                    - sumGradientToNormalized
                    - normalized[normalizedOffset + index]
                        * sumGradientToNormalizedTimesNormalized);
            firstDestination[destinationOffset + index] +=
                sameDestination ? 2f * contribution : contribution;
            if (!sameDestination)
            {
                secondDestination[destinationOffset + index] +=
                    contribution;
            }
        }
    }
}
