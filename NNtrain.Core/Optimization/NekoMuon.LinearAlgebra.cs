using System.Diagnostics;
using NNtrain.Cuda.Interop;

namespace NNtrain;

public sealed partial class NekoMuon
{
    private static void ComputeSymmetricGram(
        float[] source,
        float[] destination,
        int rows,
        int columns,
        bool bfloat16MatrixOperands = false)
    {
        if (bfloat16MatrixOperands)
        {
            for (int row = 0; row < rows; row++)
            {
                int rowOffset = row * columns;
                for (int other = 0; other <= row; other++)
                {
                    int otherOffset = other * columns;
                    float dot = 0f;
                    for (int column = 0; column < columns; column++)
                    {
                        float left = TensorStorageCodec.RoundToBFloat16(
                            source[rowOffset + column]);
                        float right = TensorStorageCodec.RoundToBFloat16(
                            source[otherOffset + column]);
                        dot = MathF.FusedMultiplyAdd(left, right, dot);
                    }
                    destination[row * rows + other] = dot;
                    destination[other * rows + row] = dot;
                }
            }
            return;
        }

        if (!Tensor.SimdEnabled
            || !Vector256.IsHardwareAccelerated
            || rows % 4 != 0
            || columns < Vector256<float>.Count)
        {
            for (int row = 0; row < rows; row++)
            {
                int rowOffset = row * columns;
                for (int other = 0; other <= row; other++)
                {
                    float dot = Dot(
                        source,
                        rowOffset,
                        other * columns,
                        columns);
                    destination[row * rows + other] = dot;
                    destination[other * rows + row] = dot;
                }
            }
            return;
        }

        // Four rows and two comparison rows share six vector loads across
        // eight dot products. This removes most repeated reads in the two
        // Gram products that dominate Newton-Schulz on wide FFN matrices.
        for (int rowBase = 0; rowBase < rows; rowBase += 4)
        {
            for (int otherBase = 0;
                otherBase <= rowBase + 2;
                otherBase += 2)
            {
                ComputeGramFourByTwo(
                    source,
                    destination,
                    rows,
                    columns,
                    rowBase,
                    otherBase);
            }
        }
    }

    private static void ComputeGramFourByTwo(
        float[] source,
        float[] destination,
        int rows,
        int columns,
        int rowBase,
        int otherBase)
    {
        int row0 = rowBase * columns;
        int row1 = row0 + columns;
        int row2 = row1 + columns;
        int row3 = row2 + columns;
        int other0 = otherBase * columns;
        int other1 = other0 + columns;
        int index = 0;
        int width = Vector256<float>.Count;
        int vectorizedLength = columns - columns % width;
        Vector256<float> sum00 = Vector256<float>.Zero;
        Vector256<float> sum01 = Vector256<float>.Zero;
        Vector256<float> sum10 = Vector256<float>.Zero;
        Vector256<float> sum11 = Vector256<float>.Zero;
        Vector256<float> sum20 = Vector256<float>.Zero;
        Vector256<float> sum21 = Vector256<float>.Zero;
        Vector256<float> sum30 = Vector256<float>.Zero;
        Vector256<float> sum31 = Vector256<float>.Zero;

        for (; index < vectorizedLength; index += width)
        {
            Vector256<float> a0 = Vector256.LoadUnsafe(ref source[row0 + index]);
            Vector256<float> a1 = Vector256.LoadUnsafe(ref source[row1 + index]);
            Vector256<float> a2 = Vector256.LoadUnsafe(ref source[row2 + index]);
            Vector256<float> a3 = Vector256.LoadUnsafe(ref source[row3 + index]);
            Vector256<float> b0 = Vector256.LoadUnsafe(
                ref source[other0 + index]);
            Vector256<float> b1 = Vector256.LoadUnsafe(
                ref source[other1 + index]);
            sum00 = Vector256.FusedMultiplyAdd(a0, b0, sum00);
            sum01 = Vector256.FusedMultiplyAdd(a0, b1, sum01);
            sum10 = Vector256.FusedMultiplyAdd(a1, b0, sum10);
            sum11 = Vector256.FusedMultiplyAdd(a1, b1, sum11);
            sum20 = Vector256.FusedMultiplyAdd(a2, b0, sum20);
            sum21 = Vector256.FusedMultiplyAdd(a2, b1, sum21);
            sum30 = Vector256.FusedMultiplyAdd(a3, b0, sum30);
            sum31 = Vector256.FusedMultiplyAdd(a3, b1, sum31);
        }

        float scalar00 = Vector256.Sum(sum00);
        float scalar01 = Vector256.Sum(sum01);
        float scalar10 = Vector256.Sum(sum10);
        float scalar11 = Vector256.Sum(sum11);
        float scalar20 = Vector256.Sum(sum20);
        float scalar21 = Vector256.Sum(sum21);
        float scalar30 = Vector256.Sum(sum30);
        float scalar31 = Vector256.Sum(sum31);
        for (; index < columns; index++)
        {
            float a0 = source[row0 + index];
            float a1 = source[row1 + index];
            float a2 = source[row2 + index];
            float a3 = source[row3 + index];
            float b0 = source[other0 + index];
            float b1 = source[other1 + index];
            scalar00 += a0 * b0;
            scalar01 += a0 * b1;
            scalar10 += a1 * b0;
            scalar11 += a1 * b1;
            scalar20 += a2 * b0;
            scalar21 += a2 * b1;
            scalar30 += a3 * b0;
            scalar31 += a3 * b1;
        }

        StoreSymmetricGram(destination, rows, rowBase, otherBase, scalar00);
        StoreSymmetricGram(destination, rows, rowBase, otherBase + 1, scalar01);
        StoreSymmetricGram(destination, rows, rowBase + 1, otherBase, scalar10);
        StoreSymmetricGram(
            destination,
            rows,
            rowBase + 1,
            otherBase + 1,
            scalar11);
        StoreSymmetricGram(destination, rows, rowBase + 2, otherBase, scalar20);
        StoreSymmetricGram(
            destination,
            rows,
            rowBase + 2,
            otherBase + 1,
            scalar21);
        StoreSymmetricGram(destination, rows, rowBase + 3, otherBase, scalar30);
        StoreSymmetricGram(
            destination,
            rows,
            rowBase + 3,
            otherBase + 1,
            scalar31);
    }

    private static void StoreSymmetricGram(
        float[] destination,
        int rows,
        int row,
        int column,
        float value)
    {
        destination[row * rows + column] = value;
        destination[column * rows + row] = value;
    }

    private static float Dot(
        float[] values,
        int firstOffset,
        int secondOffset,
        int length)
    {
        int index = 0;
        float sum = 0f;
        if (Tensor.SimdEnabled
            && Vector256.IsHardwareAccelerated
            && length >= Vector256<float>.Count)
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = length - length % width;
            int unrolledLength = vectorizedLength - 3 * width;
            Vector256<float> sum0 = Vector256<float>.Zero;
            Vector256<float> sum1 = Vector256<float>.Zero;
            Vector256<float> sum2 = Vector256<float>.Zero;
            Vector256<float> sum3 = Vector256<float>.Zero;
            for (; index < unrolledLength; index += 4 * width)
            {
                sum0 +=
                    Vector256.LoadUnsafe(ref values[firstOffset + index])
                    * Vector256.LoadUnsafe(ref values[secondOffset + index]);
                sum1 +=
                    Vector256.LoadUnsafe(
                        ref values[firstOffset + index + width])
                    * Vector256.LoadUnsafe(
                        ref values[secondOffset + index + width]);
                sum2 +=
                    Vector256.LoadUnsafe(
                        ref values[firstOffset + index + 2 * width])
                    * Vector256.LoadUnsafe(
                        ref values[secondOffset + index + 2 * width]);
                sum3 +=
                    Vector256.LoadUnsafe(
                        ref values[firstOffset + index + 3 * width])
                    * Vector256.LoadUnsafe(
                        ref values[secondOffset + index + 3 * width]);
            }

            sum = Vector256.Sum((sum0 + sum1) + (sum2 + sum3));
            for (; index < vectorizedLength; index += width)
            {
                sum += Vector256.Sum(
                    Vector256.LoadUnsafe(ref values[firstOffset + index])
                    * Vector256.LoadUnsafe(
                        ref values[secondOffset + index]));
            }
        }

        for (; index < length; index++)
        {
            sum += values[firstOffset + index]
                * values[secondOffset + index];
        }

        return sum;
    }

    private static void Scale(
        float[] source,
        float[] destination,
        float scale)
    {
        int index = 0;
        if (Tensor.SimdEnabled
            && Vector256.IsHardwareAccelerated
            && source.Length >= Vector256<float>.Count)
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = source.Length - source.Length % width;
            Vector256<float> vectorScale = Vector256.Create(scale);
            for (; index < vectorizedLength; index += width)
            {
                (Vector256.LoadUnsafe(ref source[index]) * vectorScale)
                    .StoreUnsafe(ref destination[index]);
            }
        }

        for (; index < source.Length; index++)
            destination[index] = source[index] * scale;
    }

    private static void AddScaled(
        float[] source,
        int sourceOffset,
        float[] destination,
        int destinationOffset,
        int length,
        float scale)
    {
        int index = 0;
        if (Tensor.SimdEnabled
            && Vector256.IsHardwareAccelerated
            && length >= Vector256<float>.Count)
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = length - length % width;
            Vector256<float> vectorScale = Vector256.Create(scale);
            for (; index < vectorizedLength; index += width)
            {
                Vector256<float> result =
                    Vector256.LoadUnsafe(ref destination[
                        destinationOffset + index])
                    + vectorScale
                        * Vector256.LoadUnsafe(ref source[
                            sourceOffset + index]);
                result.StoreUnsafe(ref destination[
                    destinationOffset + index]);
            }
        }

        for (; index < length; index++)
        {
            destination[destinationOffset + index] +=
                scale * source[sourceOffset + index];
        }
    }

    private static void AddScaledFourRows(
        float[] source,
        int sourceOffset,
        float[] destination,
        int destination0,
        int destination1,
        int destination2,
        int destination3,
        int length,
        float scale0,
        float scale1,
        float scale2,
        float scale3)
    {
        int index = 0;
        if (Tensor.SimdEnabled
            && Vector256.IsHardwareAccelerated
            && length >= Vector256<float>.Count)
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = length - length % width;
            Vector256<float> vectorScale0 = Vector256.Create(scale0);
            Vector256<float> vectorScale1 = Vector256.Create(scale1);
            Vector256<float> vectorScale2 = Vector256.Create(scale2);
            Vector256<float> vectorScale3 = Vector256.Create(scale3);
            for (; index < vectorizedLength; index += width)
            {
                Vector256<float> sourceVector = Vector256.LoadUnsafe(
                    ref source[sourceOffset + index]);
                (Vector256.LoadUnsafe(ref destination[destination0 + index])
                    + vectorScale0 * sourceVector)
                    .StoreUnsafe(ref destination[destination0 + index]);
                (Vector256.LoadUnsafe(ref destination[destination1 + index])
                    + vectorScale1 * sourceVector)
                    .StoreUnsafe(ref destination[destination1 + index]);
                (Vector256.LoadUnsafe(ref destination[destination2 + index])
                    + vectorScale2 * sourceVector)
                    .StoreUnsafe(ref destination[destination2 + index]);
                (Vector256.LoadUnsafe(ref destination[destination3 + index])
                    + vectorScale3 * sourceVector)
                    .StoreUnsafe(ref destination[destination3 + index]);
            }
        }

        for (; index < length; index++)
        {
            float sourceValue = source[sourceOffset + index];
            destination[destination0 + index] += scale0 * sourceValue;
            destination[destination1 + index] += scale1 * sourceValue;
            destination[destination2 + index] += scale2 * sourceValue;
            destination[destination3 + index] += scale3 * sourceValue;
        }
    }

    private static void AddScaledEightRows(
        float[] source,
        int sourceOffset,
        float[] destination,
        int destination0,
        int destination1,
        int destination2,
        int destination3,
        int destination4,
        int destination5,
        int destination6,
        int destination7,
        int length,
        float scale0,
        float scale1,
        float scale2,
        float scale3,
        float scale4,
        float scale5,
        float scale6,
        float scale7)
    {
        int index = 0;
        if (Tensor.SimdEnabled
            && Vector256.IsHardwareAccelerated
            && length >= Vector256<float>.Count)
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = length - length % width;
            Vector256<float> vectorScale0 = Vector256.Create(scale0);
            Vector256<float> vectorScale1 = Vector256.Create(scale1);
            Vector256<float> vectorScale2 = Vector256.Create(scale2);
            Vector256<float> vectorScale3 = Vector256.Create(scale3);
            Vector256<float> vectorScale4 = Vector256.Create(scale4);
            Vector256<float> vectorScale5 = Vector256.Create(scale5);
            Vector256<float> vectorScale6 = Vector256.Create(scale6);
            Vector256<float> vectorScale7 = Vector256.Create(scale7);
            for (; index < vectorizedLength; index += width)
            {
                Vector256<float> sourceVector = Vector256.LoadUnsafe(
                    ref source[sourceOffset + index]);
                Vector256.FusedMultiplyAdd(
                    sourceVector,
                    vectorScale0,
                    Vector256.LoadUnsafe(ref destination[destination0 + index]))
                    .StoreUnsafe(ref destination[destination0 + index]);
                Vector256.FusedMultiplyAdd(
                    sourceVector,
                    vectorScale1,
                    Vector256.LoadUnsafe(ref destination[destination1 + index]))
                    .StoreUnsafe(ref destination[destination1 + index]);
                Vector256.FusedMultiplyAdd(
                    sourceVector,
                    vectorScale2,
                    Vector256.LoadUnsafe(ref destination[destination2 + index]))
                    .StoreUnsafe(ref destination[destination2 + index]);
                Vector256.FusedMultiplyAdd(
                    sourceVector,
                    vectorScale3,
                    Vector256.LoadUnsafe(ref destination[destination3 + index]))
                    .StoreUnsafe(ref destination[destination3 + index]);
                Vector256.FusedMultiplyAdd(
                    sourceVector,
                    vectorScale4,
                    Vector256.LoadUnsafe(ref destination[destination4 + index]))
                    .StoreUnsafe(ref destination[destination4 + index]);
                Vector256.FusedMultiplyAdd(
                    sourceVector,
                    vectorScale5,
                    Vector256.LoadUnsafe(ref destination[destination5 + index]))
                    .StoreUnsafe(ref destination[destination5 + index]);
                Vector256.FusedMultiplyAdd(
                    sourceVector,
                    vectorScale6,
                    Vector256.LoadUnsafe(ref destination[destination6 + index]))
                    .StoreUnsafe(ref destination[destination6 + index]);
                Vector256.FusedMultiplyAdd(
                    sourceVector,
                    vectorScale7,
                    Vector256.LoadUnsafe(ref destination[destination7 + index]))
                    .StoreUnsafe(ref destination[destination7 + index]);
            }
        }

        for (; index < length; index++)
        {
            float sourceValue = source[sourceOffset + index];
            destination[destination0 + index] += scale0 * sourceValue;
            destination[destination1 + index] += scale1 * sourceValue;
            destination[destination2 + index] += scale2 * sourceValue;
            destination[destination3 + index] += scale3 * sourceValue;
            destination[destination4 + index] += scale4 * sourceValue;
            destination[destination5 + index] += scale5 * sourceValue;
            destination[destination6 + index] += scale6 * sourceValue;
            destination[destination7 + index] += scale7 * sourceValue;
        }
    }

    private static void InterpolateInPlace(
        float[] current,
        float[] next,
        float fraction)
    {
        if (Tensor.ExecutionDevice == TensorDevice.Arc)
        {
            ArcTrainingMath.Combine(current, next, current, 1f - fraction, fraction);
            return;
        }
        int index = 0;
        if (Tensor.SimdEnabled
            && Vector256.IsHardwareAccelerated
            && current.Length >= Vector256<float>.Count)
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = current.Length - current.Length % width;
            Vector256<float> vectorFraction = Vector256.Create(fraction);
            for (; index < vectorizedLength; index += width)
            {
                Vector256<float> currentValues =
                    Vector256.LoadUnsafe(ref current[index]);
                Vector256<float> nextValues =
                    Vector256.LoadUnsafe(ref next[index]);
                (currentValues
                    + vectorFraction * (nextValues - currentValues))
                    .StoreUnsafe(ref current[index]);
            }
        }

        for (; index < current.Length; index++)
        {
            current[index] += fraction * (next[index] - current[index]);
        }
    }

    private static void TransposeBack(
        float[] transposed,
        float[] destination,
        int rows,
        int columns)
    {
        if (Tensor.ExecutionDevice == TensorDevice.Arc)
        {
            ArcTrainingMath.TransposeScale(transposed, destination, columns, rows, true, 1f);
            return;
        }
        for (int row = 0; row < rows; row++)
        {
            for (int column = 0; column < columns; column++)
            {
                destination[row * columns + column] =
                    transposed[column * rows + row];
            }
        }
    }
}
