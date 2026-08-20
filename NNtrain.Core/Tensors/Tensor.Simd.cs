using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading;

namespace NNtrain;

partial class Tensor
{
    private static int _simdEnabled = 1;
    private static int _float16NativeEnabled = 1;
    private static int _maxDegreeOfParallelism;

    /// <summary>
    /// Gets or sets whether tensor operations may use hardware-accelerated
    /// 256-bit and 128-bit SIMD.
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

    /// <summary>
    /// Gets or sets whether optional native Float16 accelerators may be used.
    /// The portable managed Float32-SIMD path is used when disabled or when no
    /// compatible native payload/CPU is available.
    /// </summary>
    public static bool Float16NativeEnabled
    {
        get => Volatile.Read(ref _float16NativeEnabled) != 0;
        set => Volatile.Write(ref _float16NativeEnabled, value ? 1 : 0);
    }

    /// <summary>Gets whether an optional native Float16 accelerator is active.</summary>
    public static bool IsFloat16NativeAccelerated
        => Float16NativeEnabled && TensorFloat16Native.IsAvailable;

    /// <summary>
    /// Gets or sets the maximum number of worker threads used by tensor and
    /// optimizer <see cref="Parallel.For(int, int, Action{int})"/> kernels.
    /// Zero selects the runtime default.
    /// </summary>
    public static int MaxDegreeOfParallelism
    {
        get => Volatile.Read(ref _maxDegreeOfParallelism);
        set
        {
            if (value < 0)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(value),
                    value,
                    "Maximum parallelism must be non-negative; zero means " +
                    "automatic.");
            }
            Volatile.Write(ref _maxDegreeOfParallelism, value);
        }
    }

    public static int EffectiveMaxDegreeOfParallelism
        => MaxDegreeOfParallelism == 0
            ? Environment.ProcessorCount
            : MaxDegreeOfParallelism;

    internal static void RunParallel(
        int fromInclusive,
        int toExclusive,
        Action<int> action)
    {
        ArgumentNullException.ThrowIfNull(action);
        int count = toExclusive - fromInclusive;
        int configured = MaxDegreeOfParallelism;
        if (count <= 1
            || configured == 1
            || (configured == 0 && Environment.ProcessorCount == 1))
        {
            for (int index = fromInclusive; index < toExclusive; index++)
                action(index);
            return;
        }

        if (configured == 0)
        {
            Parallel.For(fromInclusive, toExclusive, action);
            return;
        }

        Parallel.For(
            fromInclusive,
            toExclusive,
            new ParallelOptions
            {
                MaxDegreeOfParallelism = configured,
            },
            action);
    }

    private static bool CanUseSimd(int length)
        => SimdEnabled
            && Vector256.IsHardwareAccelerated
            && length >= Vector256<float>.Count;

    private static bool CanUseVector128(int length)
        => SimdEnabled
            && Vector128.IsHardwareAccelerated
            && length >= Vector128<float>.Count;

    private static bool CanUseTransposedRightKernel(int inputWidth, int outputWidth)
    {
        // The output-vectorized outer-product kernel is most useful for a much
        // wider projection (notably the vocabulary head). For square-ish
        // Transformer matrices, contiguous input/weight dot products avoid a
        // per-step transpose and have better cache locality.
        return inputWidth <= 128
            && CanUseSimd(outputWidth)
            && outputWidth >= 4L * inputWidth;
    }

    private static Vector256<float> LoadVector256(
        float[] values,
        int offset)
        => Vector256.LoadUnsafe(ref values[offset]);

    private static Vector256<float> LoadVector256(
        TensorStorage values,
        int offset)
        => values.LoadVector256(offset);

    private static Vector256<float> LoadVector256(
        Half[] values,
        int offset)
    {
        // Keep the direct AVX2 route in the hot loop. The JIT folds this ISA
        // check on a given machine, unlike a codec-level fallback call per
        // vector. ARM and other non-AVX2 runtimes still receive the safe
        // scalar/vector construction fallback.
        return Avx2.IsSupported
            ? TensorStorageCodec.LoadFloat16Vector256Avx2(values, offset)
            : TensorStorageCodec.LoadFloat16Vector256(values, offset);
    }

    private static void StoreVector256(
        Vector256<float> vector,
        float[] destination,
        int offset)
        => vector.StoreUnsafe(ref destination[offset]);

    private static Vector128<float> LoadVector128(
        float[] values,
        int offset)
        => Vector128.LoadUnsafe(ref values[offset]);

    private static Vector128<float> LoadVector128(
        TensorStorage values,
        int offset)
        => values.LoadVector128(offset);

    private static Vector128<float> LoadVector128(
        Half[] values,
        int offset)
        => TensorStorageCodec.LoadFloat16Vector128(values, offset);

    private static void StoreVector128(
        Vector128<float> vector,
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

        if (CanUseVector128(length - index))
        {
            StoreVector128(
                LoadVector128(destination, destinationOffset + index)
                    + LoadVector128(source, sourceOffset + index)
                        * Vector128.Create(scale),
                destination,
                destinationOffset + index);
            index += Vector128<float>.Count;
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

        if (CanUseVector128(length - index))
        {
            StoreVector128(
                LoadVector128(source, sourceOffset + index)
                    * Vector128.Create(scale),
                destination,
                destinationOffset + index);
            index += Vector128<float>.Count;
        }

        for (; index < length; index++)
        {
            destination[destinationOffset + index] =
                source[sourceOffset + index] * scale;
        }
    }

    private static void MultiplyValues(
        TensorStorage source,
        int sourceOffset,
        float scale,
        float[] destination,
        int destinationOffset,
        int length)
    {
        if (source.TryGetFloat32Buffer(out float[] sourceValues))
        {
            MultiplyValues(
                sourceValues,
                sourceOffset,
                scale,
                destination,
                destinationOffset,
                length);
            return;
        }
        bool sourceIsHalf = source.TryGetFloat16Buffer(out Half[] sourceHalf);

        int index = 0;
        if (CanUseSimd(length))
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = length - length % width;
            Vector256<float> scaleVector = Vector256.Create(scale);
            for (; index < vectorizedLength; index += width)
            {
                StoreVector256(
                    (sourceIsHalf
                        ? LoadVector256(sourceHalf, sourceOffset + index)
                        : LoadVector256(source, sourceOffset + index))
                        * scaleVector,
                    destination,
                    destinationOffset + index);
            }
        }
        if (CanUseVector128(length - index))
        {
            StoreVector128(
                LoadVector128(source, sourceOffset + index)
                    * Vector128.Create(scale),
                destination,
                destinationOffset + index);
            index += Vector128<float>.Count;
        }
        for (; index < length; index++)
        {
            destination[destinationOffset + index] =
                source[sourceOffset + index] * scale;
        }
    }

    private static void AddValues(
        float[] left,
        int leftOffset,
        float[] right,
        int rightOffset,
        float[] destination,
        int destinationOffset,
        int length)
    {
        int index = 0;
        if (CanUseSimd(length))
        {
            int vectorWidth = Vector256<float>.Count;
            int vectorizedLength = length - length % vectorWidth;
            for (; index < vectorizedLength; index += vectorWidth)
            {
                StoreVector256(
                    LoadVector256(left, leftOffset + index)
                        + LoadVector256(right, rightOffset + index),
                    destination,
                    destinationOffset + index);
            }
        }

        if (CanUseVector128(length - index))
        {
            StoreVector128(
                LoadVector128(left, leftOffset + index)
                    + LoadVector128(right, rightOffset + index),
                destination,
                destinationOffset + index);
            index += Vector128<float>.Count;
        }

        for (; index < length; index++)
        {
            destination[destinationOffset + index] =
                left[leftOffset + index] + right[rightOffset + index];
        }
    }

    private static void AddValues(
        TensorStorage left,
        int leftOffset,
        TensorStorage right,
        int rightOffset,
        float[] destination,
        int destinationOffset,
        int length)
    {
        if (left.TryGetFloat32Buffer(out float[] leftValues)
            && right.TryGetFloat32Buffer(out float[] rightValues))
        {
            AddValues(
                leftValues,
                leftOffset,
                rightValues,
                rightOffset,
                destination,
                destinationOffset,
                length);
            return;
        }
        bool leftIsHalf = left.TryGetFloat16Buffer(out Half[] leftHalf);
        bool rightIsHalf = right.TryGetFloat16Buffer(out Half[] rightHalf);

        int index = 0;
        if (CanUseSimd(length))
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = length - length % width;
            for (; index < vectorizedLength; index += width)
            {
                StoreVector256(
                    (leftIsHalf
                        ? LoadVector256(leftHalf, leftOffset + index)
                        : LoadVector256(left, leftOffset + index))
                        + (rightIsHalf
                            ? LoadVector256(rightHalf, rightOffset + index)
                            : LoadVector256(right, rightOffset + index)),
                    destination,
                    destinationOffset + index);
            }
        }
        if (CanUseVector128(length - index))
        {
            StoreVector128(
                LoadVector128(left, leftOffset + index)
                    + LoadVector128(right, rightOffset + index),
                destination,
                destinationOffset + index);
            index += Vector128<float>.Count;
        }
        for (; index < length; index++)
        {
            destination[destinationOffset + index] =
                (leftIsHalf
                    ? (float)leftHalf[leftOffset + index]
                    : left[leftOffset + index])
                + (rightIsHalf
                    ? (float)rightHalf[rightOffset + index]
                    : right[rightOffset + index]);
        }
    }

    private static void MultiplyElementwiseValues(
        float[] left,
        int leftOffset,
        float[] right,
        int rightOffset,
        float[] destination,
        int destinationOffset,
        int length)
    {
        int index = 0;
        if (CanUseSimd(length))
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = length - length % width;
            for (; index < vectorizedLength; index += width)
            {
                StoreVector256(
                    LoadVector256(left, leftOffset + index)
                        * LoadVector256(right, rightOffset + index),
                    destination,
                    destinationOffset + index);
            }
        }


        if (CanUseVector128(length - index))
        {
            StoreVector128(
                LoadVector128(left, leftOffset + index)
                    * LoadVector128(right, rightOffset + index),
                destination,
                destinationOffset + index);
            index += Vector128<float>.Count;
        }

        for (; index < length; index++)
        {
            destination[destinationOffset + index] =
                left[leftOffset + index] * right[rightOffset + index];
        }
    }

    private static void AddScaledValues(
        float[] destination,
        int destinationOffset,
        TensorStorage source,
        int sourceOffset,
        float scale,
        int length)
    {
        if (source.TryGetFloat32Buffer(out float[] sourceValues))
        {
            AddScaledValues(
                destination,
                destinationOffset,
                sourceValues,
                sourceOffset,
                scale,
                length);
            return;
        }
        bool sourceIsHalf = source.TryGetFloat16Buffer(out Half[] sourceHalf);

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
                Vector256<float> sourceVector = sourceIsHalf
                    ? LoadVector256(sourceHalf, sourceOffset + index)
                    : LoadVector256(source, sourceOffset + index);
                StoreVector256(
                    destinationVector + sourceVector * scaleVector,
                    destination,
                    destinationOffset + index);
            }
        }

        if (CanUseVector128(length - index))
        {
            StoreVector128(
                LoadVector128(destination, destinationOffset + index)
                    + LoadVector128(source, sourceOffset + index)
                        * Vector128.Create(scale),
                destination,
                destinationOffset + index);
            index += Vector128<float>.Count;
        }

        for (; index < length; index++)
        {
            destination[destinationOffset + index] +=
                source[sourceOffset + index] * scale;
        }
    }

    /// <summary>
    /// Adds a sum of strided element-wise products to one contiguous vector.
    /// Keeping four destination vectors in registers avoids a load/store pair
    /// for every convolution tap while retaining contiguous source reads.
    /// </summary>
    private static void AddStridedElementwiseProductSum(
        float[] destination,
        int destinationOffset,
        float[] left,
        int leftOffset,
        int leftStride,
        float[] right,
        int rightOffset,
        int rightStride,
        int termCount,
        int length)
    {
        int index = 0;
        if (CanUseSimd(length))
        {
            int width = Vector256<float>.Count;
            int blockWidth = 4 * width;
            int blockEnd = length - length % blockWidth;
            for (; index < blockEnd; index += blockWidth)
            {
                Vector256<float> sum0 = LoadVector256(
                    destination,
                    destinationOffset + index);
                Vector256<float> sum1 = LoadVector256(
                    destination,
                    destinationOffset + index + width);
                Vector256<float> sum2 = LoadVector256(
                    destination,
                    destinationOffset + index + 2 * width);
                Vector256<float> sum3 = LoadVector256(
                    destination,
                    destinationOffset + index + 3 * width);
                int leftTermOffset = leftOffset + index;
                int rightTermOffset = rightOffset + index;
                for (int term = 0; term < termCount; term++)
                {
                    sum0 = Vector256.FusedMultiplyAdd(
                        LoadVector256(left, leftTermOffset),
                        LoadVector256(right, rightTermOffset),
                        sum0);
                    sum1 = Vector256.FusedMultiplyAdd(
                        LoadVector256(left, leftTermOffset + width),
                        LoadVector256(right, rightTermOffset + width),
                        sum1);
                    sum2 = Vector256.FusedMultiplyAdd(
                        LoadVector256(left, leftTermOffset + 2 * width),
                        LoadVector256(right, rightTermOffset + 2 * width),
                        sum2);
                    sum3 = Vector256.FusedMultiplyAdd(
                        LoadVector256(left, leftTermOffset + 3 * width),
                        LoadVector256(right, rightTermOffset + 3 * width),
                        sum3);
                    leftTermOffset += leftStride;
                    rightTermOffset += rightStride;
                }
                StoreVector256(sum0, destination, destinationOffset + index);
                StoreVector256(
                    sum1,
                    destination,
                    destinationOffset + index + width);
                StoreVector256(
                    sum2,
                    destination,
                    destinationOffset + index + 2 * width);
                StoreVector256(
                    sum3,
                    destination,
                    destinationOffset + index + 3 * width);
            }

            int vectorEnd = length - length % width;
            for (; index < vectorEnd; index += width)
            {
                Vector256<float> sum = LoadVector256(
                    destination,
                    destinationOffset + index);
                int leftTermOffset = leftOffset + index;
                int rightTermOffset = rightOffset + index;
                for (int term = 0; term < termCount; term++)
                {
                    sum = Vector256.FusedMultiplyAdd(
                        LoadVector256(left, leftTermOffset),
                        LoadVector256(right, rightTermOffset),
                        sum);
                    leftTermOffset += leftStride;
                    rightTermOffset += rightStride;
                }
                StoreVector256(sum, destination, destinationOffset + index);
            }
        }

        if (CanUseVector128(length - index))
        {
            Vector128<float> sum = LoadVector128(
                destination,
                destinationOffset + index);
            int leftTermOffset = leftOffset + index;
            int rightTermOffset = rightOffset + index;
            for (int term = 0; term < termCount; term++)
            {
                sum = Vector128.FusedMultiplyAdd(
                    LoadVector128(left, leftTermOffset),
                    LoadVector128(right, rightTermOffset),
                    sum);
                leftTermOffset += leftStride;
                rightTermOffset += rightStride;
            }
            StoreVector128(sum, destination, destinationOffset + index);
            index += Vector128<float>.Count;
        }

        for (; index < length; index++)
        {
            float sum = destination[destinationOffset + index];
            int leftTermOffset = leftOffset + index;
            int rightTermOffset = rightOffset + index;
            for (int term = 0; term < termCount; term++)
            {
                sum += left[leftTermOffset] * right[rightTermOffset];
                leftTermOffset += leftStride;
                rightTermOffset += rightStride;
            }
            destination[destinationOffset + index] = sum;
        }
    }

    private static void AddConstantValuesInPlace(
        float[] values,
        int offset,
        float constant,
        int length)
    {
        int index = 0;
        if (CanUseSimd(length))
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = length - length % width;
            Vector256<float> constantVector = Vector256.Create(constant);
            for (; index < vectorizedLength; index += width)
            {
                StoreVector256(
                    LoadVector256(values, offset + index) + constantVector,
                    values,
                    offset + index);
            }
        }


        if (CanUseVector128(length - index))
        {
            StoreVector128(
                LoadVector128(values, offset + index)
                    + Vector128.Create(constant),
                values,
                offset + index);
            index += Vector128<float>.Count;
        }

        for (; index < length; index++)
            values[offset + index] += constant;
    }

    private static float MaxValues(
        float[] values,
        int offset,
        int length)
    {
        if (length <= 0)
            throw new ArgumentOutOfRangeException(nameof(length));

        int index = 1;
        float maximum = values[offset];
        if (CanUseSimd(length))
        {
            int vectorWidth = Vector256<float>.Count;
            int vectorizedLength = length - length % vectorWidth;
            Vector256<float> maximumVector = LoadVector256(values, offset);
            index = vectorWidth;
            for (; index < vectorizedLength; index += vectorWidth)
            {
                maximumVector = Vector256.Max(
                    maximumVector,
                    LoadVector256(values, offset + index));
            }

            for (int lane = 0; lane < vectorWidth; lane++)
                maximum = MathF.Max(maximum, maximumVector.GetElement(lane));
        }

        for (; index < length; index++)
            maximum = MathF.Max(maximum, values[offset + index]);
        return maximum;
    }

    private static float ExpShiftedValues(
        float[] source,
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
            int vectorizedLength = length - length % width;
            Vector256<float> shiftVector = Vector256.Create(shift);
            Vector256<float> sumVector = Vector256<float>.Zero;

            for (; index < vectorizedLength; index += width)
            {
                Vector256<float> result = ExpVector(Vector256.Max(
                    LoadVector256(source, sourceOffset + index)
                        - shiftVector,
                    Vector256.Create(-87.33654f)));
                StoreVector256(
                    result,
                    destination,
                    destinationOffset + index);
                sumVector += result;
            }
            sum = Vector256.Sum(sumVector);
        }

        for (; index < length; index++)
        {
            float result = MathF.Exp(source[sourceOffset + index] - shift);
            destination[destinationOffset + index] = result;
            sum += result;
        }
        return sum;
    }

    private static float SumExpShiftedValues(
        float[] source,
        int sourceOffset,
        float shift,
        int length)
    {
        int index = 0;
        float sum = 0f;
        if (CanUseSimd(length))
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = length - length % width;
            Vector256<float> shiftVector = Vector256.Create(shift);
            Vector256<float> minimum = Vector256.Create(-87.33654f);
            Vector256<float> sumVector = Vector256<float>.Zero;
            for (; index < vectorizedLength; index += width)
            {
                sumVector += ExpVector(Vector256.Max(
                    LoadVector256(source, sourceOffset + index)
                        - shiftVector,
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
        float[] logits,
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
            int vectorizedLength = length - length % width;
            Vector256<float> maximumVector = Vector256.Create(maximum);
            Vector256<float> minimum = Vector256.Create(-87.33654f);
            Vector256<float> scaleVector = Vector256.Create(probabilityScale);
            Vector256<float> uniformVector = Vector256.Create(uniformTarget);
            for (; index < vectorizedLength; index += width)
            {
                Vector256<float> probability = ExpVector(Vector256.Max(
                    LoadVector256(logits, logitsOffset + index)
                        - maximumVector,
                    minimum));
                StoreVector256(
                    LoadVector256(destination, destinationOffset + index)
                        + probability * scaleVector
                        - uniformVector,
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

    private static Vector256<float> ExpVector(Vector256<float> value)
    {
        Vector256<float> exponent = Vector256.Floor(
            value * Vector256.Create(1.44269504088896341f)
                + Vector256.Create(0.5f));
        value -= exponent * Vector256.Create(0.693359375f);
        value -= exponent * Vector256.Create(-2.12194440e-4f);
        Vector256<float> square = value * value;

        Vector256<float> polynomial =
            Vector256.Create(1.9875691500e-4f);
        polynomial = polynomial * value
            + Vector256.Create(1.3981999507e-3f);
        polynomial = polynomial * value
            + Vector256.Create(8.3334519073e-3f);
        polynomial = polynomial * value
            + Vector256.Create(4.1665795894e-2f);
        polynomial = polynomial * value
            + Vector256.Create(1.6666665459e-1f);
        polynomial = polynomial * value
            + Vector256.Create(5.0000001201e-1f);
        polynomial = polynomial * square + value + Vector256.Create(1f);

        Vector256<int> exponentBits =
            (Vector256.ConvertToInt32(exponent) + Vector256.Create(127)) << 23;
        return polynomial * exponentBits.AsSingle();
    }

    private static void SubtractShiftAndScalarValues(
        float[] source,
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
            int vectorizedLength = length - length % width;
            Vector256<float> shiftVector = Vector256.Create(shift);
            Vector256<float> scalarVector = Vector256.Create(scalar);
            for (; index < vectorizedLength; index += width)
            {
                StoreVector256(
                    (LoadVector256(source, sourceOffset + index)
                        - shiftVector)
                        - scalarVector,
                    destination,
                    destinationOffset + index);
            }
        }

        for (; index < length; index++)
        {
            destination[destinationOffset + index] =
                (source[sourceOffset + index] - shift) - scalar;
        }
    }

    private static void ReluValuesInPlace(
        float[] values,
        int offset,
        int length)
    {
        int index = 0;
        if (CanUseSimd(length))
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = length - length % width;
            Vector256<float> zero = Vector256<float>.Zero;
            for (; index < vectorizedLength; index += width)
            {
                Vector256<float> value = LoadVector256(
                    values,
                    offset + index);
                StoreVector256(
                    Vector256.ConditionalSelect(
                        Vector256.GreaterThan(value, zero),
                        value,
                        zero),
                    values,
                    offset + index);
            }
        }

        for (; index < length; index++)
        {
            float value = values[offset + index];
            values[offset + index] = value > 0f ? value : 0f;
        }
    }

    private static float DotProductMaskedByPositive(
        float[] gradient,
        int gradientOffset,
        float[] activation,
        int activationOffset,
        float[] weight,
        int weightOffset,
        int length)
    {
        int index = 0;
        float sum = 0f;
        if (CanUseSimd(length))
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = length - length % width;
            Vector256<float> zero = Vector256<float>.Zero;
            Vector256<float> sumVector = zero;
            for (; index < vectorizedLength; index += width)
            {
                Vector256<float> activationVector = LoadVector256(
                    activation,
                    activationOffset + index);
                Vector256<float> maskedGradient =
                    Vector256.ConditionalSelect(
                        Vector256.GreaterThan(activationVector, zero),
                        LoadVector256(gradient, gradientOffset + index),
                        zero);
                sumVector += maskedGradient
                    * LoadVector256(weight, weightOffset + index);
            }
            sum = Vector256.Sum(sumVector);
        }

        for (; index < length; index++)
        {
            if (activation[activationOffset + index] > 0f)
            {
                sum += gradient[gradientOffset + index]
                    * weight[weightOffset + index];
            }
        }
        return sum;
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
