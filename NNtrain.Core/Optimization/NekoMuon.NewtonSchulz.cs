using System.Diagnostics;
using NNtrain.Cuda.Interop;

namespace NNtrain;

public sealed partial class NekoMuon
{
    internal static float ResolveNewtonSchulzDepth(
        NekoMuonOptions options,
        float confidence,
        bool runNewtonSchulz)
    {
        ArgumentNullException.ThrowIfNull(options);
        return ResolveNewtonSchulzDepth(
            options.MaxNewtonSchulzSteps,
            options.NewtonSchulzDepthMode,
            options.NewtonSchulzDepth,
            confidence,
            runNewtonSchulz);
    }

    internal static float ResolveNewtonSchulzDepth(
        int maxNewtonSchulzSteps,
        NekoMuonNewtonSchulzDepthMode mode,
        float configuredDepth,
        float confidence,
        bool runNewtonSchulz)
    {
        if (!runNewtonSchulz)
            return 0f;

        float adaptiveDepth = maxNewtonSchulzSteps * confidence;
        float depth = mode switch
        {
            NekoMuonNewtonSchulzDepthMode.Adaptive => adaptiveDepth,
            NekoMuonNewtonSchulzDepthMode.Minimum =>
                MathF.Max(adaptiveDepth, configuredDepth),
            NekoMuonNewtonSchulzDepthMode.Fixed => configuredDepth,
            _ => throw new ArgumentOutOfRangeException(
                nameof(mode),
                mode,
                "NekoMuon Newton-Schulz depth mode is invalid."),
        };
        return Math.Clamp(depth, 0f, maxNewtonSchulzSteps);
    }

    private static void InitializeMuonMatrix(
        float[] source,
        float[] destination,
        int rows,
        int columns,
        bool transpose,
        float epsilon)
    {
        if (Tensor.ExecutionDevice == TensorDevice.Arc)
        {
            float inverse = 1f / (MathF.Sqrt(ArcTrainingMath.Sum(source, squared: true)) + epsilon);
            ArcTrainingMath.TransposeScale(source, destination, rows, columns, transpose, inverse);
            return;
        }
        double normSquared = 0d;
        int index = 0;
        if (Tensor.SimdEnabled
            && Vector256.IsHardwareAccelerated
            && source.Length >= Vector256<float>.Count)
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = source.Length - source.Length % width;
            for (; index < vectorizedLength; index += width)
            {
                Vector256<float> values =
                    Vector256.LoadUnsafe(ref source[index]);
                normSquared += Vector256.Sum(values * values);
            }
        }

        for (; index < source.Length; index++)
            normSquared += (double)source[index] * source[index];
        float inverseNorm = 1f / ((float)Math.Sqrt(normSquared) + epsilon);

        if (!transpose)
        {
            Scale(source, destination, inverseNorm);
            return;
        }

        for (int row = 0; row < rows; row++)
        {
            for (int column = 0; column < columns; column++)
            {
                destination[column * rows + row] =
                    source[row * columns + column] * inverseNorm;
            }
        }
    }

    private static void NewtonSchulz(
        float[] source,
        float[] destination,
        float[] gram,
        float[] gramSquared,
        int rows,
        int columns,
        bool bfloat16MatrixOperands,
        long[]? profileTicks)
    {
        if (Tensor.ExecutionDevice == TensorDevice.Arc)
        {
            long start = profileTicks is null ? 0 : Stopwatch.GetTimestamp();
            ArcTrainingMath.NewtonSchulz(source, destination, gram, gramSquared, rows, columns,
                bfloat16MatrixOperands, NewtonSchulzA, NewtonSchulzB, NewtonSchulzC);
            AddProfileTicks(profileTicks, 8, start);
            return;
        }
        if (Tensor.ExecutionDevice == TensorDevice.Cuda)
        {
            long cudaStart = profileTicks is null
                ? 0L
                : Stopwatch.GetTimestamp();
            CudaOptimizerKernels.NekoMuonNewtonSchulz(
                source,
                destination,
                gram,
                gramSquared,
                rows,
                columns,
                NewtonSchulzA,
                NewtonSchulzB,
                NewtonSchulzC);
            AddProfileTicks(profileTicks, 8, cudaStart);
            return;
        }

        long phaseStart = profileTicks is null
            ? 0L
            : Stopwatch.GetTimestamp();
        ComputeSymmetricGram(
            source, gram, rows, columns, bfloat16MatrixOperands);
        AddProfileTicks(profileTicks, 6, phaseStart);

        phaseStart = profileTicks is null ? 0L : Stopwatch.GetTimestamp();
        ComputeSymmetricGram(
            gram, gramSquared, rows, rows, bfloat16MatrixOperands);
        AddProfileTicks(profileTicks, 7, phaseStart);

        phaseStart = profileTicks is null ? 0L : Stopwatch.GetTimestamp();
        if (bfloat16MatrixOperands)
        {
            // The CUDA Tensor Core path performs three BF16-operand/FP32-
            // accumulate GEMMs: X*X^T, G*G, and P*X.  Mirror those operand
            // boundaries on CPU while retaining FP32 reductions and output
            // storage, as required by PrecisionPolicy.
            for (int bfOutputRow = 0; bfOutputRow < rows; bfOutputRow++)
            {
                int coefficientOffset = bfOutputRow * rows;
                int destinationOffset = bfOutputRow * columns;
                for (int column = 0; column < columns; column++)
                {
                    float sum = 0f;
                    for (int inner = 0; inner < rows; inner++)
                    {
                        float coefficient = MathF.FusedMultiplyAdd(
                            NewtonSchulzC,
                            gramSquared[coefficientOffset + inner],
                            NewtonSchulzB
                                * gram[coefficientOffset + inner]);
                        if (bfOutputRow == inner)
                            coefficient += NewtonSchulzA;
                        coefficient = TensorStorageCodec.RoundToBFloat16(
                            coefficient);
                        float sourceValue = TensorStorageCodec.RoundToBFloat16(
                            source[inner * columns + column]);
                        sum = MathF.FusedMultiplyAdd(
                            coefficient, sourceValue, sum);
                    }
                    destination[destinationOffset + column] = sum;
                }
            }
            AddProfileTicks(profileTicks, 8, phaseStart);
            return;
        }

        Scale(source, destination, NewtonSchulzA);
        int outputRow = 0;
        for (; outputRow + 7 < rows; outputRow += 8)
        {
            int destination0 = outputRow * columns;
            int destination1 = destination0 + columns;
            int destination2 = destination1 + columns;
            int destination3 = destination2 + columns;
            int destination4 = destination3 + columns;
            int destination5 = destination4 + columns;
            int destination6 = destination5 + columns;
            int destination7 = destination6 + columns;
            int coefficient0 = outputRow * rows;
            int coefficient1 = coefficient0 + rows;
            int coefficient2 = coefficient1 + rows;
            int coefficient3 = coefficient2 + rows;
            int coefficient4 = coefficient3 + rows;
            int coefficient5 = coefficient4 + rows;
            int coefficient6 = coefficient5 + rows;
            int coefficient7 = coefficient6 + rows;

            for (int inner = 0; inner < rows; inner++)
            {
                AddScaledEightRows(
                    source,
                    inner * columns,
                    destination,
                    destination0,
                    destination1,
                    destination2,
                    destination3,
                    destination4,
                    destination5,
                    destination6,
                    destination7,
                    columns,
                    NewtonSchulzB * gram[coefficient0 + inner]
                        + NewtonSchulzC * gramSquared[coefficient0 + inner],
                    NewtonSchulzB * gram[coefficient1 + inner]
                        + NewtonSchulzC * gramSquared[coefficient1 + inner],
                    NewtonSchulzB * gram[coefficient2 + inner]
                        + NewtonSchulzC * gramSquared[coefficient2 + inner],
                    NewtonSchulzB * gram[coefficient3 + inner]
                        + NewtonSchulzC * gramSquared[coefficient3 + inner],
                    NewtonSchulzB * gram[coefficient4 + inner]
                        + NewtonSchulzC * gramSquared[coefficient4 + inner],
                    NewtonSchulzB * gram[coefficient5 + inner]
                        + NewtonSchulzC * gramSquared[coefficient5 + inner],
                    NewtonSchulzB * gram[coefficient6 + inner]
                        + NewtonSchulzC * gramSquared[coefficient6 + inner],
                    NewtonSchulzB * gram[coefficient7 + inner]
                        + NewtonSchulzC * gramSquared[coefficient7 + inner]);
            }
        }

        for (; outputRow + 3 < rows; outputRow += 4)
        {
            int destination0 = outputRow * columns;
            int destination1 = destination0 + columns;
            int destination2 = destination1 + columns;
            int destination3 = destination2 + columns;
            int coefficient0 = outputRow * rows;
            int coefficient1 = coefficient0 + rows;
            int coefficient2 = coefficient1 + rows;
            int coefficient3 = coefficient2 + rows;

            for (int inner = 0; inner < rows; inner++)
            {
                AddScaledFourRows(
                    source,
                    inner * columns,
                    destination,
                    destination0,
                    destination1,
                    destination2,
                    destination3,
                    columns,
                    NewtonSchulzB * gram[coefficient0 + inner]
                        + NewtonSchulzC
                            * gramSquared[coefficient0 + inner],
                    NewtonSchulzB * gram[coefficient1 + inner]
                        + NewtonSchulzC
                            * gramSquared[coefficient1 + inner],
                    NewtonSchulzB * gram[coefficient2 + inner]
                        + NewtonSchulzC
                            * gramSquared[coefficient2 + inner],
                    NewtonSchulzB * gram[coefficient3 + inner]
                        + NewtonSchulzC
                            * gramSquared[coefficient3 + inner]);
            }
        }

        for (; outputRow < rows; outputRow++)
        {
            int destinationOffset = outputRow * columns;
            for (int inner = 0; inner < rows; inner++)
            {
                float coefficient =
                    NewtonSchulzB * gram[outputRow * rows + inner]
                    + NewtonSchulzC
                        * gramSquared[outputRow * rows + inner];
                AddScaled(
                    source,
                    inner * columns,
                    destination,
                    destinationOffset,
                    columns,
                    coefficient);
            }
        }
        AddProfileTicks(profileTicks, 8, phaseStart);
    }

}
