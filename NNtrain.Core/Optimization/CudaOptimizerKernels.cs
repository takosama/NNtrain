using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace NNtrain;

internal static class CudaOptimizerKernels
{
    internal static void AdamWUpdate(
        float[] data,
        float[] gradient,
        float[] firstMoment,
        float[] secondMoment,
        float beta1,
        float beta2,
        float learningRate,
        float weightDecay,
        float updateScale,
        float scaledEpsilon,
        bool applyWeightDecay)
    {
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        using var dataBuffer = accelerator.Allocate1D(data);
        using var gradientBuffer = accelerator.Allocate1D(gradient);
        using var firstMomentBuffer = accelerator.Allocate1D(firstMoment);
        using var secondMomentBuffer = accelerator.Allocate1D(secondMoment);
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
            ArrayView<float>, float, float, float, float, float, float, int>(
                AdamWKernel);
        kernel(
            data.Length,
            dataBuffer.View,
            gradientBuffer.View,
            firstMomentBuffer.View,
            secondMomentBuffer.View,
            beta1,
            beta2,
            learningRate,
            weightDecay,
            updateScale,
            scaledEpsilon,
            applyWeightDecay ? 1 : 0);
        accelerator.Synchronize();
        dataBuffer.CopyToCPU(data);
        firstMomentBuffer.CopyToCPU(firstMoment);
        secondMomentBuffer.CopyToCPU(secondMoment);
    }

    internal static void NekoMuonMoments(
        float[] gradient,
        float[] fast,
        float[] slow,
        float[] fastHat,
        float[] slowHat,
        float betaFast,
        float betaSlow,
        float fastCorrection,
        float slowCorrection)
    {
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        using var gradientBuffer = accelerator.Allocate1D(gradient);
        using var fastBuffer = accelerator.Allocate1D(fast);
        using var slowBuffer = accelerator.Allocate1D(slow);
        using var fastHatBuffer = accelerator.Allocate1D<float>(fastHat.Length);
        using var slowHatBuffer = accelerator.Allocate1D<float>(slowHat.Length);
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
            ArrayView<float>, ArrayView<float>, float, float, float, float>(
                NekoMuonMomentsKernel);
        kernel(
            fast.Length,
            gradientBuffer.View,
            fastBuffer.View,
            slowBuffer.View,
            fastHatBuffer.View,
            slowHatBuffer.View,
            betaFast,
            betaSlow,
            fastCorrection,
            slowCorrection);
        accelerator.Synchronize();
        fastBuffer.CopyToCPU(fast);
        slowBuffer.CopyToCPU(slow);
        fastHatBuffer.CopyToCPU(fastHat);
        slowHatBuffer.CopyToCPU(slowHat);
    }

    internal static void NekoMuonApplyUpdate(
        float[] data,
        float[] update,
        float learningRate,
        float finalScale,
        float weightDecay,
        bool applyWeightDecay)
    {
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        using var dataBuffer = accelerator.Allocate1D(data);
        using var updateBuffer = accelerator.Allocate1D(update);
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, float, float, float,
            int>(NekoMuonApplyUpdateKernel);
        kernel(
            data.Length,
            dataBuffer.View,
            updateBuffer.View,
            learningRate,
            finalScale,
            weightDecay,
            applyWeightDecay ? 1 : 0);
        accelerator.Synchronize();
        dataBuffer.CopyToCPU(data);
    }

    internal static void NekoMuonNewtonSchulz(
        float[] source,
        float[] destination,
        float[] gram,
        float[] gramSquared,
        int rows,
        int columns,
        float coefficientA,
        float coefficientB,
        float coefficientC)
    {
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        using var sourceBuffer = accelerator.Allocate1D(source);
        using var destinationBuffer =
            accelerator.Allocate1D<float>(destination.Length);
        using var gramBuffer = accelerator.Allocate1D<float>(gram.Length);
        using var gramSquaredBuffer =
            accelerator.Allocate1D<float>(gramSquared.Length);
        var gramKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, int, int>(
                SymmetricGramKernel);
        gramKernel(
            checked(rows * rows),
            sourceBuffer.View,
            gramBuffer.View,
            rows,
            columns);
        gramKernel(
            checked(rows * rows),
            gramBuffer.View,
            gramSquaredBuffer.View,
            rows,
            rows);
        var updateKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
            ArrayView<float>, int, int, float, float, float>(
                NewtonSchulzKernel);
        updateKernel(
            source.Length,
            sourceBuffer.View,
            gramBuffer.View,
            gramSquaredBuffer.View,
            destinationBuffer.View,
            rows,
            columns,
            coefficientA,
            coefficientB,
            coefficientC);
        accelerator.Synchronize();
        destinationBuffer.CopyToCPU(destination);
        gramBuffer.CopyToCPU(gram);
        gramSquaredBuffer.CopyToCPU(gramSquared);
    }

    private static void AdamWKernel(
        Index1D index,
        ArrayView<float> data,
        ArrayView<float> gradient,
        ArrayView<float> firstMoment,
        ArrayView<float> secondMoment,
        float beta1,
        float beta2,
        float learningRate,
        float weightDecay,
        float updateScale,
        float scaledEpsilon,
        int applyWeightDecay)
    {
        int i = index;
        float g = gradient[i];
        float first = beta1 * firstMoment[i] + (1f - beta1) * g;
        float second = beta2 * secondMoment[i] + (1f - beta2) * g * g;
        firstMoment[i] = first;
        secondMoment[i] = second;
        float parameter = data[i];
        if (applyWeightDecay != 0)
            parameter *= 1f - learningRate * weightDecay;
        data[i] = parameter - updateScale * first /
            (XMath.Sqrt(second) + scaledEpsilon);
    }

    private static void NekoMuonMomentsKernel(
        Index1D index,
        ArrayView<float> gradient,
        ArrayView<float> fast,
        ArrayView<float> slow,
        ArrayView<float> fastHat,
        ArrayView<float> slowHat,
        float betaFast,
        float betaSlow,
        float fastCorrection,
        float slowCorrection)
    {
        int i = index;
        float nextFast = betaFast * fast[i] +
            (1f - betaFast) * gradient[i];
        float nextSlow = betaSlow * slow[i] +
            (1f - betaSlow) * gradient[i];
        fast[i] = nextFast;
        slow[i] = nextSlow;
        fastHat[i] = nextFast / fastCorrection;
        slowHat[i] = nextSlow / slowCorrection;
    }

    private static void NekoMuonApplyUpdateKernel(
        Index1D index,
        ArrayView<float> data,
        ArrayView<float> update,
        float learningRate,
        float finalScale,
        float weightDecay,
        int applyWeightDecay)
    {
        int i = index;
        float parameter = data[i];
        if (applyWeightDecay != 0)
            parameter -= learningRate * weightDecay * parameter;
        data[i] = parameter - learningRate * finalScale * update[i];
    }

    private static void SymmetricGramKernel(
        Index1D index,
        ArrayView<float> source,
        ArrayView<float> destination,
        int rows,
        int columns)
    {
        int linear = index;
        int row = linear / rows;
        int other = linear - row * rows;
        float sum = 0f;
        int rowOffset = row * columns;
        int otherOffset = other * columns;
        for (int column = 0; column < columns; column++)
            sum += source[rowOffset + column] * source[otherOffset + column];
        destination[linear] = sum;
    }

    private static void NewtonSchulzKernel(
        Index1D index,
        ArrayView<float> source,
        ArrayView<float> gram,
        ArrayView<float> gramSquared,
        ArrayView<float> destination,
        int rows,
        int columns,
        float coefficientA,
        float coefficientB,
        float coefficientC)
    {
        int linear = index;
        int row = linear / columns;
        int column = linear - row * columns;
        float result = coefficientA * source[linear];
        int coefficientOffset = row * rows;
        for (int inner = 0; inner < rows; inner++)
        {
            float coefficient =
                coefficientB * gram[coefficientOffset + inner] +
                coefficientC * gramSquared[coefficientOffset + inner];
            result += coefficient * source[inner * columns + column];
        }
        destination[linear] = result;
    }
}
