using System.Buffers;
using System.Collections.Concurrent;

namespace NNtrain;

partial class Tensor
{
    private const long FftScratchBudgetBytes = 128L * 1024 * 1024;
    private static readonly ConcurrentDictionary<int, FftPlan> FftPlans = new();

    private static int GetFftLength(int sequence)
    {
        int required = checked(2 * sequence - 1);
        int length = 1;
        while (length < required)
            length = checked(length * 2);
        return length;
    }

    private static int GetFftChunkSize(
        int batch,
        int spectrumElements,
        int buffersPerTransform)
    {
        long bytesPerTransform = checked(
            (long)spectrumElements * buffersPerTransform * sizeof(float));
        long budgetCount = Math.Max(1, FftScratchBudgetBytes / bytesPerTransform);
        return (int)Math.Min(batch, budgetCount);
    }

    private static void AddCausalConvolutionFft(
        float[] signal,
        int batch,
        int sequence,
        int width,
        TensorStorage filter,
        float[] destination)
    {
        int fftLength = GetFftLength(sequence);
        int spectrumElements = checked(fftLength * width);
        ArrayPool<float> pool = ArrayPool<float>.Shared;
        float[] filterReal = pool.Rent(spectrumElements);
        float[] filterImaginary = pool.Rent(spectrumElements);
        try
        {
            int chunkCapacity = GetFftChunkSize(
                batch,
                spectrumElements,
                buffersPerTransform: 2);
            int chunkElements = checked(chunkCapacity * spectrumElements);
            float[] signalReal = pool.Rent(chunkElements);
            float[] signalImaginary = pool.Rent(chunkElements);
            try
            {
                bool filterSpectrumReady = false;
                for (int batchStart = 0;
                    batchStart < batch;
                    batchStart += chunkCapacity)
                {
                    int chunkCount = Math.Min(chunkCapacity, batch - batchStart);
                    int activeElements = checked(chunkCount * spectrumElements);
                    signalReal.AsSpan(0, activeElements).Clear();
                    signalImaginary.AsSpan(0, activeElements).Clear();
                    for (int localBatch = 0;
                        localBatch < chunkCount;
                        localBatch++)
                    {
                        Array.Copy(
                            signal,
                            checked((batchStart + localBatch) * sequence * width),
                            signalReal,
                            localBatch * spectrumElements,
                            checked(sequence * width));
                    }
                    if (!filterSpectrumReady)
                    {
                        filter.CopyRangeTo(
                            0,
                            signalImaginary.AsSpan(
                                0,
                                checked(sequence * width)));
                    }

                    FftChannelsInPlace(
                        signalReal,
                        signalImaginary,
                        chunkCount,
                        fftLength,
                        width,
                        inverse: false);
                    if (!filterSpectrumReady)
                    {
                        UnpackTwoRealFfts(
                            signalReal,
                            signalImaginary,
                            signalReal,
                            signalImaginary,
                            filterReal,
                            filterImaginary,
                            transformCount: 1,
                            fftLength,
                            width);
                        filterSpectrumReady = true;
                    }
                    MultiplySpectra(
                        signalReal,
                        signalImaginary,
                        filterReal,
                        filterImaginary,
                        signalReal,
                        signalImaginary,
                        chunkCount,
                        fftLength,
                        width,
                        rightShared: true,
                        conjugateRight: false);
                    FftChannelsInPlace(
                        signalReal,
                        signalImaginary,
                        chunkCount,
                        fftLength,
                        width,
                        inverse: true);

                    for (int localBatch = 0;
                        localBatch < chunkCount;
                        localBatch++)
                    {
                        AddScaledValues(
                            destination,
                            checked((batchStart + localBatch) * sequence * width),
                            signalReal,
                            localBatch * spectrumElements,
                            1f,
                            checked(sequence * width));
                    }
                }
            }
            finally
            {
                pool.Return(signalReal);
                pool.Return(signalImaginary);
            }
        }
        finally
        {
            pool.Return(filterReal);
            pool.Return(filterImaginary);
        }
    }

    private static void BackwardCausalConvolutionFft(
        float[] gated,
        float[] convolutionGradient,
        int batch,
        int sequence,
        int width,
        TensorStorage filter,
        float[] gatedGradient,
        float[] localFilterGradient)
    {
        int fftLength = GetFftLength(sequence);
        int spectrumElements = checked(fftLength * width);
        ArrayPool<float> pool = ArrayPool<float>.Shared;
        float[] filterReal = pool.Rent(spectrumElements);
        float[] filterImaginary = pool.Rent(spectrumElements);
        try
        {
            filterReal.AsSpan(0, spectrumElements).Clear();
            filterImaginary.AsSpan(0, spectrumElements).Clear();
            filter.CopyRangeTo(
                0,
                filterReal.AsSpan(0, checked(sequence * width)));
            FftChannelsInPlace(
                filterReal,
                filterImaginary,
                transformCount: 1,
                fftLength,
                width,
                inverse: false);

            int chunkCapacity = GetFftChunkSize(
                batch,
                spectrumElements,
                buffersPerTransform: 4);
            int chunkElements = checked(chunkCapacity * spectrumElements);
            float[] gatedReal = pool.Rent(chunkElements);
            float[] gatedImaginary = pool.Rent(chunkElements);
            float[] gradientReal = pool.Rent(chunkElements);
            float[] gradientImaginary = pool.Rent(chunkElements);
            try
            {
                for (int batchStart = 0;
                    batchStart < batch;
                    batchStart += chunkCapacity)
                {
                    int chunkCount = Math.Min(chunkCapacity, batch - batchStart);
                    int activeElements = checked(chunkCount * spectrumElements);
                    gatedReal.AsSpan(0, activeElements).Clear();
                    gatedImaginary.AsSpan(0, activeElements).Clear();
                    for (int localBatch = 0;
                        localBatch < chunkCount;
                        localBatch++)
                    {
                        int sourceOffset = checked(
                            (batchStart + localBatch) * sequence * width);
                        int destinationOffset = localBatch * spectrumElements;
                        Array.Copy(
                            gated,
                            sourceOffset,
                            gatedReal,
                            destinationOffset,
                            checked(sequence * width));
                        Array.Copy(
                            convolutionGradient,
                            sourceOffset,
                            gatedImaginary,
                            destinationOffset,
                            checked(sequence * width));
                    }

                    FftChannelsInPlace(
                        gatedReal,
                        gatedImaginary,
                        chunkCount,
                        fftLength,
                        width,
                        inverse: false);
                    UnpackTwoRealFfts(
                        gatedReal,
                        gatedImaginary,
                        gatedReal,
                        gatedImaginary,
                        gradientReal,
                        gradientImaginary,
                        chunkCount,
                        fftLength,
                        width);

                    MultiplySpectra(
                        gradientReal,
                        gradientImaginary,
                        gatedReal,
                        gatedImaginary,
                        gatedReal,
                        gatedImaginary,
                        chunkCount,
                        fftLength,
                        width,
                        rightShared: false,
                        conjugateRight: true);
                    MultiplySpectra(
                        gradientReal,
                        gradientImaginary,
                        filterReal,
                        filterImaginary,
                        gradientReal,
                        gradientImaginary,
                        chunkCount,
                        fftLength,
                        width,
                        rightShared: true,
                        conjugateRight: true);

                    PackTwoRealSpectra(
                        gatedReal,
                        gatedImaginary,
                        gradientReal,
                        gradientImaginary,
                        chunkCount,
                        fftLength,
                        width);
                    FftChannelsInPlace(
                        gatedReal,
                        gatedImaginary,
                        chunkCount,
                        fftLength,
                        width,
                        inverse: true);

                    for (int localBatch = 0;
                        localBatch < chunkCount;
                        localBatch++)
                    {
                        int destinationOffset = checked(
                            (batchStart + localBatch) * sequence * width);
                        int sourceOffset = localBatch * spectrumElements;
                        AddScaledValues(
                            localFilterGradient,
                            destinationOffset,
                            gatedReal,
                            sourceOffset,
                            1f,
                            checked(sequence * width));
                        AddScaledValues(
                            gatedGradient,
                            destinationOffset,
                            gatedImaginary,
                            sourceOffset,
                            1f,
                            checked(sequence * width));
                    }
                }
            }
            finally
            {
                pool.Return(gatedReal);
                pool.Return(gatedImaginary);
                pool.Return(gradientReal);
                pool.Return(gradientImaginary);
            }
        }
        finally
        {
            pool.Return(filterReal);
            pool.Return(filterImaginary);
        }
    }

    private static void FftChannelsInPlace(
        float[] real,
        float[] imaginary,
        int transformCount,
        int fftLength,
        int width,
        bool inverse)
    {
        FftPlan plan = FftPlans.GetOrAdd(fftLength, static length => new(length));
        int tilesPerTransform = Math.Max(
            1,
            (int)Math.Min(
                width,
                (2L * EffectiveMaxDegreeOfParallelism + transformCount - 1)
                    / transformCount));
        int unalignedTileWidth = (width + tilesPerTransform - 1)
            / tilesPerTransform;
        int tileWidth = Math.Min(
            width,
            Math.Max(32, (unalignedTileWidth + 31) / 32 * 32));
        int tileCount = (width + tileWidth - 1) / tileWidth;
        int workItems = checked(transformCount * tileCount);

        void TransformTile(int workItem)
        {
            int transform = workItem / tileCount;
            int tile = workItem % tileCount;
            int channelStart = tile * tileWidth;
            int channelCount = Math.Min(tileWidth, width - channelStart);
            int transformOffset = checked(transform * fftLength * width);

            for (int index = 0; index < fftLength; index++)
            {
                int reversed = plan.BitReversal[index];
                if (index >= reversed)
                    continue;
                int firstOffset = transformOffset + index * width + channelStart;
                int secondOffset = transformOffset + reversed * width + channelStart;
                SwapFftValues(real, firstOffset, secondOffset, channelCount);
                SwapFftValues(imaginary, firstOffset, secondOffset, channelCount);
            }

            for (int stage = 0; stage < plan.TwiddleReal.Length; stage++)
            {
                float[] twiddleReal = plan.TwiddleReal[stage];
                float[] twiddleImaginary = plan.TwiddleImaginary[stage];
                int half = twiddleReal.Length;
                int size = 2 * half;
                for (int block = 0; block < fftLength; block += size)
                {
                    for (int pair = 0; pair < half; pair++)
                    {
                        float wi = inverse
                            ? -twiddleImaginary[pair]
                            : twiddleImaginary[pair];
                        int upperOffset = transformOffset
                            + (block + pair) * width
                            + channelStart;
                        int lowerOffset = upperOffset + half * width;
                        FftButterfly(
                            real,
                            imaginary,
                            upperOffset,
                            lowerOffset,
                            twiddleReal[pair],
                            wi,
                            channelCount);
                    }
                }
            }

            if (!inverse)
                return;

            float scale = 1f / fftLength;
            for (int index = 0; index < fftLength; index++)
            {
                int offset = transformOffset + index * width + channelStart;
                MultiplyValues(real, offset, scale, real, offset, channelCount);
                MultiplyValues(
                    imaginary,
                    offset,
                    scale,
                    imaginary,
                    offset,
                    channelCount);
            }
        }

        RunBatches(
            workItems,
            (long)fftLength * plan.TwiddleReal.Length * tileWidth,
            TransformTile);
    }

    private static void MultiplySpectra(
        float[] leftReal,
        float[] leftImaginary,
        float[] rightReal,
        float[] rightImaginary,
        float[] destinationReal,
        float[] destinationImaginary,
        int transformCount,
        int fftLength,
        int width,
        bool rightShared,
        bool conjugateRight)
    {
        int rows = checked(transformCount * fftLength);

        void MultiplyRow(int row)
        {
            int leftOffset = row * width;
            int rightOffset = rightShared
                ? row % fftLength * width
                : leftOffset;
            MultiplyComplexValues(
                leftReal,
                leftImaginary,
                leftOffset,
                rightReal,
                rightImaginary,
                rightOffset,
                destinationReal,
                destinationImaginary,
                leftOffset,
                conjugateRight,
                width);
        }

        RunBatches(rows, width, MultiplyRow);
    }

    private static void UnpackTwoRealFfts(
        float[] packedReal,
        float[] packedImaginary,
        float[] firstReal,
        float[] firstImaginary,
        float[] secondReal,
        float[] secondImaginary,
        int transformCount,
        int fftLength,
        int width)
    {
        int uniqueFrequencies = fftLength / 2 + 1;
        int rows = checked(transformCount * uniqueFrequencies);

        void UnpackRow(int row)
        {
            int transform = row / uniqueFrequencies;
            int frequency = row % uniqueFrequencies;
            int mirror = frequency == 0 ? 0 : fftLength - frequency;
            int transformOffset = transform * fftLength * width;
            int frequencyOffset = transformOffset + frequency * width;
            int mirrorOffset = transformOffset + mirror * width;
            UnpackRealFftValues(
                packedReal,
                packedImaginary,
                frequencyOffset,
                mirrorOffset,
                firstReal,
                firstImaginary,
                secondReal,
                secondImaginary,
                frequencyOffset,
                mirrorOffset,
                frequency != mirror,
                width);
        }

        RunBatches(rows, width * 4L, UnpackRow);
    }

    private static void PackTwoRealSpectra(
        float[] firstReal,
        float[] firstImaginary,
        float[] secondReal,
        float[] secondImaginary,
        int transformCount,
        int fftLength,
        int width)
    {
        int rows = checked(transformCount * fftLength);

        void PackRow(int row)
        {
            int offset = row * width;
            int index = 0;
            if (CanUseSimd(width))
            {
                int vectorWidth = Vector256<float>.Count;
                int vectorEnd = width - width % vectorWidth;
                for (; index < vectorEnd; index += vectorWidth)
                {
                    Vector256<float> ar = LoadVector256(
                        firstReal,
                        offset + index);
                    Vector256<float> ai = LoadVector256(
                        firstImaginary,
                        offset + index);
                    Vector256<float> br = LoadVector256(
                        secondReal,
                        offset + index);
                    Vector256<float> bi = LoadVector256(
                        secondImaginary,
                        offset + index);
                    StoreVector256(
                        ar - bi,
                        firstReal,
                        offset + index);
                    StoreVector256(
                        ai + br,
                        firstImaginary,
                        offset + index);
                }
            }

            if (CanUseVector128(width - index))
            {
                Vector128<float> ar = LoadVector128(firstReal, offset + index);
                Vector128<float> ai = LoadVector128(
                    firstImaginary,
                    offset + index);
                Vector128<float> br = LoadVector128(secondReal, offset + index);
                Vector128<float> bi = LoadVector128(
                    secondImaginary,
                    offset + index);
                StoreVector128(ar - bi, firstReal, offset + index);
                StoreVector128(ai + br, firstImaginary, offset + index);
                index += Vector128<float>.Count;
            }

            for (; index < width; index++)
            {
                float ar = firstReal[offset + index];
                float ai = firstImaginary[offset + index];
                float br = secondReal[offset + index];
                float bi = secondImaginary[offset + index];
                firstReal[offset + index] = ar - bi;
                firstImaginary[offset + index] = ai + br;
            }
        }

        RunBatches(rows, width * 2L, PackRow);
    }

    private static void UnpackRealFftValues(
        float[] packedReal,
        float[] packedImaginary,
        int frequencyOffset,
        int mirrorOffset,
        float[] firstReal,
        float[] firstImaginary,
        float[] secondReal,
        float[] secondImaginary,
        int firstDestinationOffset,
        int mirrorDestinationOffset,
        bool hasDistinctMirror,
        int length)
    {
        int index = 0;
        if (CanUseSimd(length))
        {
            int vectorWidth = Vector256<float>.Count;
            int vectorEnd = length - length % vectorWidth;
            Vector256<float> half = Vector256.Create(0.5f);
            for (; index < vectorEnd; index += vectorWidth)
            {
                Vector256<float> zr = LoadVector256(
                    packedReal,
                    frequencyOffset + index);
                Vector256<float> zi = LoadVector256(
                    packedImaginary,
                    frequencyOffset + index);
                Vector256<float> mirrorReal = LoadVector256(
                    packedReal,
                    mirrorOffset + index);
                Vector256<float> mirrorImaginary = LoadVector256(
                    packedImaginary,
                    mirrorOffset + index);
                Vector256<float> ar = (zr + mirrorReal) * half;
                Vector256<float> ai = (zi - mirrorImaginary) * half;
                Vector256<float> br = (zi + mirrorImaginary) * half;
                Vector256<float> bi = (mirrorReal - zr) * half;
                StoreVector256(
                    ar,
                    firstReal,
                    firstDestinationOffset + index);
                StoreVector256(
                    ai,
                    firstImaginary,
                    firstDestinationOffset + index);
                StoreVector256(
                    br,
                    secondReal,
                    firstDestinationOffset + index);
                StoreVector256(
                    bi,
                    secondImaginary,
                    firstDestinationOffset + index);
                if (hasDistinctMirror)
                {
                    StoreVector256(
                        ar,
                        firstReal,
                        mirrorDestinationOffset + index);
                    StoreVector256(
                        Vector256<float>.Zero - ai,
                        firstImaginary,
                        mirrorDestinationOffset + index);
                    StoreVector256(
                        br,
                        secondReal,
                        mirrorDestinationOffset + index);
                    StoreVector256(
                        Vector256<float>.Zero - bi,
                        secondImaginary,
                        mirrorDestinationOffset + index);
                }
            }
        }

        if (CanUseVector128(length - index))
        {
            Vector128<float> zr = LoadVector128(
                packedReal,
                frequencyOffset + index);
            Vector128<float> zi = LoadVector128(
                packedImaginary,
                frequencyOffset + index);
            Vector128<float> mirrorReal = LoadVector128(
                packedReal,
                mirrorOffset + index);
            Vector128<float> mirrorImaginary = LoadVector128(
                packedImaginary,
                mirrorOffset + index);
            Vector128<float> half = Vector128.Create(0.5f);
            Vector128<float> ar = (zr + mirrorReal) * half;
            Vector128<float> ai = (zi - mirrorImaginary) * half;
            Vector128<float> br = (zi + mirrorImaginary) * half;
            Vector128<float> bi = (mirrorReal - zr) * half;
            StoreVector128(
                ar,
                firstReal,
                firstDestinationOffset + index);
            StoreVector128(
                ai,
                firstImaginary,
                firstDestinationOffset + index);
            StoreVector128(
                br,
                secondReal,
                firstDestinationOffset + index);
            StoreVector128(
                bi,
                secondImaginary,
                firstDestinationOffset + index);
            if (hasDistinctMirror)
            {
                StoreVector128(
                    ar,
                    firstReal,
                    mirrorDestinationOffset + index);
                StoreVector128(
                    Vector128<float>.Zero - ai,
                    firstImaginary,
                    mirrorDestinationOffset + index);
                StoreVector128(
                    br,
                    secondReal,
                    mirrorDestinationOffset + index);
                StoreVector128(
                    Vector128<float>.Zero - bi,
                    secondImaginary,
                    mirrorDestinationOffset + index);
            }
            index += Vector128<float>.Count;
        }

        for (; index < length; index++)
        {
            float zr = packedReal[frequencyOffset + index];
            float zi = packedImaginary[frequencyOffset + index];
            float mirrorReal = packedReal[mirrorOffset + index];
            float mirrorImaginary = packedImaginary[mirrorOffset + index];
            float ar = 0.5f * (zr + mirrorReal);
            float ai = 0.5f * (zi - mirrorImaginary);
            float br = 0.5f * (zi + mirrorImaginary);
            float bi = 0.5f * (mirrorReal - zr);
            firstReal[firstDestinationOffset + index] = ar;
            firstImaginary[firstDestinationOffset + index] = ai;
            secondReal[firstDestinationOffset + index] = br;
            secondImaginary[firstDestinationOffset + index] = bi;
            if (hasDistinctMirror)
            {
                firstReal[mirrorDestinationOffset + index] = ar;
                firstImaginary[mirrorDestinationOffset + index] = -ai;
                secondReal[mirrorDestinationOffset + index] = br;
                secondImaginary[mirrorDestinationOffset + index] = -bi;
            }
        }
    }

    private static void MultiplyComplexValues(
        float[] leftReal,
        float[] leftImaginary,
        int leftOffset,
        float[] rightReal,
        float[] rightImaginary,
        int rightOffset,
        float[] destinationReal,
        float[] destinationImaginary,
        int destinationOffset,
        bool conjugateRight,
        int length)
    {
        int index = 0;
        if (CanUseSimd(length))
        {
            int vectorWidth = Vector256<float>.Count;
            int vectorEnd = length - length % vectorWidth;
            for (; index < vectorEnd; index += vectorWidth)
            {
                Vector256<float> ar = LoadVector256(leftReal, leftOffset + index);
                Vector256<float> ai = LoadVector256(leftImaginary, leftOffset + index);
                Vector256<float> br = LoadVector256(rightReal, rightOffset + index);
                Vector256<float> bi = LoadVector256(rightImaginary, rightOffset + index);
                Vector256<float> realProduct;
                Vector256<float> imaginaryProduct;
                if (conjugateRight)
                {
                    realProduct = Vector256.FusedMultiplyAdd(ar, br, ai * bi);
                    imaginaryProduct = Vector256.FusedMultiplyAdd(
                        ai,
                        br,
                        Vector256<float>.Zero - ar * bi);
                }
                else
                {
                    realProduct = Vector256.FusedMultiplyAdd(
                        ar,
                        br,
                        Vector256<float>.Zero - ai * bi);
                    imaginaryProduct = Vector256.FusedMultiplyAdd(ar, bi, ai * br);
                }
                StoreVector256(
                    realProduct,
                    destinationReal,
                    destinationOffset + index);
                StoreVector256(
                    imaginaryProduct,
                    destinationImaginary,
                    destinationOffset + index);
            }
        }

        if (CanUseVector128(length - index))
        {
            Vector128<float> ar = LoadVector128(leftReal, leftOffset + index);
            Vector128<float> ai = LoadVector128(leftImaginary, leftOffset + index);
            Vector128<float> br = LoadVector128(rightReal, rightOffset + index);
            Vector128<float> bi = LoadVector128(rightImaginary, rightOffset + index);
            Vector128<float> realProduct;
            Vector128<float> imaginaryProduct;
            if (conjugateRight)
            {
                realProduct = Vector128.FusedMultiplyAdd(ar, br, ai * bi);
                imaginaryProduct = Vector128.FusedMultiplyAdd(
                    ai,
                    br,
                    Vector128<float>.Zero - ar * bi);
            }
            else
            {
                realProduct = Vector128.FusedMultiplyAdd(
                    ar,
                    br,
                    Vector128<float>.Zero - ai * bi);
                imaginaryProduct = Vector128.FusedMultiplyAdd(ar, bi, ai * br);
            }
            StoreVector128(
                realProduct,
                destinationReal,
                destinationOffset + index);
            StoreVector128(
                imaginaryProduct,
                destinationImaginary,
                destinationOffset + index);
            index += Vector128<float>.Count;
        }

        for (; index < length; index++)
        {
            float ar = leftReal[leftOffset + index];
            float ai = leftImaginary[leftOffset + index];
            float br = rightReal[rightOffset + index];
            float bi = rightImaginary[rightOffset + index];
            destinationReal[destinationOffset + index] = conjugateRight
                ? ar * br + ai * bi
                : ar * br - ai * bi;
            destinationImaginary[destinationOffset + index] = conjugateRight
                ? ai * br - ar * bi
                : ar * bi + ai * br;
        }
    }

    private static void FftButterfly(
        float[] real,
        float[] imaginary,
        int upperOffset,
        int lowerOffset,
        float twiddleReal,
        float twiddleImaginary,
        int length)
    {
        int index = 0;
        if (CanUseSimd(length))
        {
            int vectorWidth = Vector256<float>.Count;
            int vectorEnd = length - length % vectorWidth;
            Vector256<float> wr = Vector256.Create(twiddleReal);
            Vector256<float> wi = Vector256.Create(twiddleImaginary);
            for (; index < vectorEnd; index += vectorWidth)
            {
                Vector256<float> upperReal = LoadVector256(real, upperOffset + index);
                Vector256<float> upperImaginary = LoadVector256(
                    imaginary,
                    upperOffset + index);
                Vector256<float> lowerReal = LoadVector256(real, lowerOffset + index);
                Vector256<float> lowerImaginary = LoadVector256(
                    imaginary,
                    lowerOffset + index);
                Vector256<float> productReal = Vector256.FusedMultiplyAdd(
                    lowerReal,
                    wr,
                    Vector256<float>.Zero - lowerImaginary * wi);
                Vector256<float> productImaginary = Vector256.FusedMultiplyAdd(
                    lowerReal,
                    wi,
                    lowerImaginary * wr);
                StoreVector256(
                    upperReal + productReal,
                    real,
                    upperOffset + index);
                StoreVector256(
                    upperImaginary + productImaginary,
                    imaginary,
                    upperOffset + index);
                StoreVector256(
                    upperReal - productReal,
                    real,
                    lowerOffset + index);
                StoreVector256(
                    upperImaginary - productImaginary,
                    imaginary,
                    lowerOffset + index);
            }
        }

        if (CanUseVector128(length - index))
        {
            Vector128<float> upperReal = LoadVector128(real, upperOffset + index);
            Vector128<float> upperImaginary = LoadVector128(
                imaginary,
                upperOffset + index);
            Vector128<float> lowerReal = LoadVector128(real, lowerOffset + index);
            Vector128<float> lowerImaginary = LoadVector128(
                imaginary,
                lowerOffset + index);
            Vector128<float> wr = Vector128.Create(twiddleReal);
            Vector128<float> wi = Vector128.Create(twiddleImaginary);
            Vector128<float> productReal = Vector128.FusedMultiplyAdd(
                lowerReal,
                wr,
                Vector128<float>.Zero - lowerImaginary * wi);
            Vector128<float> productImaginary = Vector128.FusedMultiplyAdd(
                lowerReal,
                wi,
                lowerImaginary * wr);
            StoreVector128(
                upperReal + productReal,
                real,
                upperOffset + index);
            StoreVector128(
                upperImaginary + productImaginary,
                imaginary,
                upperOffset + index);
            StoreVector128(
                upperReal - productReal,
                real,
                lowerOffset + index);
            StoreVector128(
                upperImaginary - productImaginary,
                imaginary,
                lowerOffset + index);
            index += Vector128<float>.Count;
        }

        for (; index < length; index++)
        {
            float upperReal = real[upperOffset + index];
            float upperImaginary = imaginary[upperOffset + index];
            float lowerReal = real[lowerOffset + index];
            float lowerImaginary = imaginary[lowerOffset + index];
            float productReal = lowerReal * twiddleReal
                - lowerImaginary * twiddleImaginary;
            float productImaginary = lowerReal * twiddleImaginary
                + lowerImaginary * twiddleReal;
            real[upperOffset + index] = upperReal + productReal;
            imaginary[upperOffset + index] = upperImaginary + productImaginary;
            real[lowerOffset + index] = upperReal - productReal;
            imaginary[lowerOffset + index] = upperImaginary - productImaginary;
        }
    }

    private static void SwapFftValues(
        float[] values,
        int firstOffset,
        int secondOffset,
        int length)
    {
        int index = 0;
        if (CanUseSimd(length))
        {
            int vectorWidth = Vector256<float>.Count;
            int vectorEnd = length - length % vectorWidth;
            for (; index < vectorEnd; index += vectorWidth)
            {
                Vector256<float> first = LoadVector256(values, firstOffset + index);
                Vector256<float> second = LoadVector256(values, secondOffset + index);
                StoreVector256(second, values, firstOffset + index);
                StoreVector256(first, values, secondOffset + index);
            }
        }

        if (CanUseVector128(length - index))
        {
            Vector128<float> first = LoadVector128(values, firstOffset + index);
            Vector128<float> second = LoadVector128(values, secondOffset + index);
            StoreVector128(second, values, firstOffset + index);
            StoreVector128(first, values, secondOffset + index);
            index += Vector128<float>.Count;
        }

        for (; index < length; index++)
        {
            (values[firstOffset + index], values[secondOffset + index]) =
                (values[secondOffset + index], values[firstOffset + index]);
        }
    }

    private sealed class FftPlan
    {
        internal FftPlan(int length)
        {
            if (length <= 0 || (length & (length - 1)) != 0)
                throw new ArgumentOutOfRangeException(nameof(length));

            BitReversal = new int[length];
            int bits = int.Log2(length);
            for (int index = 0; index < length; index++)
                BitReversal[index] = ReverseBits(index, bits);

            TwiddleReal = new float[bits][];
            TwiddleImaginary = new float[bits][];
            for (int stage = 0, size = 2;
                stage < bits;
                stage++, size *= 2)
            {
                int half = size / 2;
                TwiddleReal[stage] = new float[half];
                TwiddleImaginary[stage] = new float[half];
                for (int pair = 0; pair < half; pair++)
                {
                    float angle = -2f * MathF.PI * pair / size;
                    (float sine, float cosine) = MathF.SinCos(angle);
                    TwiddleReal[stage][pair] = cosine;
                    TwiddleImaginary[stage][pair] = sine;
                }
            }
        }

        internal int[] BitReversal { get; }

        internal float[][] TwiddleReal { get; }

        internal float[][] TwiddleImaginary { get; }

        private static int ReverseBits(int value, int bits)
        {
            int reversed = 0;
            for (int bit = 0; bit < bits; bit++)
            {
                reversed = (reversed << 1) | (value & 1);
                value >>= 1;
            }
            return reversed;
        }
    }
}
