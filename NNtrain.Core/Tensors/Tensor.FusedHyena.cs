using System.Buffers;

namespace NNtrain;

partial class Tensor
{
    private const int HyenaFftTrainingThreshold = 1024;
    private const int HyenaFftInferenceThreshold = 2048;

    /// <summary>
    /// Applies the order-2 causal Hyena recurrence to a projected tensor.
    /// </summary>
    /// <remarks>
    /// The projection is arranged as three width-sized streams. A causal
    /// depthwise width-3 filter produces two gates and one value stream. The
    /// second gate controls a causal long convolution and the first gate
    /// controls its output. The implementation is fused so its intermediate
    /// tensors stay contiguous and channel work can use SIMD.
    /// </remarks>
    public Tensor FusedCausalHyenaOrder2(
        Tensor shortFilter,
        Tensor longFilter,
        Tensor diagonalBias,
        HyenaConvolutionAlgorithm convolutionAlgorithm =
            HyenaConvolutionAlgorithm.Auto)
    {
        ArgumentNullException.ThrowIfNull(shortFilter);
        ArgumentNullException.ThrowIfNull(longFilter);
        ArgumentNullException.ThrowIfNull(diagonalBias);
        CheckRank(3);
        shortFilter.CheckRank(2);
        longFilter.CheckRank(2);
        diagonalBias.CheckRank(1);

        int batch = _shape[0];
        int sequence = _shape[1];
        int channels = _shape[2];
        if (channels % 3 != 0)
        {
            throw new ArgumentException(
                "Hyena projection width must be three times the model width.");
        }

        int width = channels / 3;
        if (shortFilter._shape[0] != 3
            || shortFilter._shape[1] != channels)
        {
            throw new ArgumentException(
                "Hyena short filter must have shape [3, 3 * width].",
                nameof(shortFilter));
        }
        if (longFilter._shape[0] != sequence
            || longFilter._shape[1] != width)
        {
            throw new ArgumentException(
                "Hyena long filter must have shape [sequence, width].",
                nameof(longFilter));
        }
        if (diagonalBias._shape[0] != width)
        {
            throw new ArgumentException(
                "Hyena diagonal bias must have shape [width].",
                nameof(diagonalBias));
        }
        if (!Enum.IsDefined(convolutionAlgorithm))
        {
            throw new ArgumentOutOfRangeException(
                nameof(convolutionAlgorithm),
                convolutionAlgorithm,
                "Unknown Hyena convolution algorithm.");
        }

        bool useFft = convolutionAlgorithm switch
        {
            HyenaConvolutionAlgorithm.Fft => true,
            HyenaConvolutionAlgorithm.Direct => false,
            _ => sequence >= (AutogradContext.IsRecordingEnabled
                ? HyenaFftTrainingThreshold
                : HyenaFftInferenceThreshold),
        };

        int rows = checked(batch * sequence);
        var shortOutput = new float[checked(rows * channels)];
        var gated = new float[checked(rows * width)];
        var convolved = new float[checked(rows * width)];
        var output = AutogradContext.IsRecordingEnabled
            ? new float[checked(rows * width)]
            : convolved;

        void ForwardShortRow(int row)
        {
            int time = row % sequence;
            int batchIndex = row / sequence;
            int shortOffset = row * channels;
            int inputOffset =
                (batchIndex * sequence + time) * channels;
            AddStridedElementwiseProductSum(
                shortOutput,
                shortOffset,
                _data,
                inputOffset,
                -channels,
                shortFilter._data,
                0,
                channels,
                Math.Min(3, time + 1),
                channels);

            int x1Offset = shortOffset + width;
            int valueOffset = shortOffset + 2 * width;
            int mixedOffset = row * width;
            MultiplyElementwiseValues(
                shortOutput,
                valueOffset,
                shortOutput,
                x1Offset,
                gated,
                mixedOffset,
                width);
        }

        RunBatches(rows, (long)channels * 3, ForwardShortRow);

        void InitializeConvolutionRow(int row)
        {
            int mixedOffset = row * width;
            MultiplyElementwiseValues(
                gated,
                mixedOffset,
                diagonalBias._data,
                0,
                convolved,
                mixedOffset,
                width);
        }

        void FinishConvolutionRow(int row)
        {
            int shortOffset = row * channels;
            int mixedOffset = row * width;
            MultiplyElementwiseValues(
                shortOutput,
                shortOffset,
                convolved,
                mixedOffset,
                output,
                mixedOffset,
                width);
        }

        if (useFft)
        {
            RunBatches(rows, width, InitializeConvolutionRow);
            AddCausalConvolutionFft(
                gated,
                batch,
                sequence,
                width,
                longFilter._data,
                convolved);
            RunBatches(rows, width, FinishConvolutionRow);
        }
        else
        {
            void ForwardConvolutionRow(int row)
            {
                int time = row % sequence;
                int mixedOffset = row * width;
                InitializeConvolutionRow(row);
                AddStridedElementwiseProductSum(
                    convolved,
                    mixedOffset,
                    gated,
                    mixedOffset,
                    -width,
                    longFilter._data,
                    0,
                    width,
                    time + 1,
                    width);
                FinishConvolutionRow(row);
            }

            RunBatches(
                rows,
                (long)sequence * width,
                ForwardConvolutionRow);
        }

        var result = new Tensor(
            output,
            [batch, sequence, width],
            [this, shortFilter, longFilter, diagonalBias]);
        result.Node.BackwardAction = () =>
        {
            int shortGradientLength = shortOutput.Length;
            int mixedGradientLength = gated.Length;
            int localLongFilterGradientLength =
                checked(batch * sequence * width);
            int localDiagonalGradientLength = checked(batch * width);
            int localShortFilterGradientLength =
                checked(batch * 3 * channels);
            ArrayPool<float> pool = ArrayPool<float>.Shared;
            float[] shortGradient = pool.Rent(shortGradientLength);
            float[] convolutionGradient = pool.Rent(mixedGradientLength);
            float[] gatedGradient = pool.Rent(mixedGradientLength);
            float[] localLongFilterGradient =
                pool.Rent(localLongFilterGradientLength);
            float[] localDiagonalGradient =
                pool.Rent(localDiagonalGradientLength);
            float[] localShortFilterGradient =
                pool.Rent(localShortFilterGradientLength);

            shortGradient.AsSpan(0, shortGradientLength).Clear();
            convolutionGradient.AsSpan(0, mixedGradientLength).Clear();
            gatedGradient.AsSpan(0, mixedGradientLength).Clear();
            localLongFilterGradient
                .AsSpan(0, localLongFilterGradientLength)
                .Clear();
            localDiagonalGradient
                .AsSpan(0, localDiagonalGradientLength)
                .Clear();
            localShortFilterGradient
                .AsSpan(0, localShortFilterGradientLength)
                .Clear();

            try
            {

                void BackwardOutputRow(int row)
                {
                    int shortOffset = row * channels;
                    int mixedOffset = row * width;
                    MultiplyElementwiseValues(
                        result._grad,
                        mixedOffset,
                        convolved,
                        mixedOffset,
                        shortGradient,
                        shortOffset,
                        width);
                    MultiplyElementwiseValues(
                        result._grad,
                        mixedOffset,
                        shortOutput,
                        shortOffset,
                        convolutionGradient,
                        mixedOffset,
                        width);
                    MultiplyElementwiseValues(
                        convolutionGradient,
                        mixedOffset,
                        diagonalBias._data,
                        0,
                        gatedGradient,
                        mixedOffset,
                        width);
                }

                RunBatches(rows, width, BackwardOutputRow);

                if (useFft)
                {
                    void AccumulateDiagonalBatch(int batchIndex)
                    {
                        int batchMixedOffset = batchIndex * sequence * width;
                        AddStridedElementwiseProductSum(
                            localDiagonalGradient,
                            batchIndex * width,
                            convolutionGradient,
                            batchMixedOffset,
                            width,
                            gated,
                            batchMixedOffset,
                            width,
                            sequence,
                            width);
                    }

                    RunBatches(
                        batch,
                        (long)sequence * width,
                        AccumulateDiagonalBatch);
                    BackwardCausalConvolutionFft(
                        gated,
                        convolutionGradient,
                        batch,
                        sequence,
                        width,
                        longFilter._data,
                        gatedGradient,
                        localLongFilterGradient);
                }
                else
                {
                    int convolutionTilesPerBatch = Math.Max(
                        1,
                        (int)Math.Min(
                            width,
                            (2L * EffectiveMaxDegreeOfParallelism + batch - 1)
                                / batch));
                    int unalignedConvolutionTileWidth =
                        (width + convolutionTilesPerBatch - 1)
                        / convolutionTilesPerBatch;
                    int convolutionTileWidth = Math.Min(
                        width,
                        Math.Max(
                            32,
                            (unalignedConvolutionTileWidth + 31) / 32 * 32));
                    int convolutionTiles =
                        (width + convolutionTileWidth - 1) / convolutionTileWidth;
                    int convolutionWorkItems = checked(batch * convolutionTiles);

                    void BackwardConvolutionTile(int workItem)
                    {
                        int batchIndex = workItem / convolutionTiles;
                        int tile = workItem % convolutionTiles;
                        int channelStart = tile * convolutionTileWidth;
                        int channelCount = Math.Min(
                            convolutionTileWidth,
                            width - channelStart);
                        int batchMixedOffset = batchIndex * sequence * width;
                        int localFilterBase = batchIndex * sequence * width;
                        int localDiagonalOffset = batchIndex * width;
                        AddStridedElementwiseProductSum(
                            localDiagonalGradient,
                            localDiagonalOffset + channelStart,
                            convolutionGradient,
                            batchMixedOffset + channelStart,
                            width,
                            gated,
                            batchMixedOffset + channelStart,
                            width,
                            sequence,
                            channelCount);

                        for (int sourceTime = 0;
                            sourceTime < sequence;
                            sourceTime++)
                        {
                            AddStridedElementwiseProductSum(
                                gatedGradient,
                                batchMixedOffset
                                    + sourceTime * width
                                    + channelStart,
                                convolutionGradient,
                                batchMixedOffset
                                    + sourceTime * width
                                    + channelStart,
                                width,
                                longFilter._data,
                                channelStart,
                                width,
                                sequence - sourceTime,
                                channelCount);
                        }

                        for (int lag = 0; lag < sequence; lag++)
                        {
                            AddStridedElementwiseProductSum(
                                localLongFilterGradient,
                                localFilterBase + lag * width + channelStart,
                                convolutionGradient,
                                batchMixedOffset + lag * width + channelStart,
                                width,
                                gated,
                                batchMixedOffset + channelStart,
                                width,
                                sequence - lag,
                                channelCount);
                        }
                    }

                    RunBatches(
                        convolutionWorkItems,
                        (long)sequence * sequence * convolutionTileWidth,
                        BackwardConvolutionTile);
                }

                void ReduceLongFilterLag(int lag)
                {
                    int filterOffset = lag * width;
                    for (int batchIndex = 0; batchIndex < batch; batchIndex++)
                    {
                        AddScaledValues(
                            longFilter._grad,
                            filterOffset,
                            localLongFilterGradient,
                            (batchIndex * sequence + lag) * width,
                            1f,
                            width);
                    }
                }

                RunBatches(
                    sequence,
                    (long)batch * width,
                    ReduceLongFilterLag);

                for (int batchIndex = 0; batchIndex < batch; batchIndex++)
                {
                    AddScaledValues(
                        diagonalBias._grad,
                        0,
                        localDiagonalGradient,
                        batchIndex * width,
                        1f,
                        width);
                }

                void BackwardGateRow(int row)
                {
                    int shortOffset = row * channels;
                    int mixedOffset = row * width;
                    int x1Offset = shortOffset + width;
                    int valueOffset = shortOffset + 2 * width;
                    MultiplyElementwiseValues(
                        gatedGradient,
                        mixedOffset,
                        shortOutput,
                        valueOffset,
                        shortGradient,
                        x1Offset,
                        width);
                    MultiplyElementwiseValues(
                        gatedGradient,
                        mixedOffset,
                        shortOutput,
                        x1Offset,
                        shortGradient,
                        valueOffset,
                        width);
                }

                RunBatches(rows, width, BackwardGateRow);

                int shortTilesPerBatch = Math.Max(
                    1,
                    (int)Math.Min(
                        channels,
                        (2L * EffectiveMaxDegreeOfParallelism + batch - 1)
                            / batch));
                int unalignedShortTileWidth =
                    (channels + shortTilesPerBatch - 1) / shortTilesPerBatch;
                int shortTileWidth = Math.Min(
                    channels,
                    Math.Max(
                        32,
                        (unalignedShortTileWidth + 31) / 32 * 32));
                int shortTiles =
                    (channels + shortTileWidth - 1) / shortTileWidth;
                int shortWorkItems = checked(batch * shortTiles);

                void BackwardShortTile(int workItem)
                {
                    int batchIndex = workItem / shortTiles;
                    int tile = workItem % shortTiles;
                    int channelStart = tile * shortTileWidth;
                    int channelCount = Math.Min(
                        shortTileWidth,
                        channels - channelStart);
                    int batchShortOffset = batchIndex * sequence * channels;
                    int localFilterBase = batchIndex * 3 * channels;
                    for (int inputTime = 0;
                        inputTime < sequence;
                        inputTime++)
                    {
                        AddStridedElementwiseProductSum(
                            _grad,
                            batchShortOffset
                                + inputTime * channels
                                + channelStart,
                            shortGradient,
                            batchShortOffset
                                + inputTime * channels
                                + channelStart,
                            channels,
                            shortFilter._data,
                            channelStart,
                            channels,
                            Math.Min(3, sequence - inputTime),
                            channelCount);
                    }

                    for (int tap = 0; tap < 3; tap++)
                    {
                        AddStridedElementwiseProductSum(
                            localShortFilterGradient,
                            localFilterBase + tap * channels + channelStart,
                            shortGradient,
                            batchShortOffset + tap * channels + channelStart,
                            channels,
                            _data,
                            batchShortOffset + channelStart,
                            channels,
                            sequence - tap,
                            channelCount);
                    }
                }

                RunBatches(
                    shortWorkItems,
                    (long)sequence * shortTileWidth * 3,
                    BackwardShortTile);

                void ReduceShortFilterTap(int tap)
                {
                    int filterOffset = tap * channels;
                    for (int batchIndex = 0; batchIndex < batch; batchIndex++)
                    {
                        AddScaledValues(
                            shortFilter._grad,
                            filterOffset,
                            localShortFilterGradient,
                            (batchIndex * 3 + tap) * channels,
                            1f,
                            channels);
                    }
                }

                RunBatches(
                    3,
                    (long)batch * channels,
                    ReduceShortFilterTap);
            }
            finally
            {
                pool.Return(shortGradient);
                pool.Return(convolutionGradient);
                pool.Return(gatedGradient);
                pool.Return(localLongFilterGradient);
                pool.Return(localDiagonalGradient);
                pool.Return(localShortFilterGradient);
            }
        };

        return result;
    }
}
