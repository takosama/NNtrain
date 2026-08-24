namespace NNtrain;

partial class Tensor
{
    /// <summary>
    /// Looks up rows in a rank-2 embedding table and appends the embedding
    /// width to the requested output shape.
    /// </summary>
    public Tensor EmbeddingLookup(int[] indices, params int[] outputShape)
    {
        ArgumentNullException.ThrowIfNull(indices);
        ArgumentNullException.ThrowIfNull(outputShape);
        if (Rank != 2)
        {
            throw new InvalidOperationException(
                "EmbeddingLookup requires a rank-2 embedding table.");
        }
        if (outputShape.Length == 0)
        {
            throw new ArgumentException(
                "Embedding output shape must contain at least one dimension.",
                nameof(outputShape));
        }

        int indexCount = 1;
        foreach (int dimension in outputShape)
        {
            if (dimension <= 0)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(outputShape),
                    dimension,
                    "Embedding output dimensions must be positive.");
            }
            indexCount = checked(indexCount * dimension);
        }
        if (indexCount != indices.Length)
        {
            throw new ArgumentException(
                $"Embedding output shape contains {indexCount} positions, " +
                $"but {indices.Length} indices were supplied.",
                nameof(outputShape));
        }

        int rows = _shape[0];
        int width = _shape[1];
        int[] retainedIndices = (int[])indices.Clone();
        var resultShape = new int[outputShape.Length + 1];
        outputShape.CopyTo(resultShape, 0);
        resultShape[^1] = width;
        for (int position = 0; position < retainedIndices.Length; position++)
        {
            int row = retainedIndices[position];
            if ((uint)row >= (uint)rows)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(indices),
                    row,
                    $"Embedding index at position {position} must be " +
                    $"between 0 and {rows - 1}.");
            }
        }
        if (ExecutionDevice == TensorDevice.Cuda)
        {
            if (DType == TensorDType.BFloat16)
            {
                var bfloat16Context = CudaOperationProfiler.IsEnabled
                    ? CudaOperationProfiler.Measure(
                        "forward.embedding",
                        () => TensorCudaKernels.EmbeddingForwardBFloat16Resident(
                            this,
                            retainedIndices,
                            width))
                    : TensorCudaKernels.EmbeddingForwardBFloat16Resident(
                        this,
                        retainedIndices,
                        width);
                Tensor bfloat16Result = FromCudaResult(
                    bfloat16Context.Output,
                    CudaDeviceIndex,
                    resultShape,
                    [this],
                    TensorDType.BFloat16);
                if (AutogradContext.IsRecordingEnabled)
                {
                    bfloat16Result.Node.RegisterResource(bfloat16Context);
                    bfloat16Result.Node.BackwardAction = () =>
                    {
                        if (CudaOperationProfiler.IsEnabled)
                        {
                            CudaOperationProfiler.Measure(
                                "backward.embedding",
                                () => TensorCudaKernels
                                    .EmbeddingBackwardBFloat16Resident(
                                        bfloat16Result,
                                        this,
                                        bfloat16Context,
                                        width));
                        }
                        else
                        {
                            TensorCudaKernels.EmbeddingBackwardBFloat16Resident(
                                bfloat16Result,
                                this,
                                bfloat16Context,
                                width);
                        }
                    };
                }
                else if (!CudaInferenceScope.TrackResource(bfloat16Context))
                {
                    bfloat16Context.Dispose();
                }
                return bfloat16Result;
            }
            var cudaBuffer = CudaOperationProfiler.IsEnabled
                ? CudaOperationProfiler.Measure(
                    "forward.embedding",
                    () => TensorCudaKernels.EmbeddingForwardResident(
                        this,
                        retainedIndices,
                        width))
                : TensorCudaKernels.EmbeddingForwardResident(
                    this,
                    retainedIndices,
                    width);
            Tensor cudaResult = FromCudaResult(
                cudaBuffer,
                CudaDeviceIndex,
                resultShape,
                [this]);
            if (AutogradContext.IsRecordingEnabled)
            {
                cudaResult.Node.BackwardAction = () =>
                {
                    if (CudaOperationProfiler.IsEnabled)
                    {
                        CudaOperationProfiler.Measure(
                            "backward.embedding",
                            () => TensorCudaKernels.EmbeddingBackwardResident(
                                cudaResult,
                                this,
                                retainedIndices,
                                width));
                    }
                    else
                    {
                        TensorCudaKernels.EmbeddingBackwardResident(
                            cudaResult,
                            this,
                            retainedIndices,
                            width);
                    }
                };
            }
            return cudaResult;
        }
        var resultData = new float[checked(indices.Length * width)];
        for (int position = 0; position < retainedIndices.Length; position++)
        {
            int row = retainedIndices[position];
            _data.CopyRangeTo(
                row * width,
                resultData.AsSpan(position * width, width));
        }

        EmbeddingGradientGroups gradientGroups =
            CreateEmbeddingGradientGroups(retainedIndices, rows);

        var result = new Tensor(resultData, resultShape, [this]);
        result.Node.BackwardAction = () =>
        {
            AccumulateEmbeddingGradients(
                _grad,
                result._grad,
                width,
                gradientGroups);
        };
        return result;
    }

    /// <summary>
    /// Fuses token lookup, position lookup, and their element-wise addition
    /// into one training operation shaped [batch, sequence, width].
    /// </summary>
    public Tensor EmbeddingLookupWithPositions(
        Tensor positionTable,
        int[] tokenIndices,
        int batchSize,
        int sequenceLength)
    {
        ArgumentNullException.ThrowIfNull(positionTable);
        ArgumentNullException.ThrowIfNull(tokenIndices);
        CheckRank(2);
        positionTable.CheckRank(2);
        if (ReferenceEquals(this, positionTable))
        {
            throw new ArgumentException(
                "Token and position embedding tables must be distinct.",
                nameof(positionTable));
        }
        if (batchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(batchSize));
        if (sequenceLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(sequenceLength));
        if (tokenIndices.Length != checked(batchSize * sequenceLength))
        {
            throw new ArgumentException(
                "Token count must equal batchSize * sequenceLength.",
                nameof(tokenIndices));
        }

        int tokenRows = _shape[0];
        int width = _shape[1];
        if (positionTable._shape[1] != width)
        {
            throw new ArgumentException(
                "Token and position embedding widths must match.",
                nameof(positionTable));
        }
        if (sequenceLength > positionTable._shape[0])
        {
            throw new ArgumentOutOfRangeException(
                nameof(sequenceLength),
                sequenceLength,
                "Sequence length exceeds the position embedding table.");
        }

        int[] retainedIndices = (int[])tokenIndices.Clone();
        for (int position = 0; position < retainedIndices.Length; position++)
        {
            int token = retainedIndices[position];
            if ((uint)token >= (uint)tokenRows)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(tokenIndices),
                    token,
                    $"Token at position {position} must be between 0 and " +
                    $"{tokenRows - 1}.");
            }
        }

        if (ExecutionDevice == TensorDevice.Cuda)
        {
            if (DType == TensorDType.BFloat16
                && positionTable.DType == TensorDType.BFloat16)
            {
                var bfloat16Context = CudaOperationProfiler.IsEnabled
                    ? CudaOperationProfiler.Measure(
                        "forward.embedding_position",
                        () => TensorCudaKernels
                            .EmbeddingWithPositionsForwardBFloat16Resident(
                                this,
                                positionTable,
                                retainedIndices,
                                sequenceLength,
                                width))
                    : TensorCudaKernels
                        .EmbeddingWithPositionsForwardBFloat16Resident(
                            this,
                            positionTable,
                            retainedIndices,
                            sequenceLength,
                            width);
                Tensor bfloat16Result = FromCudaResult(
                    bfloat16Context.Output,
                    CudaDeviceIndex,
                    [batchSize, sequenceLength, width],
                    [this, positionTable],
                    TensorDType.BFloat16);
                if (AutogradContext.IsRecordingEnabled)
                {
                    bfloat16Result.Node.RegisterResource(bfloat16Context);
                    bfloat16Result.Node.BackwardAction = () =>
                    {
                        if (CudaOperationProfiler.IsEnabled)
                        {
                            CudaOperationProfiler.Measure(
                                "backward.embedding_position",
                                () => TensorCudaKernels
                                    .EmbeddingWithPositionsBackwardBFloat16Resident(
                                        bfloat16Result,
                                        this,
                                        positionTable,
                                        bfloat16Context,
                                        sequenceLength,
                                        width));
                        }
                        else
                        {
                            TensorCudaKernels
                                .EmbeddingWithPositionsBackwardBFloat16Resident(
                                    bfloat16Result,
                                    this,
                                    positionTable,
                                    bfloat16Context,
                                    sequenceLength,
                                    width);
                        }
                    };
                }
                else if (!CudaInferenceScope.TrackResource(bfloat16Context))
                {
                    bfloat16Context.Dispose();
                }
                return bfloat16Result;
            }
            var context = CudaOperationProfiler.IsEnabled
                ? CudaOperationProfiler.Measure(
                    "forward.embedding_position",
                    () => TensorCudaKernels.EmbeddingWithPositionsForwardResident(
                        this,
                        positionTable,
                        retainedIndices,
                        sequenceLength,
                        width))
                : TensorCudaKernels.EmbeddingWithPositionsForwardResident(
                    this,
                    positionTable,
                    retainedIndices,
                    sequenceLength,
                    width);
            Tensor cudaResult = FromCudaResult(
                context.Output,
                CudaDeviceIndex,
                [batchSize, sequenceLength, width],
                [this, positionTable]);
            if (AutogradContext.IsRecordingEnabled)
            {
                cudaResult.Node.RegisterResource(context);
                cudaResult.Node.BackwardAction = () =>
                {
                    if (CudaOperationProfiler.IsEnabled)
                    {
                        CudaOperationProfiler.Measure(
                            "backward.embedding_position",
                            () => TensorCudaKernels.EmbeddingWithPositionsBackwardResident(
                                cudaResult,
                                this,
                                positionTable,
                                context,
                                sequenceLength,
                                width));
                    }
                    else
                    {
                        TensorCudaKernels.EmbeddingWithPositionsBackwardResident(
                            cudaResult,
                            this,
                            positionTable,
                            context,
                            sequenceLength,
                            width);
                    }
                };
            }
            else
            {
                if (!CudaInferenceScope.TrackResource(context))
                    context.Dispose();
            }
            return cudaResult;
        }

        var output = new float[checked(retainedIndices.Length * width)];
        void ForwardPosition(int position)
        {
            AddValues(
                _data,
                retainedIndices[position] * width,
                positionTable._data,
                position % sequenceLength * width,
                output,
                position * width,
                width);
        }

        RunBatches(retainedIndices.Length, width, ForwardPosition);
        EmbeddingGradientGroups tokenGradientGroups =
            CreateEmbeddingGradientGroups(retainedIndices, tokenRows);
        var result = new Tensor(
            output,
            [batchSize, sequenceLength, width],
            [this, positionTable]);
        result.Node.BackwardAction = () =>
        {
            AccumulateEmbeddingGradients(
                _grad,
                result._grad,
                width,
                tokenGradientGroups);

            void AccumulatePosition(int position)
            {
                int destinationOffset = position * width;
                for (int batch = 0; batch < batchSize; batch++)
                {
                    int sourceOffset =
                        (batch * sequenceLength + position) * width;
                    AddScaledValues(
                        positionTable._grad,
                        destinationOffset,
                        result._grad,
                        sourceOffset,
                        1f,
                        width);
                }
            }

            RunBatches(
                sequenceLength,
                (long)batchSize * width,
                AccumulatePosition);
        };
        return result;
    }

    private static EmbeddingGradientGroups CreateEmbeddingGradientGroups(
        int[] indices,
        int rowCount)
    {
        var counts = new int[rowCount];
        int uniqueCount = 0;
        foreach (int row in indices)
        {
            if (counts[row]++ == 0)
                uniqueCount++;
        }

        var rows = new int[uniqueCount];
        var offsets = new int[uniqueCount + 1];
        var rowToGroup = new int[rowCount];
        Array.Fill(rowToGroup, -1);
        int group = 0;
        for (int row = 0; row < rowCount; row++)
        {
            if (counts[row] == 0)
                continue;
            rows[group] = row;
            rowToGroup[row] = group;
            offsets[group + 1] = offsets[group] + counts[row];
            group++;
        }

        var positions = new int[indices.Length];
        var cursors = (int[])offsets.Clone();
        for (int position = 0; position < indices.Length; position++)
        {
            int targetGroup = rowToGroup[indices[position]];
            positions[cursors[targetGroup]++] = position;
        }
        return new EmbeddingGradientGroups(rows, offsets, positions);
    }

    private static void AccumulateEmbeddingGradients(
        float[] destination,
        float[] source,
        int width,
        EmbeddingGradientGroups groups)
    {
        void AccumulateGroup(int group)
        {
            int destinationOffset = groups.Rows[group] * width;
            for (int index = groups.Offsets[group];
                index < groups.Offsets[group + 1];
                index++)
            {
                AddScaledValues(
                    destination,
                    destinationOffset,
                    source,
                    groups.Positions[index] * width,
                    1f,
                    width);
            }
        }

        long workPerGroup = groups.Rows.Length == 0
            ? 0
            : (long)groups.Positions.Length * width / groups.Rows.Length;
        RunBatches(groups.Rows.Length, workPerGroup, AccumulateGroup);
    }

    private sealed record EmbeddingGradientGroups(
        int[] Rows,
        int[] Offsets,
        int[] Positions);
}
