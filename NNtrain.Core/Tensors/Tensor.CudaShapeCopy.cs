namespace NNtrain;

partial class Tensor
{
    private readonly record struct CudaCopySegment(
        int SourceIndex,
        int SourceOffset,
        int DestinationOffset,
        int Length);

    private Tensor SliceCuda(int dim, int start, int length)
    {
        if (Rank is < 1 or > 3)
            throw new NotSupportedException("Slice supports rank1/rank2/rank3 only");
        if ((uint)dim >= (uint)Rank)
            throw new ArgumentOutOfRangeException(nameof(dim));
        ValidateSliceRange(_shape[dim], start, length);

        int inner = 1;
        for (int index = dim + 1; index < Rank; index++)
            inner = checked(inner * _shape[index]);
        int outer = 1;
        for (int index = 0; index < dim; index++)
            outer = checked(outer * _shape[index]);
        var segments = new CudaCopySegment[outer];
        for (int outerIndex = 0; outerIndex < outer; outerIndex++)
        {
            segments[outerIndex] = new CudaCopySegment(
                SourceIndex: 0,
                SourceOffset: checked(
                    (outerIndex * _shape[dim] + start) * inner),
                DestinationOffset: checked(outerIndex * length * inner),
                Length: checked(length * inner));
        }

        int[] outputShape = (int[])_shape.Clone();
        outputShape[dim] = length;
        return CopySegmentsCuda([this], segments, outputShape);
    }

    private static Tensor ConcatCuda(int dim, Tensor[] tensors)
    {
        int rank = tensors[0].Rank;
        if (rank is < 1 or > 3)
            throw new NotSupportedException("Concat supports rank1/rank2/rank3 only");
        if ((uint)dim >= (uint)rank)
            throw new ArgumentOutOfRangeException(nameof(dim));
        TensorDType dtype = tensors[0].DType;
        if (dtype is not TensorDType.Float32
            and not TensorDType.BFloat16
            and not TensorDType.Bfp8
            || tensors.Any(tensor => tensor.DType != dtype))
        {
            ThrowIfCudaHostFallback(nameof(Concat));
        }

        int[] outputShape = tensors[0]._shape.ToArray();
        int totalDimension = 0;
        foreach (Tensor tensor in tensors)
        {
            for (int axis = 0; axis < rank; axis++)
            {
                if (axis != dim && tensor._shape[axis] != outputShape[axis])
                {
                    throw new ArgumentException(
                        "All tensors passed to Concat must match outside the " +
                        "concatenated dimension.",
                        nameof(tensors));
                }
            }
            totalDimension = checked(totalDimension + tensor._shape[dim]);
        }
        outputShape[dim] = totalDimension;

        int inner = 1;
        for (int index = dim + 1; index < rank; index++)
            inner = checked(inner * outputShape[index]);
        int outer = 1;
        for (int index = 0; index < dim; index++)
            outer = checked(outer * outputShape[index]);
        var segments = new List<CudaCopySegment>(checked(outer * tensors.Length));
        for (int outerIndex = 0; outerIndex < outer; outerIndex++)
        {
            int destinationDimensionOffset = 0;
            for (int sourceIndex = 0; sourceIndex < tensors.Length; sourceIndex++)
            {
                Tensor tensor = tensors[sourceIndex];
                int copyLength = checked(tensor._shape[dim] * inner);
                segments.Add(new CudaCopySegment(
                    sourceIndex,
                    SourceOffset: checked(outerIndex * copyLength),
                    DestinationOffset: checked(
                        (outerIndex * totalDimension
                            + destinationDimensionOffset) * inner),
                    copyLength));
                destinationDimensionOffset += tensor._shape[dim];
            }
        }
        return CopySegmentsCuda(tensors, segments, outputShape);
    }

    private static Tensor CopySegmentsCuda(
        Tensor[] sources,
        IReadOnlyList<CudaCopySegment> segments,
        int[] outputShape)
    {
        int deviceIndex = CudaDeviceIndex;
        int outputLength = NumelOf(outputShape);
        TensorDType dtype = sources[0].DType;
        Tensor result;
        if (dtype == TensorDType.Bfp8)
        {
            Bfp8QuantizationDescriptor descriptor =
                SelectBfp8ResultDescriptor(sources);
            NativeCudaDevice accelerator =
                ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
            NativeCudaBuffer<ushort> decodedOutput =
                RentCudaBFloat16Buffer(deviceIndex, outputLength);
            var leases = new CudaBfp8BFloat16Lease?[sources.Length];
            using CudaBfp8OwnedBuffers output = CudaBfp8OwnedBuffers.Allocate(
                accelerator,
                outputLength,
                descriptor);
            try
            {
                for (int index = 0; index < sources.Length; index++)
                {
                    leases[index] = sources[index]
                        .AcquireCudaBfp8BFloat16Buffer(deviceIndex);
                }
                foreach (CudaCopySegment segment in segments)
                {
                    leases[segment.SourceIndex]!.Buffer.View
                        .SubView(segment.SourceOffset, segment.Length)
                        .CopyTo(decodedOutput.View.SubView(
                            segment.DestinationOffset,
                            segment.Length));
                }
                CudaBfp8Native.QuantizeBFloat16(
                    deviceIndex,
                    decodedOutput,
                    output.Payload,
                    output.Scales,
                    descriptor,
                    accelerator.DefaultStream);
                result = FromCudaBfp8Result(
                    output,
                    deviceIndex,
                    outputShape,
                    sources);
            }
            finally
            {
                foreach (CudaBfp8BFloat16Lease? lease in leases)
                    lease?.Dispose();
                ReturnCudaBFloat16Buffer(accelerator, decodedOutput);
            }
        }
        else if (dtype == TensorDType.BFloat16)
        {
            NativeCudaDevice accelerator =
                ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
            NativeCudaBuffer<ushort> output =
                RentCudaBFloat16Buffer(deviceIndex, outputLength);
            try
            {
                foreach (CudaCopySegment segment in segments)
                {
                    sources[segment.SourceIndex]
                        .EnsureCudaBFloat16Buffer(deviceIndex).View
                        .SubView(segment.SourceOffset, segment.Length)
                        .CopyTo(output.View.SubView(
                            segment.DestinationOffset,
                            segment.Length));
                }
                result = FromCudaResult(
                    output,
                    deviceIndex,
                    outputShape,
                    sources,
                    TensorDType.BFloat16);
            }
            catch
            {
                ReturnCudaBFloat16Buffer(accelerator, output);
                throw;
            }
        }
        else
        {
            NativeCudaDevice accelerator =
                ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
            NativeCudaBuffer<float> output =
                RentCudaFloatBuffer(deviceIndex, outputLength);
            try
            {
                foreach (CudaCopySegment segment in segments)
                {
                    sources[segment.SourceIndex]
                        .EnsureCudaFloat32Buffer(deviceIndex).View
                        .SubView(segment.SourceOffset, segment.Length)
                        .CopyTo(output.View.SubView(
                            segment.DestinationOffset,
                            segment.Length));
                }
                result = FromCudaResult(
                    output,
                    deviceIndex,
                    outputShape,
                    sources,
                    TensorDType.Float32);
            }
            catch
            {
                ReturnCudaFloatBuffer(accelerator, output);
                throw;
            }
        }

        if (AutogradContext.IsRecordingEnabled)
        {
            result.Node.BackwardAction = () =>
                CopySegmentsBackwardCuda(result, sources, segments);
        }
        return result;
    }

    private static void CopySegmentsBackwardCuda(
        Tensor result,
        Tensor[] destinations,
        IReadOnlyList<CudaCopySegment> segments)
    {
        int deviceIndex = CudaDeviceIndex;
        if (result.DType == TensorDType.BFloat16
            && TensorExecutionContext.UsesBFloat16GradientStorage
            && TryCopySegmentsBackwardBFloat16Direct(
                result, destinations, segments, deviceIndex))
        {
            return;
        }

        NativeCudaBuffer<float> sourceGradient =
            result.EnsureCudaGradientBuffer(deviceIndex);
        foreach (CudaCopySegment segment in segments)
        {
            Tensor destination = destinations[segment.SourceIndex];
            CudaTensorNative.Accumulate(
                deviceIndex,
                sourceGradient.NativePtr,
                destination.EnsureCudaGradientBuffer(deviceIndex).NativePtr,
                segment.Length,
                sourceOffset: segment.DestinationOffset,
                destinationOffset: segment.SourceOffset);
        }
        foreach (Tensor destination in destinations.Distinct())
        {
            destination.MarkCudaGradientMutated(deviceIndex);
        }
    }

    private static bool TryCopySegmentsBackwardBFloat16Direct(
        Tensor result,
        Tensor[] destinations,
        IReadOnlyList<CudaCopySegment> segments,
        int deviceIndex)
    {
        Tensor[] unique = destinations
            .Distinct<Tensor>(ReferenceEqualityComparer.Instance)
            .ToArray();
        foreach (Tensor destination in unique)
        {
            if (!destination.TryGetCudaBFloat16GradientBuffer(
                    deviceIndex,
                    out _)
                && destination.HasGradientBuffer)
            {
                return false;
            }
        }

        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        bool outputBorrowed = result.TryGetCudaBFloat16GradientBuffer(
            deviceIndex,
            out NativeCudaBuffer<ushort>? outputExisting);
        NativeCudaBuffer<ushort>? rentedOutput = null;
        NativeCudaBuffer<ushort> outputGradient = outputBorrowed
            ? outputExisting!
            : rentedOutput = RentCudaBFloat16Buffer(
                deviceIndex, result.Numel);
        var gradients = new Dictionary<Tensor, NativeCudaBuffer<ushort>>(
            ReferenceEqualityComparer.Instance);
        var fresh = new HashSet<Tensor>(ReferenceEqualityComparer.Instance);
        try
        {
            if (!outputBorrowed)
            {
                CudaTensorNative.EncodeBFloat16(
                    deviceIndex,
                    result.EnsureCudaGradientBuffer(deviceIndex).NativePtr,
                    outputGradient.NativePtr,
                    result.Numel);
            }
            foreach (Tensor destination in unique)
            {
                if (destination.TryGetCudaBFloat16GradientBuffer(
                        deviceIndex,
                        out NativeCudaBuffer<ushort>? existing))
                {
                    gradients.Add(destination, existing!);
                }
                else
                {
                    NativeCudaBuffer<ushort> gradient =
                        RentCudaBFloat16Buffer(
                            deviceIndex, destination.Numel);
                    gradient.MemSetToZero();
                    gradients.Add(destination, gradient);
                    fresh.Add(destination);
                }
            }
            foreach (CudaCopySegment segment in segments)
            {
                Tensor destination = destinations[segment.SourceIndex];
                CudaPublicOpsNative.ShapeAccumulateBFloat16Gradient(
                    deviceIndex,
                    outputGradient.NativePtr,
                    gradients[destination].NativePtr,
                    segment.Length,
                    segment.DestinationOffset,
                    segment.SourceOffset,
                    accelerator.DefaultStream);
            }
            foreach (Tensor destination in fresh)
            {
                destination.AdoptCudaBFloat16GradientBuffer(
                    gradients[destination], deviceIndex);
                gradients.Remove(destination);
            }
            return true;
        }
        finally
        {
            if (rentedOutput is not null)
                ReturnCudaBFloat16Buffer(accelerator, rentedOutput);
            foreach ((Tensor destination, NativeCudaBuffer<ushort> gradient)
                in gradients)
            {
                if (fresh.Contains(destination))
                    ReturnCudaBFloat16Buffer(accelerator, gradient);
            }
        }
    }

    private static bool TryAccumulateBFloat16GradientRangeCuda(
        Tensor source,
        Tensor destination,
        int sourceOffset,
        int destinationOffset,
        int length)
    {
        if (!TensorExecutionContext.UsesBFloat16GradientStorage)
        {
            return false;
        }
        int deviceIndex = CudaDeviceIndex;
        bool destinationBorrowed =
            destination.TryGetCudaBFloat16GradientBuffer(
                deviceIndex,
                out NativeCudaBuffer<ushort>? destinationExisting);
        if (!destinationBorrowed && destination.HasGradientBuffer)
            return false;
        bool sourceBorrowed = source.TryGetCudaBFloat16GradientBuffer(
            deviceIndex,
            out NativeCudaBuffer<ushort>? sourceExisting);
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        NativeCudaBuffer<ushort>? rentedSource = null;
        NativeCudaBuffer<ushort>? rentedDestination = null;
        NativeCudaBuffer<ushort> sourceGradient = sourceBorrowed
            ? sourceExisting!
            : rentedSource = RentCudaBFloat16Buffer(
                deviceIndex, source.Numel);
        NativeCudaBuffer<ushort> destinationGradient = destinationBorrowed
            ? destinationExisting!
            : rentedDestination = RentCudaBFloat16Buffer(
                deviceIndex, destination.Numel);
        try
        {
            if (!sourceBorrowed)
            {
                CudaTensorNative.EncodeBFloat16(
                    deviceIndex,
                    source.EnsureCudaGradientBuffer(deviceIndex).NativePtr,
                    sourceGradient.NativePtr,
                    source.Numel);
            }
            if (!destinationBorrowed)
                destinationGradient.MemSetToZero();
            CudaPublicOpsNative.ShapeAccumulateBFloat16Gradient(
                deviceIndex,
                sourceGradient.NativePtr,
                destinationGradient.NativePtr,
                length,
                sourceOffset,
                destinationOffset,
                accelerator.DefaultStream);
            if (!destinationBorrowed)
            {
                destination.AdoptCudaBFloat16GradientBuffer(
                    destinationGradient, deviceIndex);
                rentedDestination = null;
            }
            return true;
        }
        finally
        {
            if (rentedSource is not null)
                ReturnCudaBFloat16Buffer(accelerator, rentedSource);
            if (rentedDestination is not null)
                ReturnCudaBFloat16Buffer(accelerator, rentedDestination);
        }
    }

    private Tensor TransposeCuda()
    {
        int rows = _shape[0];
        int columns = _shape[1];
        int deviceIndex = CudaDeviceIndex;
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        if (DType == TensorDType.BFloat16)
        {
            return TransposeBFloat16Cuda(
                rows, columns, deviceIndex, accelerator);
        }
        NativeCudaBuffer<float> transposed =
            RentCudaFloatBuffer(deviceIndex, Numel);
        NativeCudaBuffer<float>? decodedSource = null;
        Tensor result;
        try
        {
            NativeCudaBuffer<float> source;
            if (DType == TensorDType.Float32)
            {
                source = EnsureCudaFloat32Buffer(deviceIndex);
            }
            else
            {
                decodedSource = RentCudaFloatBuffer(deviceIndex, Numel);
                source = decodedSource;
                if (DType == TensorDType.BFloat16)
                {
                    CudaTensorNative.DecodeBFloat16(
                        deviceIndex,
                        EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
                        decodedSource.NativePtr,
                        Numel);
                }
                else
                {
                    CudaBfp8BufferView encodedSource =
                        EnsureCudaBfp8Buffer(deviceIndex);
                    CudaBfp8Native.DequantizeFloat32(
                        deviceIndex,
                        encodedSource.Payload,
                        encodedSource.Scales,
                        decodedSource,
                        encodedSource.Descriptor,
                        accelerator.DefaultStream);
                }
            }
            CudaBlas.TransposeFloat32(
                accelerator,
                deviceIndex,
                source,
                transposed,
                rows,
                columns);
            if (DType == TensorDType.Bfp8)
            {
                Bfp8QuantizationDescriptor descriptor =
                    SelectBfp8ResultDescriptor(this);
                using CudaBfp8OwnedBuffers encoded =
                    CudaBfp8OwnedBuffers.Allocate(
                        accelerator,
                        Numel,
                        descriptor);
                CudaBfp8Native.QuantizeFloat32(
                    deviceIndex,
                    transposed,
                    encoded.Payload,
                    encoded.Scales,
                    descriptor,
                    accelerator.DefaultStream);
                result = FromCudaBfp8Result(
                    encoded,
                    deviceIndex,
                    [columns, rows],
                    [this]);
            }
            else if (DType == TensorDType.BFloat16)
            {
                NativeCudaBuffer<ushort> encoded =
                    RentCudaBFloat16Buffer(deviceIndex, Numel);
                try
                {
                    CudaTensorNative.EncodeBFloat16(
                        deviceIndex,
                        transposed.NativePtr,
                        encoded.NativePtr,
                        Numel);
                    result = FromCudaResult(
                        encoded,
                        deviceIndex,
                        [columns, rows],
                        [this],
                        TensorDType.BFloat16);
                }
                catch
                {
                    ReturnCudaBFloat16Buffer(accelerator, encoded);
                    throw;
                }
            }
            else
            {
                result = FromCudaResult(
                    transposed,
                    deviceIndex,
                    [columns, rows],
                    [this],
                    TensorDType.Float32);
                transposed = null!;
            }
        }
        finally
        {
            if (decodedSource is not null)
                ReturnCudaFloatBuffer(accelerator, decodedSource);
            if (transposed is not null)
                ReturnCudaFloatBuffer(accelerator, transposed);
        }

        if (AutogradContext.IsRecordingEnabled)
        {
            result.Node.BackwardAction = () =>
            {
                NativeCudaBuffer<float> contribution =
                    RentCudaFloatBuffer(deviceIndex, Numel);
                try
                {
                    CudaBlas.TransposeFloat32(
                        accelerator,
                        deviceIndex,
                        result.EnsureCudaGradientBuffer(deviceIndex),
                        contribution,
                        columns,
                        rows);
                    CudaTensorNative.Accumulate(
                        deviceIndex,
                        contribution.NativePtr,
                        EnsureCudaGradientBuffer(deviceIndex).NativePtr,
                        Numel);
                    MarkCudaGradientMutated(deviceIndex);
                }
                finally
                {
                    ReturnCudaFloatBuffer(accelerator, contribution);
                }
            };
        }
        return result;
    }

    private Tensor TransposeBFloat16Cuda(
        int rows,
        int columns,
        int deviceIndex,
        NativeCudaDevice accelerator)
    {
        NativeCudaBuffer<ushort> transposed =
            RentCudaBFloat16Buffer(deviceIndex, Numel);
        Tensor result;
        try
        {
            CudaPublicOpsNative.TransposeBFloat16(
                deviceIndex,
                EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
                transposed.NativePtr,
                rows,
                columns,
                accelerator.DefaultStream);
            result = FromCudaResult(
                transposed,
                deviceIndex,
                [columns, rows],
                [this],
                TensorDType.BFloat16);
        }
        catch
        {
            ReturnCudaBFloat16Buffer(accelerator, transposed);
            throw;
        }
        if (AutogradContext.IsRecordingEnabled)
        {
            result.Node.BackwardAction = () =>
            {
                if (TensorExecutionContext.UsesBFloat16GradientStorage
                    && TryTransposeBackwardBFloat16Direct(
                        result,
                        rows,
                        columns,
                        deviceIndex,
                        accelerator))
                {
                    return;
                }
                NativeCudaBuffer<float> contribution =
                    RentCudaFloatBuffer(deviceIndex, Numel);
                try
                {
                    CudaBlas.TransposeFloat32(
                        accelerator,
                        deviceIndex,
                        result.EnsureCudaGradientBuffer(deviceIndex),
                        contribution,
                        columns,
                        rows);
                    CudaTensorNative.Accumulate(
                        deviceIndex,
                        contribution.NativePtr,
                        EnsureCudaGradientBuffer(deviceIndex).NativePtr,
                        Numel);
                    MarkCudaGradientMutated(deviceIndex);
                }
                finally
                {
                    ReturnCudaFloatBuffer(accelerator, contribution);
                }
            };
        }
        return result;
    }

    private bool TryTransposeBackwardBFloat16Direct(
        Tensor output,
        int rows,
        int columns,
        int deviceIndex,
        NativeCudaDevice accelerator)
    {
        bool inputBorrowed = TryGetCudaBFloat16GradientBuffer(
            deviceIndex,
            out NativeCudaBuffer<ushort>? inputExisting);
        if (!inputBorrowed && HasGradientBuffer)
            return false;
        bool outputBorrowed = output.TryGetCudaBFloat16GradientBuffer(
            deviceIndex,
            out NativeCudaBuffer<ushort>? outputExisting);
        NativeCudaBuffer<ushort>? rentedOutput = null;
        NativeCudaBuffer<ushort>? rentedInput = null;
        NativeCudaBuffer<ushort> outputGradient = outputBorrowed
            ? outputExisting!
            : rentedOutput = RentCudaBFloat16Buffer(deviceIndex, Numel);
        NativeCudaBuffer<ushort> inputGradient = inputBorrowed
            ? inputExisting!
            : rentedInput = RentCudaBFloat16Buffer(deviceIndex, Numel);
        try
        {
            if (!outputBorrowed)
            {
                CudaTensorNative.EncodeBFloat16(
                    deviceIndex,
                    output.EnsureCudaGradientBuffer(deviceIndex).NativePtr,
                    outputGradient.NativePtr,
                    Numel);
            }
            if (!inputBorrowed)
                inputGradient.MemSetToZero();
            CudaPublicOpsNative.TransposeBackwardBFloat16Gradient(
                deviceIndex,
                outputGradient.NativePtr,
                inputGradient.NativePtr,
                rows,
                columns,
                accelerator.DefaultStream);
            if (!inputBorrowed)
            {
                AdoptCudaBFloat16GradientBuffer(inputGradient, deviceIndex);
                rentedInput = null;
            }
            return true;
        }
        finally
        {
            if (rentedOutput is not null)
                ReturnCudaBFloat16Buffer(accelerator, rentedOutput);
            if (rentedInput is not null)
                ReturnCudaBFloat16Buffer(accelerator, rentedInput);
        }
    }
}
