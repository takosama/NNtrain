namespace NNtrain;

partial class Tensor
{
    private static Tensor ApplyBinaryElementwiseCuda(
        Tensor left,
        Tensor right,
        BinaryBroadcastPlan plan,
        BinaryOperation operation)
    {
        if (left.DType != right.DType)
        {
            throw new InvalidOperationException(
                "CUDA elementwise operations require matching physical " +
                "storage types. Convert operands explicitly before dispatch.");
        }

        int deviceIndex = CudaDeviceIndex;
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        nint stream = accelerator.DefaultStream;
        CudaPublicBinaryOperation cudaOperation = operation switch
        {
            BinaryOperation.Add => CudaPublicBinaryOperation.Add,
            BinaryOperation.Subtract => CudaPublicBinaryOperation.Subtract,
            BinaryOperation.Multiply => CudaPublicBinaryOperation.Multiply,
            BinaryOperation.Divide => CudaPublicBinaryOperation.Divide,
            _ => throw new InvalidOperationException(
                $"Unknown binary operation '{operation}'."),
        };

        Tensor result;
        if (left.DType == TensorDType.Bfp8)
        {
            Bfp8QuantizationDescriptor descriptor =
                SelectBfp8ResultDescriptor(left, right);
            using CudaBfp8BFloat16Lease leftDecoded =
                left.AcquireCudaBfp8BFloat16Buffer(deviceIndex);
            using CudaBfp8BFloat16Lease rightDecoded =
                right.AcquireCudaBfp8BFloat16Buffer(deviceIndex);
            NativeCudaBuffer<ushort> encodedOutput =
                RentCudaBFloat16Buffer(deviceIndex, plan.ElementCount);
            using CudaBfp8OwnedBuffers output = CudaBfp8OwnedBuffers.Allocate(
                accelerator,
                plan.ElementCount,
                descriptor);
            try
            {
                CudaPublicOpsNative.Binary(
                    deviceIndex,
                    leftDecoded.Buffer.NativePtr,
                    rightDecoded.Buffer.NativePtr,
                    encodedOutput.NativePtr,
                    plan.ElementCount,
                    plan.LeftIsScalar,
                    plan.RightIsScalar,
                    cudaOperation,
                    bfloat16: true,
                    stream);
                CudaBfp8Native.QuantizeBFloat16(
                    deviceIndex,
                    encodedOutput,
                    output.Payload,
                    output.Scales,
                    descriptor,
                    stream);
                result = FromCudaBfp8Result(
                    output,
                    deviceIndex,
                    plan.ResultShape,
                    [left, right]);
            }
            finally
            {
                ReturnCudaBFloat16Buffer(accelerator, encodedOutput);
            }
        }
        else if (left.DType == TensorDType.BFloat16)
        {
            NativeCudaBuffer<ushort> output =
                RentCudaBFloat16Buffer(deviceIndex, plan.ElementCount);
            try
            {
                CudaPublicOpsNative.Binary(
                    deviceIndex,
                    left.EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
                    right.EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
                    output.NativePtr,
                    plan.ElementCount,
                    plan.LeftIsScalar,
                    plan.RightIsScalar,
                    cudaOperation,
                    bfloat16: true,
                    stream);
                result = FromCudaResult(
                    output,
                    deviceIndex,
                    plan.ResultShape,
                    [left, right],
                    TensorDType.BFloat16);
            }
            catch
            {
                ReturnCudaBFloat16Buffer(accelerator, output);
                throw;
            }
        }
        else if (left.DType == TensorDType.Float32)
        {
            NativeCudaBuffer<float> output =
                RentCudaFloatBuffer(deviceIndex, plan.ElementCount);
            try
            {
                CudaPublicOpsNative.Binary(
                    deviceIndex,
                    left.EnsureCudaFloat32Buffer(deviceIndex).NativePtr,
                    right.EnsureCudaFloat32Buffer(deviceIndex).NativePtr,
                    output.NativePtr,
                    plan.ElementCount,
                    plan.LeftIsScalar,
                    plan.RightIsScalar,
                    cudaOperation,
                    bfloat16: false,
                    stream);
                result = FromCudaResult(
                    output,
                    deviceIndex,
                    plan.ResultShape,
                    [left, right],
                    TensorDType.Float32);
            }
            catch
            {
                ReturnCudaFloatBuffer(accelerator, output);
                throw;
            }
        }
        else
        {
            ThrowIfCudaHostFallback($"Elementwise {operation}");
            throw new InvalidOperationException(
                "CUDA elementwise dispatch did not select a supported dtype.");
        }

        if (AutogradContext.IsRecordingEnabled)
        {
            result.Node.BackwardAction = () => BinaryBackwardCuda(
                left,
                right,
                result,
                plan,
                cudaOperation);
        }
        return result;
    }

    private static void BinaryBackwardCuda(
        Tensor left,
        Tensor right,
        Tensor output,
        BinaryBroadcastPlan plan,
        CudaPublicBinaryOperation operation)
    {
        int deviceIndex = CudaDeviceIndex;
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        nint stream = accelerator.DefaultStream;

        if (left.DType == TensorDType.BFloat16
            && TensorExecutionContext.UsesBFloat16GradientStorage
            && TryBinaryBackwardBFloat16Direct(
                left,
                right,
                output,
                plan,
                operation,
                deviceIndex,
                accelerator,
                stream))
        {
            return;
        }

        NativeCudaBuffer<float> outputGradient =
            output.EnsureCudaGradientBuffer(deviceIndex);
        NativeCudaBuffer<float> leftGradient =
            left.EnsureCudaGradientBuffer(deviceIndex);
        NativeCudaBuffer<float> rightGradient = ReferenceEquals(left, right)
            ? leftGradient
            : right.EnsureCudaGradientBuffer(deviceIndex);

        if (left.DType == TensorDType.Bfp8)
        {
            using CudaBfp8BFloat16Lease leftDecoded =
                left.AcquireCudaBfp8BFloat16Buffer(deviceIndex);
            using CudaBfp8BFloat16Lease rightDecoded =
                right.AcquireCudaBfp8BFloat16Buffer(deviceIndex);
            CudaPublicOpsNative.BinaryBackward(
                deviceIndex,
                leftDecoded.Buffer.NativePtr,
                rightDecoded.Buffer.NativePtr,
                outputGradient.NativePtr,
                leftGradient.NativePtr,
                rightGradient.NativePtr,
                plan.ElementCount,
                plan.LeftIsScalar,
                plan.RightIsScalar,
                ReferenceEquals(left, right),
                operation,
                bfloat16: true,
                stream);
        }
        else if (left.DType == TensorDType.BFloat16)
        {
            CudaPublicOpsNative.BinaryBackward(
                deviceIndex,
                left.EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
                right.EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
                outputGradient.NativePtr,
                leftGradient.NativePtr,
                rightGradient.NativePtr,
                plan.ElementCount,
                plan.LeftIsScalar,
                plan.RightIsScalar,
                ReferenceEquals(left, right),
                operation,
                bfloat16: true,
                stream);
        }
        else
        {
            CudaPublicOpsNative.BinaryBackward(
                deviceIndex,
                left.EnsureCudaFloat32Buffer(deviceIndex).NativePtr,
                right.EnsureCudaFloat32Buffer(deviceIndex).NativePtr,
                outputGradient.NativePtr,
                leftGradient.NativePtr,
                rightGradient.NativePtr,
                plan.ElementCount,
                plan.LeftIsScalar,
                plan.RightIsScalar,
                ReferenceEquals(left, right),
                operation,
                bfloat16: false,
                stream);
        }
        left.MarkCudaGradientMutated(deviceIndex);
        if (!ReferenceEquals(left, right))
            right.MarkCudaGradientMutated(deviceIndex);
    }

    private static bool TryBinaryBackwardBFloat16Direct(
        Tensor left,
        Tensor right,
        Tensor output,
        BinaryBroadcastPlan plan,
        CudaPublicBinaryOperation operation,
        int deviceIndex,
        NativeCudaDevice accelerator,
        nint stream)
    {
        bool sameParent = ReferenceEquals(left, right);
        bool leftBorrowed = left.TryGetCudaBFloat16GradientBuffer(
            deviceIndex,
            out NativeCudaBuffer<ushort>? leftExisting);
        NativeCudaBuffer<ushort>? rightExisting = null;
        bool rightBorrowed = sameParent || right.TryGetCudaBFloat16GradientBuffer(
            deviceIndex,
            out rightExisting);
        if ((!leftBorrowed && left.HasGradientBuffer)
            || (!sameParent && !rightBorrowed && right.HasGradientBuffer))
        {
            return false;
        }

        NativeCudaBuffer<ushort>? rentedOutput = null;
        NativeCudaBuffer<ushort>? rentedLeft = null;
        NativeCudaBuffer<ushort>? rentedRight = null;
        bool outputBorrowed = output.TryGetCudaBFloat16GradientBuffer(
            deviceIndex,
            out NativeCudaBuffer<ushort>? outputExisting);
        NativeCudaBuffer<ushort> outputGradient = outputBorrowed
            ? outputExisting!
            : rentedOutput = RentCudaBFloat16Buffer(
                deviceIndex,
                output.Numel);
        NativeCudaBuffer<ushort> leftGradient = leftBorrowed
            ? leftExisting!
            : rentedLeft = RentCudaBFloat16Buffer(deviceIndex, left.Numel);
        NativeCudaBuffer<ushort> rightGradient = sameParent
            ? leftGradient
            : rightBorrowed
                ? rightExisting!
                : rentedRight = RentCudaBFloat16Buffer(
                    deviceIndex,
                    right.Numel);
        try
        {
            if (!outputBorrowed)
            {
                CudaTensorNative.EncodeBFloat16(
                    deviceIndex,
                    output.EnsureCudaGradientBuffer(deviceIndex).NativePtr,
                    outputGradient.NativePtr,
                    output.Numel);
            }
            if (!leftBorrowed)
                leftGradient.MemSetToZero();
            if (!sameParent && !rightBorrowed)
                rightGradient.MemSetToZero();
            CudaPublicOpsNative.BinaryBackwardBFloat16Gradient(
                deviceIndex,
                left.EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
                right.EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
                outputGradient.NativePtr,
                leftGradient.NativePtr,
                rightGradient.NativePtr,
                plan.ElementCount,
                plan.LeftIsScalar,
                plan.RightIsScalar,
                sameParent,
                operation,
                stream);
            if (!leftBorrowed)
            {
                left.AdoptCudaBFloat16GradientBuffer(
                    leftGradient,
                    deviceIndex);
                rentedLeft = null;
            }
            if (!sameParent && !rightBorrowed)
            {
                right.AdoptCudaBFloat16GradientBuffer(
                    rightGradient,
                    deviceIndex);
                rentedRight = null;
            }
            return true;
        }
        finally
        {
            if (rentedOutput is not null)
                ReturnCudaBFloat16Buffer(accelerator, rentedOutput);
            if (rentedLeft is not null)
                ReturnCudaBFloat16Buffer(accelerator, rentedLeft);
            if (rentedRight is not null)
                ReturnCudaBFloat16Buffer(accelerator, rentedRight);
        }
    }

    private Tensor ApplyUnaryCuda(
        CudaPublicUnaryOperation operation,
        float parameter = 0f)
    {
        int deviceIndex = CudaDeviceIndex;
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        nint stream = accelerator.DefaultStream;
        Tensor result;

        if (DType == TensorDType.Bfp8)
        {
            Bfp8QuantizationDescriptor descriptor =
                SelectBfp8ResultDescriptor(this);
            using CudaBfp8BFloat16Lease decoded =
                AcquireCudaBfp8BFloat16Buffer(deviceIndex);
            NativeCudaBuffer<ushort> encodedOutput =
                RentCudaBFloat16Buffer(deviceIndex, Numel);
            using CudaBfp8OwnedBuffers output = CudaBfp8OwnedBuffers.Allocate(
                accelerator,
                Numel,
                descriptor);
            try
            {
                CudaPublicOpsNative.Unary(
                    deviceIndex,
                    decoded.Buffer.NativePtr,
                    encodedOutput.NativePtr,
                    Numel,
                    operation,
                    parameter,
                    bfloat16: true,
                    stream);
                CudaBfp8Native.QuantizeBFloat16(
                    deviceIndex,
                    encodedOutput,
                    output.Payload,
                    output.Scales,
                    descriptor,
                    stream);
                result = FromCudaBfp8Result(
                    output,
                    deviceIndex,
                    _shape,
                    [this]);
            }
            finally
            {
                ReturnCudaBFloat16Buffer(accelerator, encodedOutput);
            }
        }
        else if (DType == TensorDType.BFloat16)
        {
            NativeCudaBuffer<ushort> output =
                RentCudaBFloat16Buffer(deviceIndex, Numel);
            try
            {
                CudaPublicOpsNative.Unary(
                    deviceIndex,
                    EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
                    output.NativePtr,
                    Numel,
                    operation,
                    parameter,
                    bfloat16: true,
                    stream);
                result = FromCudaResult(
                    output,
                    deviceIndex,
                    _shape,
                    [this],
                    TensorDType.BFloat16);
            }
            catch
            {
                ReturnCudaBFloat16Buffer(accelerator, output);
                throw;
            }
        }
        else if (DType == TensorDType.Float32)
        {
            NativeCudaBuffer<float> output =
                RentCudaFloatBuffer(deviceIndex, Numel);
            try
            {
                CudaPublicOpsNative.Unary(
                    deviceIndex,
                    EnsureCudaFloat32Buffer(deviceIndex).NativePtr,
                    output.NativePtr,
                    Numel,
                    operation,
                    parameter,
                    bfloat16: false,
                    stream);
                result = FromCudaResult(
                    output,
                    deviceIndex,
                    _shape,
                    [this],
                    TensorDType.Float32);
            }
            catch
            {
                ReturnCudaFloatBuffer(accelerator, output);
                throw;
            }
        }
        else
        {
            ThrowIfCudaHostFallback(operation.ToString());
            throw new InvalidOperationException(
                "CUDA unary dispatch did not select a supported dtype.");
        }

        if (AutogradContext.IsRecordingEnabled)
        {
            result.Node.BackwardAction = () =>
                UnaryBackwardCuda(result, operation, parameter);
        }
        return result;
    }

    private void UnaryBackwardCuda(
        Tensor output,
        CudaPublicUnaryOperation operation,
        float parameter)
    {
        int deviceIndex = CudaDeviceIndex;
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        nint stream = accelerator.DefaultStream;

        if (DType == TensorDType.BFloat16
            && TensorExecutionContext.UsesBFloat16GradientStorage
            && TryUnaryBackwardBFloat16Direct(
                output,
                operation,
                parameter,
                deviceIndex,
                accelerator,
                stream))
        {
            return;
        }

        nint outputGradient =
            output.EnsureCudaGradientBuffer(deviceIndex).NativePtr;
        nint inputGradient =
            EnsureCudaGradientBuffer(deviceIndex).NativePtr;

        if (DType == TensorDType.Bfp8)
        {
            using CudaBfp8BFloat16Lease inputDecoded =
                AcquireCudaBfp8BFloat16Buffer(deviceIndex);
            using CudaBfp8BFloat16Lease outputDecoded =
                output.AcquireCudaBfp8BFloat16Buffer(deviceIndex);
            CudaPublicOpsNative.UnaryBackward(
                deviceIndex,
                inputDecoded.Buffer.NativePtr,
                outputDecoded.Buffer.NativePtr,
                outputGradient,
                inputGradient,
                Numel,
                operation,
                parameter,
                bfloat16: true,
                stream);
        }
        else if (DType == TensorDType.BFloat16)
        {
            CudaPublicOpsNative.UnaryBackward(
                deviceIndex,
                EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
                output.EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
                outputGradient,
                inputGradient,
                Numel,
                operation,
                parameter,
                bfloat16: true,
                stream);
        }
        else
        {
            CudaPublicOpsNative.UnaryBackward(
                deviceIndex,
                EnsureCudaFloat32Buffer(deviceIndex).NativePtr,
                output.EnsureCudaFloat32Buffer(deviceIndex).NativePtr,
                outputGradient,
                inputGradient,
                Numel,
                operation,
                parameter,
                bfloat16: false,
                stream);
        }
        MarkCudaGradientMutated(deviceIndex);
    }

    private bool TryUnaryBackwardBFloat16Direct(
        Tensor output,
        CudaPublicUnaryOperation operation,
        float parameter,
        int deviceIndex,
        NativeCudaDevice accelerator,
        nint stream)
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
            : rentedOutput = RentCudaBFloat16Buffer(deviceIndex, output.Numel);
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
                    output.Numel);
            }
            if (!inputBorrowed)
                inputGradient.MemSetToZero();
            CudaPublicOpsNative.UnaryBackwardBFloat16Gradient(
                deviceIndex,
                EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
                output.EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
                outputGradient.NativePtr,
                inputGradient.NativePtr,
                Numel,
                operation,
                parameter,
                stream);
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

    private Tensor ReduceCuda(CudaPublicReductionOperation operation)
    {
        int deviceIndex = CudaDeviceIndex;
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        nint stream = accelerator.DefaultStream;
        NativeCudaBuffer<float> output = RentCudaFloatBuffer(deviceIndex, 1);
        try
        {
            if (DType == TensorDType.Bfp8)
            {
                using CudaBfp8BFloat16Lease decoded =
                    AcquireCudaBfp8BFloat16Buffer(deviceIndex);
                CudaPublicOpsNative.Reduce(
                    deviceIndex,
                    decoded.Buffer.NativePtr,
                    output.NativePtr,
                    Numel,
                    operation,
                    bfloat16: true,
                    stream);
            }
            else if (DType == TensorDType.BFloat16)
            {
                CudaPublicOpsNative.Reduce(
                    deviceIndex,
                    EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
                    output.NativePtr,
                    Numel,
                    operation,
                    bfloat16: true,
                    stream);
            }
            else if (DType == TensorDType.Float32)
            {
                CudaPublicOpsNative.Reduce(
                    deviceIndex,
                    EnsureCudaFloat32Buffer(deviceIndex).NativePtr,
                    output.NativePtr,
                    Numel,
                    operation,
                    bfloat16: false,
                    stream);
            }
            else
            {
                ThrowIfCudaHostFallback(operation.ToString());
            }

            Tensor result = FromCudaResult(
                output,
                deviceIndex,
                [1],
                [this],
                TensorDType.Float32);
            if (AutogradContext.IsRecordingEnabled)
            {
                result.Node.BackwardAction = () =>
                    ReductionBackwardCuda(result, operation);
            }
            return result;
        }
        catch
        {
            ReturnCudaFloatBuffer(accelerator, output);
            throw;
        }
    }

    private void ReductionBackwardCuda(
        Tensor output,
        CudaPublicReductionOperation operation)
    {
        int deviceIndex = CudaDeviceIndex;
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        nint stream = accelerator.DefaultStream;
        nint outputValue =
            output.EnsureCudaFloat32Buffer(deviceIndex).NativePtr;
        nint outputGradient =
            output.EnsureCudaGradientBuffer(deviceIndex).NativePtr;

        if (DType == TensorDType.BFloat16
            && TensorExecutionContext.UsesBFloat16GradientStorage)
        {
            bool borrowed = TryGetCudaBFloat16GradientBuffer(
                deviceIndex,
                out NativeCudaBuffer<ushort>? existing);
            if (borrowed || !HasGradientBuffer)
            {
                NativeCudaBuffer<ushort>? rented = borrowed
                    ? null
                    : RentCudaBFloat16Buffer(deviceIndex, Numel);
                NativeCudaBuffer<ushort> direct = existing ?? rented!;
                try
                {
                    if (!borrowed)
                        direct.MemSetToZero();
                    CudaPublicOpsNative.ReduceBackwardBFloat16Gradient(
                        deviceIndex,
                        EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
                        outputValue,
                        outputGradient,
                        direct.NativePtr,
                        Numel,
                        operation,
                        stream);
                    if (!borrowed)
                    {
                        AdoptCudaBFloat16GradientBuffer(direct, deviceIndex);
                        rented = null;
                    }
                    return;
                }
                finally
                {
                    if (rented is not null)
                        ReturnCudaBFloat16Buffer(accelerator, rented);
                }
            }
        }

        nint inputGradient =
            EnsureCudaGradientBuffer(deviceIndex).NativePtr;

        if (DType == TensorDType.Bfp8)
        {
            using CudaBfp8BFloat16Lease decoded =
                AcquireCudaBfp8BFloat16Buffer(deviceIndex);
            CudaPublicOpsNative.ReduceBackward(
                deviceIndex,
                decoded.Buffer.NativePtr,
                outputValue,
                outputGradient,
                inputGradient,
                Numel,
                operation,
                bfloat16: true,
                stream);
        }
        else if (DType == TensorDType.BFloat16)
        {
            CudaPublicOpsNative.ReduceBackward(
                deviceIndex,
                EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
                outputValue,
                outputGradient,
                inputGradient,
                Numel,
                operation,
                bfloat16: true,
                stream);
        }
        else
        {
            CudaPublicOpsNative.ReduceBackward(
                deviceIndex,
                EnsureCudaFloat32Buffer(deviceIndex).NativePtr,
                outputValue,
                outputGradient,
                inputGradient,
                Numel,
                operation,
                bfloat16: false,
                stream);
        }
        MarkCudaGradientMutated(deviceIndex);
    }
}
