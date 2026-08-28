using System.Runtime.ExceptionServices;
using NNtrain.Cuda.Memory;

namespace NNtrain;

internal readonly record struct CudaBfp8ForgetMemoryTelemetrySnapshot(
    long TensorCoreForwardExecutions,
    long GenericCudaForwardExecutions)
{
    public static CudaBfp8ForgetMemoryTelemetrySnapshot operator -(
        CudaBfp8ForgetMemoryTelemetrySnapshot left,
        CudaBfp8ForgetMemoryTelemetrySnapshot right)
        => new(
            left.TensorCoreForwardExecutions
                - right.TensorCoreForwardExecutions,
            left.GenericCudaForwardExecutions
                - right.GenericCudaForwardExecutions);
}

internal static class CudaBfp8ForgetMemoryTelemetry
{
    private static long _tensorCoreForwardExecutions;
    private static long _genericCudaForwardExecutions;

    internal static CudaBfp8ForgetMemoryTelemetrySnapshot Snapshot => new(
        Interlocked.Read(ref _tensorCoreForwardExecutions),
        Interlocked.Read(ref _genericCudaForwardExecutions));

    internal static void Record(bool tensorCore)
    {
        if (tensorCore)
            Interlocked.Increment(ref _tensorCoreForwardExecutions);
        else
            Interlocked.Increment(ref _genericCudaForwardExecutions);
    }
}

/// <summary>
/// Resident BFP8/mix8_32 ForgetMemory implementation. Encoded activations
/// remain authoritative on CUDA. The recurrent state, all reductions, and
/// backward accumulation stay in FP32; the existing BF16 Tensor Core kernel
/// is used for supported shapes and the generic BF16 CUDA kernel handles all
/// other shapes without a CPU fallback.
/// </summary>
internal static class CudaBfp8ForgetMemory
{
    internal static CudaBfp8ForgetMemoryResidentContext ForwardResident(
        Tensor projected,
        Bfp8QuantizationDescriptor outputDescriptor,
        int batch,
        int sequence,
        int projectionWidth,
        int keyWidth,
        int valueWidth,
        float retentionFloor,
        bool useV3,
        bool useDrn)
    {
        Validate(
            projected,
            outputDescriptor,
            batch,
            sequence,
            projectionWidth,
            keyWidth,
            valueWidth);

        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        int matrixSize = checked(keyWidth * valueWidth);
        int outputLength = checked(batch * sequence * valueWidth);
        int projectedLength = checked(batch * sequence * projectionWidth);

        NativeCudaBuffer<ushort>? decodedProjection = null;
        NativeCudaBuffer<ushort>? decodedOutput = null;
        NativeCudaBuffer<float>? states = null;
        NativeCudaBuffer<float>? state = null;
        CudaBfp8OwnedBuffers? encodedOutput = null;
        try
        {
            decodedProjection = Tensor.RentCudaBFloat16Buffer(
                deviceIndex,
                projectedLength);
            decodedOutput = Tensor.RentCudaBFloat16Buffer(
                deviceIndex,
                outputLength);
            states = accelerator.Allocate1D<float>(
                checked(batch * sequence * matrixSize),
                CudaMemoryKind.Transient);
            state = accelerator.Allocate1D<float>(
                checked(batch * matrixSize),
                CudaMemoryKind.Transient);
            encodedOutput = CudaBfp8OwnedBuffers.Allocate(
                accelerator,
                outputLength,
                outputDescriptor);

            nint stream = accelerator.DefaultStream;
            CudaBfp8BufferView projectedView =
                projected.EnsureCudaBfp8Buffer(deviceIndex);
            CudaBfp8Native.DequantizeBFloat16(
                deviceIndex,
                projectedView.Payload,
                projectedView.Scales,
                decodedProjection,
                projectedView.Descriptor,
                stream);
            state.MemSetToZero();

            bool tensorCore = LaunchForward(
                accelerator,
                decodedProjection,
                decodedOutput,
                states,
                state,
                batch,
                sequence,
                projectionWidth,
                keyWidth,
                valueWidth,
                retentionFloor,
                useV3,
                useDrn);
            CudaBfp8ForgetMemoryTelemetry.Record(tensorCore);
            CudaBfp8Native.QuantizeBFloat16(
                deviceIndex,
                decodedOutput,
                encodedOutput.Payload,
                encodedOutput.Scales,
                outputDescriptor,
                stream);

            // Forward temporaries cannot return to the shared pool until both
            // the recurrence and device-side requantization have completed.
            if (!TensorExecutionContext.TryGetCudaStreamLane(
                    deviceIndex,
                    out _))
            {
                accelerator.Synchronize();
            }
            NativeCudaBuffer<ushort> completedOutput = decodedOutput;
            decodedOutput = null;
            Tensor.ReturnCudaBFloat16Buffer(accelerator, completedOutput);
            NativeCudaBuffer<float> completedState = state;
            state = null;
            completedState.Dispose();

            var context = new CudaBfp8ForgetMemoryResidentContext(
                deviceIndex,
                accelerator,
                decodedProjection,
                states,
                encodedOutput);
            decodedProjection = null;
            states = null;
            encodedOutput = null;
            return context;
        }
        catch (Exception failure)
        {
            List<Exception>? cleanupFailures = null;
            TryCleanup(encodedOutput, ref cleanupFailures);
            TryCleanup(state, ref cleanupFailures);
            TryCleanup(states, ref cleanupFailures);
            TryReturnBFloat16(
                accelerator,
                decodedOutput,
                ref cleanupFailures);
            TryReturnBFloat16(
                accelerator,
                decodedProjection,
                ref cleanupFailures);
            RethrowWithCleanup(
                failure,
                cleanupFailures,
                "BFP8 ForgetMemory forward and rollback failed.");
            throw;
        }
    }

    internal static void BackwardResident(
        Tensor projected,
        Tensor output,
        CudaBfp8ForgetMemoryResidentContext forward,
        int batch,
        int sequence,
        int projectionWidth,
        int keyWidth,
        int valueWidth,
        float retentionFloor,
        bool useV3,
        bool useDrn)
    {
        ArgumentNullException.ThrowIfNull(projected);
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(forward);
        if (projected.DType != TensorDType.Bfp8
            || output.DType != TensorDType.Bfp8)
        {
            throw new InvalidOperationException(
                "Resident BFP8 ForgetMemory backward requires BFP8 tensors.");
        }

        int deviceIndex = forward.DeviceIndex;
        NativeCudaDevice accelerator = forward.Accelerator;
        int matrixSize = checked(keyWidth * valueWidth);
        using NativeCudaBuffer<float> stateGradient =
            accelerator.Allocate1D<float>(
                checked(batch * matrixSize),
                CudaMemoryKind.Transient);
        using NativeCudaBuffer<float> previousGradient =
            accelerator.Allocate1D<float>(
                checked(batch * matrixSize),
                CudaMemoryKind.Transient);
        stateGradient.MemSetToZero();
        previousGradient.MemSetToZero();

        CudaForgetMemoryNative.Backward(
            accelerator,
            projected: 0,
            forward.DecodedProjection.NativePtr,
            projected.EnsureCudaGradientBuffer(deviceIndex).NativePtr,
            output.EnsureCudaGradientBuffer(deviceIndex).NativePtr,
            forward.States.NativePtr,
            stateGradient.NativePtr,
            previousGradient.NativePtr,
            batch,
            sequence,
            projectionWidth,
            keyWidth,
            valueWidth,
            retentionFloor,
            useDrn ? 2 : useV3 ? 1 : 0,
            bfloat16: true);
        if (!TensorExecutionContext.TryGetCudaStreamLane(
                deviceIndex,
                out _))
        {
            accelerator.Synchronize();
        }
        projected.MarkCudaGradientMutated(deviceIndex);
    }

    /// <summary>
    /// Executes the host-state continuation API without materializing the
    /// encoded projection on the host. The only transfers are the explicitly
    /// supplied and returned FP32 recurrent-state array.
    /// </summary>
    internal static CudaBfp8OwnedBuffers ForwardContinue(
        Tensor projected,
        Bfp8QuantizationDescriptor outputDescriptor,
        int sequence,
        int projectionWidth,
        int keyWidth,
        int valueWidth,
        float retentionFloor,
        float[] hostState,
        bool useV3,
        bool useDrn)
    {
        ArgumentNullException.ThrowIfNull(hostState);
        int matrixSize = checked(keyWidth * valueWidth);
        if (hostState.Length != matrixSize)
        {
            throw new ArgumentException(
                "The recurrent state length must equal keyWidth * valueWidth.",
                nameof(hostState));
        }

        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        using NativeCudaBuffer<float> state = accelerator.Allocate1D<float>(
            matrixSize,
            CudaMemoryKind.Transient);
        state.CopyFromCPU(hostState);
        CudaBfp8OwnedBuffers output = ForwardContinue(
            projected,
            outputDescriptor,
            sequence,
            projectionWidth,
            keyWidth,
            valueWidth,
            retentionFloor,
            state,
            useV3,
            useDrn);
        try
        {
            state.CopyToCPU(hostState);
            return output;
        }
        catch
        {
            output.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Advances a device-authoritative recurrent state.  The state buffer is
    /// borrowed and remains resident; only the encoded output is returned.
    /// </summary>
    internal static CudaBfp8OwnedBuffers ForwardContinue(
        Tensor projected,
        Bfp8QuantizationDescriptor outputDescriptor,
        int sequence,
        int projectionWidth,
        int keyWidth,
        int valueWidth,
        float retentionFloor,
        NativeCudaBuffer<float> state,
        bool useV3,
        bool useDrn)
    {
        ArgumentNullException.ThrowIfNull(state);
        Validate(
            projected,
            outputDescriptor,
            batch: 1,
            sequence,
            projectionWidth,
            keyWidth,
            valueWidth);
        int matrixSize = checked(keyWidth * valueWidth);
        int deviceIndex = Tensor.CudaDeviceIndex;
        if (state.Device.Index != deviceIndex || state.Length != matrixSize)
        {
            throw new ArgumentException(
                "The recurrent CUDA state must match device and memory size.",
                nameof(state));
        }
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        int outputLength = checked(sequence * valueWidth);
        NativeCudaBuffer<ushort>? decodedProjection = null;
        NativeCudaBuffer<ushort>? decodedOutput = null;
        NativeCudaBuffer<float>? states = null;
        CudaBfp8OwnedBuffers? encodedOutput = null;
        try
        {
            decodedProjection = Tensor.RentCudaBFloat16Buffer(
                deviceIndex,
                projected.Numel);
            decodedOutput = Tensor.RentCudaBFloat16Buffer(
                deviceIndex,
                outputLength);
            states = accelerator.Allocate1D<float>(
                checked(sequence * matrixSize),
                CudaMemoryKind.Transient);
            encodedOutput = CudaBfp8OwnedBuffers.Allocate(
                accelerator,
                outputLength,
                outputDescriptor);

            nint stream = accelerator.DefaultStream;
            CudaBfp8BufferView projectedView =
                projected.EnsureCudaBfp8Buffer(deviceIndex);
            CudaBfp8Native.DequantizeBFloat16(
                deviceIndex,
                projectedView.Payload,
                projectedView.Scales,
                decodedProjection,
                projectedView.Descriptor,
                stream);
            bool tensorCore = LaunchForward(
                accelerator,
                decodedProjection,
                decodedOutput,
                states,
                state,
                batch: 1,
                sequence,
                projectionWidth,
                keyWidth,
                valueWidth,
                retentionFloor,
                useV3,
                useDrn);
            CudaBfp8ForgetMemoryTelemetry.Record(tensorCore);
            CudaBfp8Native.QuantizeBFloat16(
                deviceIndex,
                decodedOutput,
                encodedOutput.Payload,
                encodedOutput.Scales,
                outputDescriptor,
                stream);
            if (!TensorExecutionContext.TryGetCudaStreamLane(
                    deviceIndex,
                    out _))
            {
                accelerator.Synchronize();
            }

            List<Exception>? cleanupFailures = null;
            NativeCudaBuffer<float> completedStates = states;
            states = null;
            TryCleanup(completedStates, ref cleanupFailures);
            NativeCudaBuffer<ushort> completedOutput = decodedOutput;
            decodedOutput = null;
            TryReturnBFloat16(
                accelerator,
                completedOutput,
                ref cleanupFailures);
            NativeCudaBuffer<ushort> completedProjection = decodedProjection;
            decodedProjection = null;
            TryReturnBFloat16(
                accelerator,
                completedProjection,
                ref cleanupFailures);
            if (cleanupFailures is not null)
            {
                TryCleanup(encodedOutput, ref cleanupFailures);
                encodedOutput = null;
                throw new AggregateException(
                    "BFP8 ForgetMemory continuation cleanup failed.",
                    cleanupFailures!);
            }

            CudaBfp8OwnedBuffers result = encodedOutput;
            encodedOutput = null;
            return result;
        }
        catch (Exception failure)
        {
            List<Exception>? cleanupFailures = null;
            TryCleanup(encodedOutput, ref cleanupFailures);
            TryCleanup(states, ref cleanupFailures);
            TryReturnBFloat16(
                accelerator,
                decodedOutput,
                ref cleanupFailures);
            TryReturnBFloat16(
                accelerator,
                decodedProjection,
                ref cleanupFailures);
            RethrowWithCleanup(
                failure,
                cleanupFailures,
                "BFP8 ForgetMemory continuation and rollback failed.");
            throw;
        }
    }

    private static bool LaunchForward(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<ushort> projected,
        NativeCudaBuffer<ushort> output,
        NativeCudaBuffer<float> states,
        NativeCudaBuffer<float> state,
        int batch,
        int sequence,
        int projectionWidth,
        int keyWidth,
        int valueWidth,
        float retentionFloor,
        bool useV3,
        bool useDrn)
    {
        int memoryVariant = useDrn ? 2 : useV3 ? 1 : 0;
        bool tensorCore = CudaForgetMemoryTensorCore.TryForward(
            accelerator,
            projected,
            output,
            states,
            state,
            batch,
            sequence,
            projectionWidth,
            keyWidth,
            valueWidth,
            retentionFloor,
            memoryVariant);
        if (!tensorCore)
        {
            CudaForgetMemoryNative.Forward(
                accelerator,
                projected: 0,
                projected.NativePtr,
                output: 0,
                output.NativePtr,
                states.NativePtr,
                state.NativePtr,
                batch,
                sequence,
                projectionWidth,
                keyWidth,
                valueWidth,
                retentionFloor,
                memoryVariant,
                bfloat16: true);
        }
        return tensorCore;
    }

    private static void Validate(
        Tensor projected,
        Bfp8QuantizationDescriptor outputDescriptor,
        int batch,
        int sequence,
        int projectionWidth,
        int keyWidth,
        int valueWidth)
    {
        ArgumentNullException.ThrowIfNull(projected);
        ArgumentNullException.ThrowIfNull(outputDescriptor);
        if (projected.DType != TensorDType.Bfp8
            || projected.Device != TensorDevice.Cuda)
        {
            throw new InvalidOperationException(
                "The resident BFP8 ForgetMemory path requires a CUDA BFP8 tensor.");
        }
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batch);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(sequence);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(projectionWidth);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(keyWidth);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(valueWidth);
        if (projectionWidth != checked(2 * keyWidth + 3 * valueWidth)
            || projected.Numel
                != checked(batch * sequence * projectionWidth))
        {
            throw new ArgumentException(
                "ForgetMemory dimensions do not cover the BFP8 projection.");
        }
    }

    private static void TryReturnBFloat16(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<ushort>? buffer,
        ref List<Exception>? failures)
    {
        if (buffer is null)
            return;
        try
        {
            Tensor.ReturnCudaBFloat16Buffer(accelerator, buffer);
        }
        catch (Exception exception)
        {
            (failures ??= []).Add(exception);
        }
    }

    private static void TryCleanup(
        IDisposable? resource,
        ref List<Exception>? failures)
    {
        if (resource is null)
            return;
        try
        {
            resource.Dispose();
        }
        catch (Exception exception)
        {
            (failures ??= []).Add(exception);
        }
    }

    private static void RethrowWithCleanup(
        Exception failure,
        List<Exception>? cleanupFailures,
        string aggregateMessage)
    {
        if (cleanupFailures is not null)
        {
            cleanupFailures.Insert(0, failure);
            throw new AggregateException(aggregateMessage, cleanupFailures);
        }
        ExceptionDispatchInfo.Capture(failure).Throw();
    }
}

internal sealed class CudaBfp8ForgetMemoryResidentContext : IDisposable
{
    private CudaBfp8OwnedBuffers? _encodedOutput;
    private int _disposed;

    internal CudaBfp8ForgetMemoryResidentContext(
        int deviceIndex,
        NativeCudaDevice accelerator,
        NativeCudaBuffer<ushort> decodedProjection,
        NativeCudaBuffer<float> states,
        CudaBfp8OwnedBuffers encodedOutput)
    {
        DeviceIndex = deviceIndex;
        Accelerator = accelerator;
        DecodedProjection = decodedProjection;
        States = states;
        _encodedOutput = encodedOutput;
    }

    internal int DeviceIndex { get; }
    internal NativeCudaDevice Accelerator { get; }
    internal NativeCudaBuffer<ushort> DecodedProjection { get; }
    internal NativeCudaBuffer<float> States { get; }

    internal CudaBfp8OwnedBuffers DetachEncodedOutput()
    {
        ObjectDisposedException.ThrowIf(
            Volatile.Read(ref _disposed) != 0,
            this);
        return Interlocked.Exchange(ref _encodedOutput, null)
            ?? throw new InvalidOperationException(
                "The BFP8 ForgetMemory output was already detached.");
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;

        List<Exception>? failures = null;
        CudaBfp8OwnedBuffers? encodedOutput =
            Interlocked.Exchange(ref _encodedOutput, null);
        TryCleanup(encodedOutput, ref failures);
        try
        {
            Tensor.ReturnCudaBFloat16Buffer(
                Accelerator,
                DecodedProjection);
        }
        catch (Exception exception)
        {
            (failures ??= []).Add(exception);
        }
        TryCleanup(States, ref failures);
        GC.SuppressFinalize(this);

        if (failures is [Exception failure])
            ExceptionDispatchInfo.Capture(failure).Throw();
        if (failures is { Count: > 1 })
        {
            throw new AggregateException(
                "BFP8 ForgetMemory saved-context cleanup failed.",
                failures);
        }
    }

    private static void TryCleanup(
        IDisposable? resource,
        ref List<Exception>? failures)
    {
        if (resource is null)
            return;
        try
        {
            resource.Dispose();
        }
        catch (Exception exception)
        {
            (failures ??= []).Add(exception);
        }
    }
}
