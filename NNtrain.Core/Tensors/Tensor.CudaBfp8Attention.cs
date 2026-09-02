using System.Runtime.ExceptionServices;

namespace NNtrain;

internal static partial class TensorCudaKernels
{
    /// <summary>
    /// Runs the existing BF16 FlashAttention backend for a resident BFP8 QKV
    /// projection.  The decoded projection and BF16 output are owned by this
    /// autograd context rather than by a long-lived tensor decode cache.
    /// </summary>
    internal static Bfp8AttentionResidentContext
        AttentionForwardBfp8Resident(
            Tensor projected,
            Bfp8QuantizationDescriptor outputDescriptor,
            int batch,
            int sequence,
            int modelWidth,
            int numHeads,
            bool causal)
    {
        ArgumentNullException.ThrowIfNull(projected);
        ArgumentNullException.ThrowIfNull(outputDescriptor);
        if (projected.DType != TensorDType.Bfp8)
        {
            throw new InvalidOperationException(
                "The resident BFP8 attention path requires BFP8 QKV storage.");
        }

        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        int projectedLength = projected.Numel;
        int outputLength = checked(batch * sequence * modelWidth);
        int statisticsLength = checked(batch * numHeads * sequence);

        NativeCudaBuffer<ushort>? decodedProjection = null;
        NativeCudaBuffer<ushort>? decodedOutput = null;
        NativeCudaBuffer<float>? softmaxLogSumExp = null;
        NativeCudaBuffer<float>? rowDelta = null;
        CudaBfp8OwnedBuffers? encodedOutput = null;
        try
        {
            CudaBfp8BufferView projectedView =
                projected.EnsureCudaBfp8Buffer(deviceIndex);
            decodedProjection = Tensor.RentCudaBFloat16Buffer(
                deviceIndex,
                projectedLength);
            decodedOutput = Tensor.RentCudaBFloat16Buffer(
                deviceIndex,
                outputLength);
            softmaxLogSumExp = Tensor.RentCudaFloatBuffer(
                deviceIndex,
                statisticsLength);

            nint stream = accelerator.DefaultStream;
            CudaBfp8Native.DequantizeBFloat16(
                deviceIndex,
                projectedView.Payload,
                projectedView.Scales,
                decodedProjection,
                projectedView.Descriptor,
                stream);

            bool succeeded = CudaFlashAttention.TryForwardBFloat16(
                accelerator,
                decodedProjection,
                decodedOutput,
                softmaxLogSumExp,
                batch,
                sequence,
                modelWidth,
                numHeads,
                causal,
                out bool tensorCore);
            if (!succeeded)
            {
                throw new PlatformNotSupportedException(
                    "BFP8 attention requires the resident BF16 " +
                    "FlashAttention CUDA backend; CPU fallback is forbidden.");
            }

            if (tensorCore)
            {
                rowDelta = Tensor.RentCudaFloatBuffer(
                    deviceIndex,
                    statisticsLength);
            }

            encodedOutput = CudaBfp8OwnedBuffers.Allocate(
                accelerator,
                outputLength,
                outputDescriptor);
            CudaBfp8Native.QuantizeBFloat16(
                deviceIndex,
                decodedOutput,
                encodedOutput.Payload,
                encodedOutput.Scales,
                outputDescriptor,
                stream);
            // LSE/row-delta are the only forward workspaces that cannot be
            // reconstructed cheaply. QKV and the quantized attention output
            // remain authoritative BFP8 tensors, so release their full BF16
            // views now and decode them just-in-time during backward.
            Tensor.ReturnCudaBFloat16Buffer(accelerator, decodedOutput);
            decodedOutput = null;
            Tensor.ReturnCudaBFloat16Buffer(
                accelerator, decodedProjection);
            decodedProjection = null;

            var context = new Bfp8AttentionResidentContext(
                softmaxLogSumExp,
                rowDelta,
                encodedOutput,
                accelerator,
                tensorCore);
            softmaxLogSumExp = null;
            rowDelta = null;
            encodedOutput = null;
            return context;
        }
        catch (Exception failure)
        {
            List<Exception>? cleanupFailures = null;
            if (encodedOutput is not null)
                TryCleanup(encodedOutput.Dispose, ref cleanupFailures);
            TryReturnBFloat16(
                accelerator,
                decodedOutput,
                ref cleanupFailures);
            TryReturnBFloat16(
                accelerator,
                decodedProjection,
                ref cleanupFailures);
            TryReturnFloat(
                accelerator,
                rowDelta,
                ref cleanupFailures);
            TryReturnFloat(
                accelerator,
                softmaxLogSumExp,
                ref cleanupFailures);
            if (cleanupFailures is not null)
            {
                cleanupFailures.Insert(0, failure);
                throw new AggregateException(
                    "BFP8 attention forward and rollback failed.",
                    cleanupFailures);
            }
            ExceptionDispatchInfo.Capture(failure).Throw();
            throw;
        }
    }

    internal static void AttentionBackwardBfp8Resident(
        Tensor projected,
        Tensor output,
        Bfp8AttentionResidentContext context,
        int batch,
        int sequence,
        int modelWidth,
        int numHeads,
        bool causal)
    {
        ArgumentNullException.ThrowIfNull(projected);
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(context);
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);

        NativeCudaBuffer<ushort>? decodedProjection = null;
        NativeCudaBuffer<ushort>? decodedOutput = null;
        try
        {
            decodedProjection = Tensor.RentCudaBFloat16Buffer(
                deviceIndex, projected.Numel);
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
            CudaBfp8BufferView outputView =
                output.EnsureCudaBfp8Buffer(deviceIndex);
            NativeCudaBuffer<float> outputGradient =
                output.EnsureCudaGradientBuffer(deviceIndex);
            NativeCudaBuffer<float> projectedGradient =
                projected.EnsureCudaGradientBuffer(deviceIndex);

            bool directBfp8Output = context.RowDelta is not null
                && CudaFlashAttention.TryBackwardBFloat16Bfp8Output(
                    accelerator,
                    decodedProjection,
                    outputView,
                    outputGradient,
                    context.SoftmaxLogSumExp,
                    context.RowDelta,
                    projectedGradient,
                    batch,
                    sequence,
                    modelWidth,
                    numHeads,
                    causal,
                    context.TensorCore);

            // BFP8 is the storage boundary. FlashAttention still accumulates
            // its backward result in FP32; pure-BFP8 leaf publication happens
            // after the complete autograd contribution is accumulated.
            if (!directBfp8Output)
            {
                decodedOutput = Tensor.RentCudaBFloat16Buffer(
                    deviceIndex, output.Numel);
                CudaBfp8Native.DequantizeBFloat16(
                    deviceIndex,
                    outputView.Payload,
                    outputView.Scales,
                    decodedOutput,
                    outputView.Descriptor,
                    stream);
                CudaFlashAttention.BackwardBFloat16(
                    accelerator,
                    decodedProjection,
                    decodedOutput,
                    outputGradient,
                    outputGradientBFloat16: null,
                    context.SoftmaxLogSumExp,
                    context.RowDelta,
                    projectedGradient,
                    projectedGradientBFloat16: null,
                    batch,
                    sequence,
                    modelWidth,
                    numHeads,
                    causal,
                    context.TensorCore);
            }
            projected.MarkCudaGradientMutated(deviceIndex);
        }
        finally
        {
            if (decodedOutput is not null)
            {
                Tensor.ReturnCudaBFloat16Buffer(
                    accelerator, decodedOutput);
            }
            if (decodedProjection is not null)
            {
                Tensor.ReturnCudaBFloat16Buffer(
                    accelerator, decodedProjection);
            }
        }
    }

    internal sealed class Bfp8AttentionResidentContext : IDisposable
    {
        private readonly NativeCudaDevice _accelerator;
        private CudaBfp8OwnedBuffers? _encodedOutput;
        private int _disposed;

        internal Bfp8AttentionResidentContext(
            NativeCudaBuffer<float> softmaxLogSumExp,
            NativeCudaBuffer<float>? rowDelta,
            CudaBfp8OwnedBuffers encodedOutput,
            NativeCudaDevice accelerator,
            bool tensorCore)
        {
            SoftmaxLogSumExp = softmaxLogSumExp;
            RowDelta = rowDelta;
            _encodedOutput = encodedOutput;
            _accelerator = accelerator;
            TensorCore = tensorCore;
        }

        internal NativeCudaBuffer<float> SoftmaxLogSumExp { get; }
        internal NativeCudaBuffer<float>? RowDelta { get; }
        internal bool TensorCore { get; }

        internal CudaBfp8OwnedBuffers DetachEncodedOutput()
        {
            ObjectDisposedException.ThrowIf(
                Volatile.Read(ref _disposed) != 0,
                this);
            return Interlocked.Exchange(ref _encodedOutput, null)
                ?? throw new InvalidOperationException(
                    "The BFP8 attention output was already detached.");
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;

            List<Exception>? failures = null;
            CudaBfp8OwnedBuffers? encodedOutput =
                Interlocked.Exchange(ref _encodedOutput, null);
            if (encodedOutput is not null)
                TryCleanup(encodedOutput.Dispose, ref failures);
            TryReturnFloat(_accelerator, RowDelta, ref failures);
            TryReturnFloat(
                _accelerator,
                SoftmaxLogSumExp,
                ref failures);
            GC.SuppressFinalize(this);

            if (failures is [Exception failure])
                ExceptionDispatchInfo.Capture(failure).Throw();
            if (failures is { Count: > 1 })
            {
                throw new AggregateException(
                    "BFP8 attention saved-context cleanup failed.",
                    failures);
            }
        }
    }

    private static void TryReturnBFloat16(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<ushort>? buffer,
        ref List<Exception>? failures)
    {
        if (buffer is not null)
        {
            TryCleanup(
                () => Tensor.ReturnCudaBFloat16Buffer(accelerator, buffer),
                ref failures);
        }
    }

    private static void TryReturnFloat(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<float>? buffer,
        ref List<Exception>? failures)
    {
        if (buffer is not null)
        {
            TryCleanup(
                () => Tensor.ReturnCudaFloatBuffer(accelerator, buffer),
                ref failures);
        }
    }

    private static void TryCleanup(
        Action? cleanup,
        ref List<Exception>? failures)
    {
        if (cleanup is null)
            return;
        try
        {
            cleanup();
        }
        catch (Exception exception)
        {
            (failures ??= []).Add(exception);
        }
    }
}
