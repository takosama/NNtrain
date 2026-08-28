using System.Runtime.ExceptionServices;

namespace NNtrain;

internal static partial class TensorCudaKernels
{
    internal static CudaBfp8OwnedBuffers DropoutForwardBfp8Resident(
        Tensor input,
        Bfp8QuantizationDescriptor outputDescriptor,
        uint seed,
        uint dropThreshold,
        float scale)
        => ElementwiseForwardBfp8Resident(
            input,
            right: null,
            outputDescriptor,
            Bfp8ElementwiseOperation.Dropout,
            seed,
            dropThreshold,
            scale);

    internal static CudaBfp8OwnedBuffers DropoutForwardBfp8GraphResident(
        Tensor input,
        Bfp8QuantizationDescriptor outputDescriptor,
        CudaGraphDropoutToken token,
        float probability)
        => ElementwiseForwardBfp8Resident(
            input,
            right: null,
            outputDescriptor,
            Bfp8ElementwiseOperation.Dropout,
            seed: 0,
            dropThreshold: 0,
            scale: 1f,
            token,
            probability);

    internal static CudaBfp8OwnedBuffers AddDropoutForwardBfp8Resident(
        Tensor residual,
        Tensor branch,
        Bfp8QuantizationDescriptor outputDescriptor,
        uint seed,
        uint dropThreshold,
        float scale)
        => ElementwiseForwardBfp8Resident(
            residual,
            branch,
            outputDescriptor,
            Bfp8ElementwiseOperation.AddDropout,
            seed,
            dropThreshold,
            scale);

    internal static CudaBfp8OwnedBuffers AddDropoutForwardBfp8GraphResident(
        Tensor residual,
        Tensor branch,
        Bfp8QuantizationDescriptor outputDescriptor,
        CudaGraphDropoutToken token,
        float probability)
        => ElementwiseForwardBfp8Resident(
            residual,
            branch,
            outputDescriptor,
            Bfp8ElementwiseOperation.AddDropout,
            seed: 0,
            dropThreshold: 0,
            scale: 1f,
            token,
            probability);

    internal static CudaBfp8OwnedBuffers AddForwardBfp8Resident(
        Tensor left,
        Tensor right,
        Bfp8QuantizationDescriptor outputDescriptor)
        => ElementwiseForwardBfp8Resident(
            left,
            right,
            outputDescriptor,
            Bfp8ElementwiseOperation.Add,
            seed: 0,
            dropThreshold: 0,
            scale: 1f);

    private static CudaBfp8OwnedBuffers ElementwiseForwardBfp8Resident(
        Tensor left,
        Tensor? right,
        Bfp8QuantizationDescriptor outputDescriptor,
        Bfp8ElementwiseOperation operation,
        uint seed,
        uint dropThreshold,
        float scale,
        CudaGraphDropoutToken? graphToken = null,
        float graphProbability = 0f)
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(outputDescriptor);
        RequireBfp8ElementwiseOperand(left, nameof(left));
        if (right is not null)
        {
            RequireBfp8ElementwiseOperand(right, nameof(right));
            if (right.Numel != left.Numel)
            {
                throw new ArgumentException(
                    "Resident BFP8 elementwise operands must have equal " +
                    "element counts.",
                    nameof(right));
            }
        }
        if (operation != Bfp8ElementwiseOperation.Dropout && right is null)
        {
            throw new ArgumentException(
                "A binary BFP8 elementwise operation requires two operands.",
                nameof(right));
        }

        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        NativeCudaBuffer<ushort>? decodedLeft = null;
        NativeCudaBuffer<ushort>? decodedRight = null;
        NativeCudaBuffer<ushort>? encodedBFloat16Output = null;
        CudaBfp8OwnedBuffers? encodedOutput = null;
        CudaBfp8OwnedBuffers? completedOutput = null;
        Exception? operationFailure = null;
        try
        {
            decodedLeft = Tensor.RentCudaBFloat16Buffer(
                deviceIndex, left.Numel);
            if (right is not null)
            {
                decodedRight = Tensor.RentCudaBFloat16Buffer(
                    deviceIndex, right.Numel);
            }
            encodedBFloat16Output = Tensor.RentCudaBFloat16Buffer(
                deviceIndex, left.Numel);
            encodedOutput = CudaBfp8OwnedBuffers.Allocate(
                accelerator,
                left.Numel,
                outputDescriptor);

            nint stream = accelerator.DefaultStream;
            DecodeBfp8ElementwiseOperand(
                left, decodedLeft, deviceIndex, stream);
            if (right is not null)
            {
                DecodeBfp8ElementwiseOperand(
                    right,
                    decodedRight
                        ?? throw new InvalidOperationException(
                            "The second BFP8 decode buffer is missing."),
                    deviceIndex,
                    stream);
            }

            switch (operation)
            {
                case Bfp8ElementwiseOperation.Dropout:
                    if (graphToken is { } dropoutToken)
                    {
                        DropoutForwardBFloat16Graph(
                            decodedLeft,
                            encodedBFloat16Output,
                            left.Numel,
                            dropoutToken,
                            graphProbability);
                    }
                    else
                    {
                        CudaTensorNative.Dropout(
                            deviceIndex,
                            decodedLeft.NativePtr,
                            encodedBFloat16Output.NativePtr,
                            left.Numel,
                            seed,
                            dropThreshold,
                            scale,
                            bfloat16: true);
                    }
                    break;
                case Bfp8ElementwiseOperation.AddDropout:
                    if (graphToken is { } addDropoutToken)
                    {
                        AddDropoutForwardBFloat16Graph(
                            decodedLeft,
                            decodedRight!,
                            encodedBFloat16Output,
                            left.Numel,
                            addDropoutToken,
                            graphProbability);
                    }
                    else
                    {
                        CudaTensorNative.AddDropout(
                            deviceIndex,
                            decodedLeft.NativePtr,
                            decodedRight!.NativePtr,
                            encodedBFloat16Output.NativePtr,
                            left.Numel,
                            seed,
                            dropThreshold,
                            scale,
                            bfloat16: true);
                    }
                    break;
                case Bfp8ElementwiseOperation.Add:
                    CudaTensorNative.Add(
                        deviceIndex,
                        decodedLeft.NativePtr,
                        decodedRight!.NativePtr,
                        encodedBFloat16Output.NativePtr,
                        left.Numel,
                        bfloat16: true);
                    break;
                default:
                    throw new InvalidOperationException(
                        $"Unknown BFP8 elementwise operation '{operation}'.");
            }

            CudaBfp8Native.QuantizeBFloat16(
                deviceIndex,
                encodedBFloat16Output,
                encodedOutput.Payload,
                encodedOutput.Scales,
                outputDescriptor,
                stream);

            completedOutput = encodedOutput;
            encodedOutput = null;
            return completedOutput;
        }
        catch (Exception failure)
        {
            operationFailure = failure;
            throw;
        }
        finally
        {
            List<Exception>? cleanupFailures = null;
            if (encodedOutput is not null)
            {
                TryCleanupBfp8LossDropout(
                    encodedOutput.Dispose,
                    ref cleanupFailures);
            }
            TryReturnBfp8LossDropoutBFloat16(
                accelerator,
                encodedBFloat16Output,
                ref cleanupFailures);
            TryReturnBfp8LossDropoutBFloat16(
                accelerator,
                decodedRight,
                ref cleanupFailures);
            TryReturnBfp8LossDropoutBFloat16(
                accelerator,
                decodedLeft,
                ref cleanupFailures);
            if (cleanupFailures is not null
                && completedOutput is not null)
            {
                // A successful kernel whose scratch cleanup fails cannot
                // publish an output that the caller never receives.
                TryCleanupBfp8LossDropout(
                    completedOutput.Dispose,
                    ref cleanupFailures);
            }
            ThrowCleanupFailures(
                "BFP8 elementwise forward cleanup failed.",
                operationFailure,
                cleanupFailures);
        }
    }

    internal static Bfp8CrossEntropyResidentContext
        CrossEntropyForwardBfp8Resident(
            Tensor logits,
            int[] labels,
            int rows,
            int columns,
            int ignoreIndex,
            int validRows,
            float labelSmoothing)
    {
        ArgumentNullException.ThrowIfNull(logits);
        ArgumentNullException.ThrowIfNull(labels);
        RequireBfp8ElementwiseOperand(logits, nameof(logits));

        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        NativeCudaBuffer<ushort>? decodedLogits = null;
        NativeCudaBuffer<int>? labelsBuffer = null;
        NativeCudaBuffer<float>? maxima = null;
        NativeCudaBuffer<float>? inverseSums = null;
        NativeCudaBuffer<float>? rowLosses = null;
        NativeCudaBuffer<float>? loss = null;
        try
        {
            decodedLogits = Tensor.RentCudaBFloat16Buffer(
                deviceIndex, logits.Numel);
            labelsBuffer = Tensor.RentCudaIntBuffer(deviceIndex, labels);
            maxima = Tensor.RentCudaFloatBuffer(deviceIndex, rows);
            inverseSums = Tensor.RentCudaFloatBuffer(deviceIndex, rows);
            rowLosses = Tensor.RentCudaFloatBuffer(deviceIndex, rows);
            loss = Tensor.RentCudaFloatBuffer(deviceIndex, 1);

            CudaBfp8BufferView logitsView =
                logits.EnsureCudaBfp8Buffer(deviceIndex);
            CudaBfp8Native.DequantizeBFloat16(
                deviceIndex,
                logitsView.Payload,
                logitsView.Scales,
                decodedLogits,
                logitsView.Descriptor,
                accelerator.DefaultStream);
            CudaTensorNative.CrossEntropy(
                deviceIndex,
                decodedLogits.NativePtr,
                labelsBuffer.NativePtr,
                maxima.NativePtr,
                inverseSums.NativePtr,
                rowLosses.NativePtr,
                loss.NativePtr,
                rows,
                columns,
                ignoreIndex,
                validRows,
                labelSmoothing,
                bfloat16: true);

            // Backward can deterministically decode the authoritative BFP8
            // logits again.  Releasing this production-sized BF16 view here
            // avoids keeping it alive across the entire forward/backward
            // boundary (about 424 MiB per shard for batch 36, sequence 512,
            // vocabulary 11,500).  Row losses are also forward-only.
            Tensor.ReturnCudaBFloat16Buffer(accelerator, decodedLogits);
            decodedLogits = null;
            Tensor.ReturnCudaFloatBuffer(accelerator, rowLosses);
            rowLosses = null;

            var context = new Bfp8CrossEntropyResidentContext(
                labelsBuffer,
                maxima,
                inverseSums,
                loss,
                accelerator);
            labelsBuffer = null;
            maxima = null;
            inverseSums = null;
            loss = null;
            return context;
        }
        catch (Exception failure)
        {
            List<Exception>? cleanupFailures = null;
            TryReturnBfp8LossDropoutFloat(
                accelerator, loss, ref cleanupFailures);
            TryReturnBfp8LossDropoutFloat(
                accelerator, rowLosses, ref cleanupFailures);
            TryReturnBfp8LossDropoutFloat(
                accelerator, inverseSums, ref cleanupFailures);
            TryReturnBfp8LossDropoutFloat(
                accelerator, maxima, ref cleanupFailures);
            TryReturnBfp8LossDropoutInt(
                accelerator, labelsBuffer, ref cleanupFailures);
            TryReturnBfp8LossDropoutBFloat16(
                accelerator, decodedLogits, ref cleanupFailures);
            ThrowCleanupFailures(
                "BFP8 cross entropy forward and rollback failed.",
                failure,
                cleanupFailures);
            throw;
        }
    }

    internal static void CrossEntropyBackwardBfp8Resident(
        Tensor logits,
        Tensor loss,
        Bfp8CrossEntropyResidentContext context,
        int columns,
        int ignoreIndex,
        int validRows,
        float labelSmoothing)
    {
        ArgumentNullException.ThrowIfNull(logits);
        ArgumentNullException.ThrowIfNull(loss);
        ArgumentNullException.ThrowIfNull(context);
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        NativeCudaBuffer<ushort> decodedLogits =
            Tensor.RentCudaBFloat16Buffer(deviceIndex, logits.Numel);
        try
        {
            CudaBfp8BufferView logitsView =
                logits.EnsureCudaBfp8Buffer(deviceIndex);
            CudaBfp8Native.DequantizeBFloat16(
                deviceIndex,
                logitsView.Payload,
                logitsView.Scales,
                decodedLogits,
                logitsView.Descriptor,
                accelerator.DefaultStream);
            CudaTensorNative.CrossEntropyBackward(
                deviceIndex,
                decodedLogits.NativePtr,
                context.Maxima.NativePtr,
                context.InverseSums.NativePtr,
                context.Labels.NativePtr,
                logits.EnsureCudaGradientBuffer(deviceIndex).NativePtr,
                loss.EnsureCudaGradientBuffer(deviceIndex).NativePtr,
                logits.Numel,
                columns,
                ignoreIndex,
                validRows,
                labelSmoothing,
                bfloat16: true);
            logits.MarkCudaGradientMutated(deviceIndex);
        }
        finally
        {
            Tensor.ReturnCudaBFloat16Buffer(accelerator, decodedLogits);
        }
    }

    private static void RequireBfp8ElementwiseOperand(
        Tensor tensor,
        string parameterName)
    {
        if (tensor.DType != TensorDType.Bfp8)
        {
            throw new ArgumentException(
                "The resident CUDA path requires BFP8 operands.",
                parameterName);
        }
    }

    private static void DecodeBfp8ElementwiseOperand(
        Tensor tensor,
        NativeCudaBuffer<ushort> destination,
        int deviceIndex,
        nint stream)
    {
        CudaBfp8BufferView view = tensor.EnsureCudaBfp8Buffer(deviceIndex);
        CudaBfp8Native.DequantizeBFloat16(
            deviceIndex,
            view.Payload,
            view.Scales,
            destination,
            view.Descriptor,
            stream);
    }

    internal sealed class Bfp8CrossEntropyResidentContext : IDisposable
    {
        private readonly NativeCudaDevice _accelerator;
        private NativeCudaBuffer<float>? _loss;
        private int _disposed;

        internal Bfp8CrossEntropyResidentContext(
            NativeCudaBuffer<int> labels,
            NativeCudaBuffer<float> maxima,
            NativeCudaBuffer<float> inverseSums,
            NativeCudaBuffer<float> loss,
            NativeCudaDevice accelerator)
        {
            Labels = labels;
            Maxima = maxima;
            InverseSums = inverseSums;
            _loss = loss;
            _accelerator = accelerator;
        }

        internal NativeCudaBuffer<int> Labels { get; }
        internal NativeCudaBuffer<float> Maxima { get; }
        internal NativeCudaBuffer<float> InverseSums { get; }

        internal NativeCudaBuffer<float> DetachLoss()
        {
            ObjectDisposedException.ThrowIf(
                Volatile.Read(ref _disposed) != 0,
                this);
            return Interlocked.Exchange(ref _loss, null)
                ?? throw new InvalidOperationException(
                    "The BFP8 cross entropy loss was already detached.");
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;

            List<Exception>? failures = null;
            TryReturnBfp8LossDropoutFloat(
                _accelerator,
                Interlocked.Exchange(ref _loss, null),
                ref failures);
            TryReturnBfp8LossDropoutFloat(
                _accelerator, InverseSums, ref failures);
            TryReturnBfp8LossDropoutFloat(
                _accelerator, Maxima, ref failures);
            TryReturnBfp8LossDropoutInt(
                _accelerator, Labels, ref failures);
            GC.SuppressFinalize(this);

            if (failures is [Exception failure])
                ExceptionDispatchInfo.Capture(failure).Throw();
            if (failures is { Count: > 1 })
            {
                throw new AggregateException(
                    "BFP8 cross entropy saved-context cleanup failed.",
                    failures);
            }
        }
    }

    private static void TryReturnBfp8LossDropoutBFloat16(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<ushort>? buffer,
        ref List<Exception>? failures)
    {
        if (buffer is not null)
        {
            TryCleanupBfp8LossDropout(
                () => Tensor.ReturnCudaBFloat16Buffer(accelerator, buffer),
                ref failures);
        }
    }

    private static void TryReturnBfp8LossDropoutFloat(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<float>? buffer,
        ref List<Exception>? failures)
    {
        if (buffer is not null)
        {
            TryCleanupBfp8LossDropout(
                () => Tensor.ReturnCudaFloatBuffer(accelerator, buffer),
                ref failures);
        }
    }

    private static void TryReturnBfp8LossDropoutInt(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<int>? buffer,
        ref List<Exception>? failures)
    {
        if (buffer is not null)
        {
            TryCleanupBfp8LossDropout(
                () => Tensor.ReturnCudaIntBuffer(accelerator, buffer),
                ref failures);
        }
    }

    private static void TryCleanupBfp8LossDropout(
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

    private static void ThrowCleanupFailures(
        string message,
        Exception? operationFailure,
        List<Exception>? cleanupFailures)
    {
        if (cleanupFailures is null)
            return;
        if (operationFailure is not null)
            cleanupFailures.Insert(0, operationFailure);
        throw new AggregateException(message, cleanupFailures);
    }

    private enum Bfp8ElementwiseOperation
    {
        Dropout,
        AddDropout,
        Add,
    }
}

public partial class Tensor
{
    private Tensor CrossEntropyWithLogitsBfp8Cuda(
        int[] labels,
        int rows,
        int columns,
        int ignoreIndex,
        int validRows,
        float labelSmoothing)
    {
        int deviceIndex = CudaDeviceIndex;
        TensorCudaKernels.Bfp8CrossEntropyResidentContext context =
            CudaOperationProfiler.IsEnabled
                ? CudaOperationProfiler.Measure(
                    "forward.cross_entropy",
                    () => TensorCudaKernels
                        .CrossEntropyForwardBfp8Resident(
                            this,
                            labels,
                            rows,
                            columns,
                            ignoreIndex,
                            validRows,
                            labelSmoothing))
                : TensorCudaKernels.CrossEntropyForwardBfp8Resident(
                    this,
                    labels,
                    rows,
                    columns,
                    ignoreIndex,
                    validRows,
                    labelSmoothing);

        Tensor result;
        try
        {
            NativeCudaBuffer<float> loss = context.DetachLoss();
            try
            {
                result = FromCudaResult(
                    loss,
                    deviceIndex,
                    [1],
                    [this],
                    dtype: TensorDType.Float32);
            }
            catch
            {
                ReturnCudaFloatBuffer(
                    ForgetMemoryV2Cuda.GetAccelerator(deviceIndex),
                    loss);
                throw;
            }
        }
        catch (Exception conversionFailure)
        {
            DisposeBfp8LossContextAfterConversionFailure(
                context,
                conversionFailure);
            throw;
        }

        if (AutogradContext.IsRecordingEnabled)
        {
            AutogradLease<TensorCudaKernels
                .Bfp8CrossEntropyResidentContext> lease =
                    AutogradLease<TensorCudaKernels
                        .Bfp8CrossEntropyResidentContext>.Own(
                            context,
                            AutogradLeaseMetadata.CudaOwned(
                                deviceIndex,
                                TensorDType.Bfp8,
                                DataVersion),
                            static saved => saved.Dispose());
            result.Node.SetBackward(lease, savedContext =>
            {
                void Backward() => TensorCudaKernels
                    .CrossEntropyBackwardBfp8Resident(
                        this,
                        result,
                        savedContext,
                        columns,
                        ignoreIndex,
                        validRows,
                        labelSmoothing);
                if (CudaOperationProfiler.IsEnabled)
                    CudaOperationProfiler.Measure(
                        "backward.cross_entropy", Backward);
                else
                    Backward();
            });
        }
        else if (!CudaInferenceScope.TrackResource(context))
        {
            context.Dispose();
        }
        return result;
    }

    private Tensor DropoutBfp8Cuda(
        uint seed,
        uint dropThreshold,
        float scale,
        CudaGraphDropoutToken? graphToken = null,
        float graphProbability = 0f)
    {
        int deviceIndex = CudaDeviceIndex;
        Bfp8QuantizationDescriptor descriptor =
            SelectBfp8ResultDescriptor(this);
        using CudaBfp8OwnedBuffers output =
            CudaOperationProfiler.IsEnabled
                ? CudaOperationProfiler.Measure(
                    "forward.dropout",
                    Forward)
                : Forward();
        CudaBfp8OwnedBuffers Forward()
            => graphToken is { } token
                ? TensorCudaKernels.DropoutForwardBfp8GraphResident(
                    this, descriptor, token, graphProbability)
                : TensorCudaKernels.DropoutForwardBfp8Resident(
                    this, descriptor, seed, dropThreshold, scale);
        Tensor result = FromCudaBfp8Result(
            output,
            deviceIndex,
            _shape,
            [this]);
        if (AutogradContext.IsRecordingEnabled)
        {
            result.Node.BackwardAction = () =>
            {
                void Backward()
                {
                    if (graphToken is { } token)
                    {
                        TensorCudaKernels.DropoutBackwardGraphResident(
                            result, this, token, graphProbability);
                    }
                    else
                    {
                        TensorCudaKernels.DropoutBackwardResident(
                            result,
                            this,
                            seed,
                            dropThreshold,
                            scale);
                    }
                }
                if (CudaOperationProfiler.IsEnabled)
                    CudaOperationProfiler.Measure(
                        "backward.dropout", Backward);
                else
                    Backward();
            };
        }
        return result;
    }

    private Tensor AddDropoutBfp8Cuda(
        Tensor branch,
        uint seed,
        uint dropThreshold,
        float scale,
        CudaGraphDropoutToken? graphToken = null,
        float graphProbability = 0f)
    {
        int deviceIndex = CudaDeviceIndex;
        Bfp8QuantizationDescriptor descriptor =
            SelectBfp8ResultDescriptor(this, branch);
        using CudaBfp8OwnedBuffers output =
            CudaOperationProfiler.IsEnabled
                ? CudaOperationProfiler.Measure(
                    "forward.residual_dropout",
                    Forward)
                : Forward();
        CudaBfp8OwnedBuffers Forward()
            => graphToken is { } token
                ? TensorCudaKernels.AddDropoutForwardBfp8GraphResident(
                    this, branch, descriptor, token, graphProbability)
                : TensorCudaKernels.AddDropoutForwardBfp8Resident(
                    this, branch, descriptor, seed, dropThreshold, scale);
        Tensor result = FromCudaBfp8Result(
            output,
            deviceIndex,
            _shape,
            [this, branch]);
        if (AutogradContext.IsRecordingEnabled)
        {
            result.Node.BackwardAction = () =>
            {
                void Backward()
                {
                    bool sameParent = ReferenceEquals(this, branch);
                    if (graphToken is { } token)
                    {
                        TensorCudaKernels.AddDropoutBackwardGraphResident(
                            result,
                            this,
                            branch,
                            sameParent,
                            token,
                            graphProbability);
                    }
                    else
                    {
                        TensorCudaKernels.AddDropoutBackwardResident(
                            result,
                            this,
                            branch,
                            sameParent,
                            seed,
                            dropThreshold,
                            scale);
                    }
                }
                if (CudaOperationProfiler.IsEnabled)
                {
                    CudaOperationProfiler.Measure(
                        "backward.residual_dropout", Backward);
                }
                else
                {
                    Backward();
                }
            };
        }
        return result;
    }

    private static Tensor AddBfp8Cuda(
        Tensor left,
        Tensor right,
        int[] resultShape)
    {
        int deviceIndex = CudaDeviceIndex;
        Bfp8QuantizationDescriptor descriptor =
            SelectBfp8ResultDescriptor(left, right);
        using CudaBfp8OwnedBuffers output =
            TensorCudaKernels.AddForwardBfp8Resident(
                left,
                right,
                descriptor);
        Tensor result = FromCudaBfp8Result(
            output,
            deviceIndex,
            resultShape,
            [left, right]);
        if (AutogradContext.IsRecordingEnabled)
        {
            result.Node.BackwardAction = () =>
                TensorCudaKernels.AddBackwardResident(
                    result,
                    left,
                    right);
        }
        return result;
    }

    private static void DisposeBfp8LossContextAfterConversionFailure(
        IDisposable context,
        Exception conversionFailure)
    {
        try
        {
            context.Dispose();
        }
        catch (Exception cleanupFailure)
        {
            throw new AggregateException(
                "BFP8 cross entropy result construction failed.",
                conversionFailure,
                cleanupFailure);
        }
        ExceptionDispatchInfo.Capture(conversionFailure).Throw();
    }
}
