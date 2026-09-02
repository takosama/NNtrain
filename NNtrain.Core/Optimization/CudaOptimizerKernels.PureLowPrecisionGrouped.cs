using NNtrain.Cuda.Interop;

namespace NNtrain;

/// <summary>
/// Batched fixed-NS5 finish paths for the pure low-precision NekoMuon
/// policies. Moment preparation and persistent-state quantization remain in
/// their precision-specific paths; only the FP32 Newton-Schulz workspace is
/// grouped, matching the existing mixed-precision implementation.
/// </summary>
internal static partial class CudaOptimizerKernels
{
    internal sealed record NekoMuonBFloat16BatchItem(
        Tensor Parameter,
        NekoMuonBFloat16ResidentState State,
        int OriginalRows,
        int OriginalColumns,
        bool ApplyWeightDecay);

    internal sealed record NekoMuonBfp8BatchItem(
        Tensor Parameter,
        NekoMuonBfp8ResidentState State,
        int OriginalRows,
        int OriginalColumns,
        bool ApplyWeightDecay);

    internal static void
        NekoMuonFinishFixedFiveBFloat16GroupedDeviceResident(
            int deviceIndex,
            IReadOnlyList<NekoMuonBFloat16BatchItem> items,
            NekoMuonDeviceScratch scratch,
            NativeCudaBuffer<int> finiteStatus,
            float fastCorrection,
            float epsilon,
            float coefficientA,
            float coefficientB,
            float coefficientC,
            float learningRate,
            float weightDecay,
            bool nesterov = false)
    {
        ArgumentNullException.ThrowIfNull(items);
        if (items.Count == 0)
            return;

        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        foreach (IGrouping<(int Rows, int Columns),
            NekoMuonBFloat16BatchItem> group in items.GroupBy(item => (
                Math.Min(item.OriginalRows, item.OriginalColumns),
                Math.Max(item.OriginalRows, item.OriginalColumns))))
        {
            NekoMuonBFloat16BatchItem[] grouped = group.ToArray();
            for (int offset = 0; offset < grouped.Length;
                offset += scratch.BatchCapacity)
            {
                int count = Math.Min(
                    scratch.BatchCapacity,
                    grouped.Length - offset);
                if (count == 1
                    || group.Key.Rows <= DirectNewtonSchulzMaxRows)
                {
                    for (int slot = 0; slot < count; slot++)
                    {
                        NekoMuonBFloat16BatchItem item =
                            grouped[offset + slot];
                        NekoMuonFixedNs5Telemetry.RecordScalar(
                            group.Key.Rows);
                        _ = NekoMuonFinishBFloat16StepResident(
                            item.Parameter,
                            deviceIndex,
                            item.State,
                            scratch,
                            finiteStatus,
                            item.OriginalRows,
                            item.OriginalColumns,
                            fastCorrection,
                            epsilon,
                            previousConfidence: 0f,
                            rho: 0f,
                            maxNewtonSchulzSteps: 5,
                            NekoMuonNewtonSchulzDepthMode.Fixed,
                            configuredDepth: 5f,
                            runNewtonSchulz: true,
                            coefficientA,
                            coefficientB,
                            coefficientC,
                            learningRate,
                            weightDecay,
                            item.ApplyWeightDecay,
                            deviceOnlyFixedFive: true,
                            forceFullNewtonSchulz: false,
                            nesterov: nesterov);
                    }
                    continue;
                }

                FinishNekoMuonFixedFiveBFloat16Batch(
                    accelerator,
                    deviceIndex,
                    grouped.AsSpan(offset, count),
                    scratch,
                    finiteStatus,
                    nesterov ? 1f : 1f / fastCorrection,
                    epsilon,
                    coefficientA,
                    coefficientB,
                    coefficientC,
                    learningRate,
                    weightDecay,
                    nesterov);
            }
        }
    }

    private static void FinishNekoMuonFixedFiveBFloat16Batch(
        NativeCudaDevice accelerator,
        int deviceIndex,
        ReadOnlySpan<NekoMuonBFloat16BatchItem> items,
        NekoMuonDeviceScratch scratch,
        NativeCudaBuffer<int> finiteStatus,
        float inverseFastCorrection,
        float epsilon,
        float coefficientA,
        float coefficientB,
        float coefficientC,
        float learningRate,
        float weightDecay,
        bool nesterov)
    {
        int count = items.Length;
        int rows = Math.Min(
            items[0].OriginalRows,
            items[0].OriginalColumns);
        int columns = Math.Max(
            items[0].OriginalRows,
            items[0].OriginalColumns);
        int length = checked(rows * columns);
        NekoMuonFixedNs5Telemetry.RecordBatch(count);

        for (int slot = 0; slot < count; slot++)
        {
            NekoMuonBFloat16BatchItem item = items[slot];
            NekoMuonBFloat16ResidentState.NekoBuffers buffers =
                item.State.GetOrCreate(deviceIndex);
            CudaOptimizerNative.NekoInitializeBFloat16FromDeviceStats(
                deviceIndex,
                (nesterov ? buffers.Slow : buffers.Fast).NativePtr,
                AddFloatOffset(scratch.X.NativePtr, slot * length),
                length,
                item.OriginalRows,
                item.OriginalColumns,
                item.OriginalRows > item.OriginalColumns,
                inverseFastCorrection,
                buffers.Stats.NativePtr,
                epsilon,
                finiteStatus.NativePtr);
        }

        nint x = scratch.X.NativePtr;
        nint next = scratch.Next.NativePtr;
        for (int step = 0; step < 5; step++)
        {
            NekoMuonNewtonSchulzBatched(
                accelerator,
                deviceIndex,
                x,
                next,
                scratch.Gram.NativePtr,
                scratch.GramSquared.NativePtr,
                rows,
                columns,
                count,
                coefficientA,
                coefficientB,
                coefficientC,
                scratch.UseBFloat16TensorCores);
            (x, next) = (next, x);
        }

        for (int slot = 0; slot < count; slot++)
        {
            NekoMuonBFloat16BatchItem item = items[slot];
            nint update = AddFloatOffset(x, slot * length);
            if (item.OriginalRows > item.OriginalColumns)
            {
                nint transposed = AddFloatOffset(next, slot * length);
                CudaOptimizerNative.NekoTransposeBack(
                    deviceIndex,
                    update,
                    transposed,
                    length,
                    item.OriginalRows,
                    item.OriginalColumns);
                update = transposed;
            }
            NativeCudaBuffer<ushort> data =
                item.Parameter.EnsureCudaBFloat16Buffer(deviceIndex);
            float finalScale = MathF.Sqrt(MathF.Max(
                1f,
                (float)item.OriginalRows / item.OriginalColumns));
            CudaOptimizerNative.NekoApplyBFloat16(
                deviceIndex,
                data.NativePtr,
                update,
                length,
                learningRate,
                finalScale,
                weightDecay,
                item.ApplyWeightDecay);
        }
    }

    internal static void NekoMuonFinishFixedFiveBfp8GroupedDeviceResident(
        int deviceIndex,
        IReadOnlyList<NekoMuonBfp8BatchItem> items,
        NekoMuonDeviceScratch scratch,
        NativeCudaBuffer<int> finiteStatus,
        float fastCorrection,
        float epsilon,
        float coefficientA,
        float coefficientB,
        float coefficientC,
        float learningRate,
        float weightDecay,
        bool nesterov = false)
    {
        ArgumentNullException.ThrowIfNull(items);
        if (items.Count == 0)
            return;

        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        foreach (IGrouping<(int Rows, int Columns), NekoMuonBfp8BatchItem>
            group in items.GroupBy(item => (
                Math.Min(item.OriginalRows, item.OriginalColumns),
                Math.Max(item.OriginalRows, item.OriginalColumns))))
        {
            NekoMuonBfp8BatchItem[] grouped = group.ToArray();
            for (int offset = 0; offset < grouped.Length;
                offset += scratch.BatchCapacity)
            {
                int count = Math.Min(
                    scratch.BatchCapacity,
                    grouped.Length - offset);
                if (count == 1
                    || group.Key.Rows <= DirectNewtonSchulzMaxRows)
                {
                    for (int slot = 0; slot < count; slot++)
                    {
                        NekoMuonBfp8BatchItem item = grouped[offset + slot];
                        NekoMuonFixedNs5Telemetry.RecordScalar(
                            group.Key.Rows);
                        _ = NekoMuonFinishBfp8StepResident(
                            item.Parameter,
                            deviceIndex,
                            item.State,
                            scratch,
                            finiteStatus,
                            item.OriginalRows,
                            item.OriginalColumns,
                            fastCorrection,
                            epsilon,
                            previousConfidence: 0f,
                            rho: 0f,
                            maxNewtonSchulzSteps: 5,
                            NekoMuonNewtonSchulzDepthMode.Fixed,
                            configuredNewtonSchulzDepth: 5f,
                            runNewtonSchulz: true,
                            coefficientA,
                            coefficientB,
                            coefficientC,
                            learningRate,
                            weightDecay,
                            item.ApplyWeightDecay,
                            deviceOnlyFixedFive: true,
                            mixedBlockState: false,
                            forceFullNewtonSchulz: false,
                            nesterov: nesterov);
                    }
                    continue;
                }

                FinishNekoMuonFixedFiveBfp8Batch(
                    accelerator,
                    deviceIndex,
                    grouped.AsSpan(offset, count),
                    scratch,
                    finiteStatus,
                    nesterov ? 1f : 1f / fastCorrection,
                    epsilon,
                    coefficientA,
                    coefficientB,
                    coefficientC,
                    learningRate,
                    weightDecay,
                    nesterov);
            }
        }
    }

    private static void FinishNekoMuonFixedFiveBfp8Batch(
        NativeCudaDevice accelerator,
        int deviceIndex,
        ReadOnlySpan<NekoMuonBfp8BatchItem> items,
        NekoMuonDeviceScratch scratch,
        NativeCudaBuffer<int> finiteStatus,
        float inverseFastCorrection,
        float epsilon,
        float coefficientA,
        float coefficientB,
        float coefficientC,
        float learningRate,
        float weightDecay,
        bool nesterov)
    {
        int count = items.Length;
        int rows = Math.Min(
            items[0].OriginalRows,
            items[0].OriginalColumns);
        int columns = Math.Max(
            items[0].OriginalRows,
            items[0].OriginalColumns);
        int length = checked(rows * columns);
        nint stream = accelerator.DefaultStream;
        NekoMuonFixedNs5Telemetry.RecordBatch(count);

        for (int slot = 0; slot < count; slot++)
        {
            NekoMuonBfp8BatchItem item = items[slot];
            NekoMuonBfp8ResidentState.NekoBuffers buffers =
                item.State.GetOrCreate(deviceIndex);
            nint decodedFast = AddFloatOffset(
                scratch.Next.NativePtr,
                slot * length);
            DequantizeBfp8ToPointer(
                deviceIndex,
                nesterov ? buffers.Slow : buffers.Fast,
                decodedFast,
                length,
                stream);
            CudaOptimizerNative.NekoInitializeFromDeviceStats(
                deviceIndex,
                decodedFast,
                AddFloatOffset(scratch.X.NativePtr, slot * length),
                length,
                item.OriginalRows,
                item.OriginalColumns,
                item.OriginalRows > item.OriginalColumns,
                inverseFastCorrection,
                buffers.Stats.NativePtr,
                epsilon,
                finiteStatus.NativePtr);
        }

        nint x = scratch.X.NativePtr;
        nint next = scratch.Next.NativePtr;
        for (int step = 0; step < 5; step++)
        {
            NekoMuonNewtonSchulzBatched(
                accelerator,
                deviceIndex,
                x,
                next,
                scratch.Gram.NativePtr,
                scratch.GramSquared.NativePtr,
                rows,
                columns,
                count,
                coefficientA,
                coefficientB,
                coefficientC,
                scratch.UseBFloat16TensorCores);
            (x, next) = (next, x);
        }

        CudaOptimizerNative.AccumulateFiniteStatus(
            deviceIndex,
            x,
            checked(length * count),
            finiteStatus.NativePtr);
        NekoMuonDeviceScratch.Bfp8Buffers workspace =
            scratch.GetBfp8Buffers(length);
        for (int slot = 0; slot < count; slot++)
        {
            NekoMuonBfp8BatchItem item = items[slot];
            nint update = AddFloatOffset(x, slot * length);
            if (item.OriginalRows > item.OriginalColumns)
            {
                nint transposed = AddFloatOffset(next, slot * length);
                CudaOptimizerNative.NekoTransposeBack(
                    deviceIndex,
                    update,
                    transposed,
                    length,
                    item.OriginalRows,
                    item.OriginalColumns);
                update = transposed;
            }

            CudaBfp8BufferView data =
                item.Parameter.EnsureCudaBfp8Buffer(deviceIndex);
            CudaBfp8Native.DequantizeFloat32(
                deviceIndex,
                data.Payload,
                data.Scales,
                workspace.Data,
                data.Descriptor,
                stream);
            float finalScale = MathF.Sqrt(MathF.Max(
                1f,
                (float)item.OriginalRows / item.OriginalColumns));
            CudaOptimizerNative.NekoApply(
                deviceIndex,
                workspace.Data.NativePtr,
                update,
                length,
                learningRate,
                finalScale,
                weightDecay,
                item.ApplyWeightDecay);
            CudaBfp8GradientNative.Quantize(
                deviceIndex,
                workspace.Data,
                data,
                finiteStatus,
                stream);
        }
    }

    private static void DequantizeBfp8ToPointer(
        int deviceIndex,
        CudaBfp8BufferView source,
        nint destination,
        int length,
        nint stream)
    {
        if (source.Payload.Device.Index != deviceIndex
            || source.Scales.Device.Index != deviceIndex
            || source.Payload.Length != length)
        {
            throw new ArgumentException(
                "BFP8 source buffers must match the destination device " +
                "and logical length.",
                nameof(source));
        }
        NativeCudaRuntime.Check(
            CudaNativeGateway.Bfp8DequantizeFloat32(
                deviceIndex,
                source.Payload.NativePtr,
                source.Scales.NativePtr,
                destination,
                length,
                source.Descriptor.GetEffectiveBlockSize(length),
                stream),
            "CUDA BFP8 batched NekoMuon fast-moment dequantize");
    }
}
