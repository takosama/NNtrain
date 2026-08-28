
namespace NNtrain;

internal static partial class CudaOptimizerKernels
{
    private static NativeCudaBuffer<float> AllocateStateBuffer(
        NativeCudaDevice accelerator,
        ReadOnlySpan<float> host)
    {
        NativeCudaBuffer<float>? buffer = null;
        try
        {
            buffer = accelerator.Allocate1D<float>(host.Length);
            if (IsAllZero(host))
                buffer.MemSetToZero();
            else
                buffer.CopyFromCPU(host);
            NativeCudaBuffer<float> result = buffer;
            buffer = null;
            return result;
        }
        finally
        {
            buffer?.Dispose();
        }
    }

    private static NativeCudaBuffer<short> AllocateStateBuffer(
        NativeCudaDevice accelerator,
        ReadOnlySpan<short> host)
    {
        NativeCudaBuffer<short>? buffer = null;
        try
        {
            buffer = accelerator.Allocate1D<short>(host.Length);
            if (IsAllZero(host))
                buffer.MemSetToZero();
            else
                buffer.CopyFromCPU(host);
            NativeCudaBuffer<short> result = buffer;
            buffer = null;
            return result;
        }
        finally
        {
            buffer?.Dispose();
        }
    }

    private static bool IsAllZero(ReadOnlySpan<float> values)
    {
        foreach (float value in values)
        {
            if (value != 0f)
                return false;
        }
        return true;
    }

    private static bool IsAllZero(ReadOnlySpan<short> values)
    {
        foreach (short value in values)
        {
            if (value != 0)
                return false;
        }
        return true;
    }

    private static CudaBfp8BufferView AllocateBfp8StateBuffer(
        NativeCudaDevice accelerator,
        ReadOnlySpan<float> host)
    {
        NativeCudaBuffer<sbyte>? payload = null;
        NativeCudaBuffer<float>? scales = null;
        try
        {
            if (IsAllZero(host))
            {
                payload = accelerator.Allocate1D<sbyte>(host.Length);
                payload.MemSetToZero();
                scales = AllocateStateBuffer(accelerator, [1f]);
            }
            else
            {
                Bfp8EncodedStorage encoded =
                    Bfp8QuantizationCodec.Default.Encode(
                        host,
                        Bfp8QuantizationDescriptor.TensorWide);
                payload = accelerator.Allocate1D(encoded.Payload.Span);
                scales = accelerator.Allocate1D(encoded.Scales.Span);
            }

            var result = new CudaBfp8BufferView(
                payload,
                scales,
                Bfp8QuantizationDescriptor.TensorWide);
            payload = null;
            scales = null;
            return result;
        }
        finally
        {
            payload?.Dispose();
            scales?.Dispose();
        }
    }

    // cuBLAS launch/setup overhead dominates the tiny Gram matrices used by
    // vectors, norms, and narrow projections.  The direct kernels also fold
    // the polynomial combine into the final multiply.
    private const int DirectNewtonSchulzMaxRows = 32;

    internal const int DirectNewtonSchulzRowLimit =
        DirectNewtonSchulzMaxRows;

    private static TResult AllocateCudaResources<TResult>(
        Func<Action<IDisposable>, TResult> allocate)
    {
        // Four is the largest staged allocation below. Reserve the tracking
        // storage before the first cudaMalloc so recording ownership cannot
        // itself allocate and lose a just-created CUDA buffer.
        var resources = new List<IDisposable>(capacity: 16);
        try
        {
            return allocate(resources.Add);
        }
        catch
        {
            // Preserve the allocation failure while still attempting to free
            // every resource whose ownership was not transferred to TResult.
            for (int index = resources.Count - 1; index >= 0; index--)
            {
                try
                {
                    resources[index].Dispose();
                }
                catch
                {
                    // A cleanup error must not hide the original CUDA error.
                }
            }
            throw;
        }
    }

    internal static void PrewarmNekoMuon(IReadOnlyList<int> deviceIndices)
    {
        foreach (int deviceIndex in deviceIndices)
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex).Bind();
    }

    internal sealed class AdamWResidentState : IDisposable
    {
        private readonly float[] _firstHost;
        private readonly float[] _secondHost;
        private readonly Dictionary<int, Buffers> _buffers = [];

        internal AdamWResidentState(float[] firstHost, float[] secondHost)
        {
            _firstHost = firstHost;
            _secondHost = secondHost;
        }

        internal Buffers GetOrCreate(int deviceIndex)
        {
            if (_buffers.TryGetValue(deviceIndex, out Buffers? buffers))
                return buffers;
            NativeCudaDevice accelerator =
                ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
            buffers = AllocateCudaResources(own =>
            {
                NativeCudaBuffer<float> first =
                    AllocateStateBuffer(accelerator, _firstHost);
                own(first);
                NativeCudaBuffer<float> second =
                    AllocateStateBuffer(accelerator, _secondHost);
                own(second);
                var created = new Buffers(first, second);
                _buffers.Add(deviceIndex, created);
                return created;
            });
            return buffers;
        }

        internal void SynchronizeHost(int deviceIndex)
        {
            if (!_buffers.TryGetValue(deviceIndex, out Buffers? buffers))
                return;
            buffers.First.CopyToCPU(_firstHost);
            buffers.Second.CopyToCPU(_secondHost);
        }

        public void Dispose()
        {
            foreach (Buffers buffers in _buffers.Values)
                buffers.Dispose();
            _buffers.Clear();
        }

        internal sealed class Buffers(
            NativeCudaBuffer<float> first,
            NativeCudaBuffer<float> second) : IDisposable
        {
            internal NativeCudaBuffer<float> First { get; } = first;
            internal NativeCudaBuffer<float> Second { get; } = second;
            public void Dispose()
            {
                First.Dispose();
                Second.Dispose();
            }
        }
    }

    internal sealed class AdamWBFloat16ResidentState : IDisposable
    {
        private readonly short[] _firstHost;
        private readonly short[] _secondHost;
        private readonly Dictionary<int, Buffers> _buffers = [];

        internal AdamWBFloat16ResidentState(
            short[] firstHost,
            short[] secondHost)
        {
            _firstHost = firstHost;
            _secondHost = secondHost;
        }

        internal Buffers GetOrCreate(int deviceIndex)
        {
            if (_buffers.TryGetValue(deviceIndex, out Buffers? buffers))
                return buffers;
            NativeCudaDevice accelerator =
                ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
            buffers = AllocateCudaResources(own =>
            {
                NativeCudaBuffer<short> first =
                    AllocateStateBuffer(accelerator, _firstHost);
                own(first);
                NativeCudaBuffer<short> second =
                    AllocateStateBuffer(accelerator, _secondHost);
                own(second);
                var created = new Buffers(first, second);
                _buffers.Add(deviceIndex, created);
                return created;
            });
            return buffers;
        }

        internal void SynchronizeHost(int deviceIndex)
        {
            if (!_buffers.TryGetValue(deviceIndex, out Buffers? buffers))
                return;
            buffers.First.CopyToCPU(_firstHost);
            buffers.Second.CopyToCPU(_secondHost);
        }

        public void Dispose()
        {
            foreach (Buffers buffers in _buffers.Values)
                buffers.Dispose();
            _buffers.Clear();
        }

        internal sealed class Buffers(
            NativeCudaBuffer<short> first,
            NativeCudaBuffer<short> second) : IDisposable
        {
            internal NativeCudaBuffer<short> First { get; } = first;
            internal NativeCudaBuffer<short> Second { get; } = second;

            public void Dispose()
            {
                First.Dispose();
                Second.Dispose();
            }
        }
    }

    internal sealed class AdamWBfp8ResidentState : IDisposable
    {
        // These arrays are checkpoint shadows only. Once a device buffer is
        // created, the signed-int8 payload plus positive FP32 scale below is
        // the sole optimizer-state authority until explicit serialization.
        private readonly float[] _firstHost;
        private readonly float[] _secondHost;
        private readonly Dictionary<int, Buffers> _buffers = [];

        internal AdamWBfp8ResidentState(
            float[] firstHost,
            float[] secondHost)
        {
            _firstHost = firstHost;
            _secondHost = secondHost;
        }

        internal Buffers GetOrCreate(int deviceIndex)
        {
            if (_buffers.TryGetValue(deviceIndex, out Buffers? buffers))
                return buffers;
            NativeCudaDevice accelerator =
                ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
            buffers = AllocateCudaResources(own =>
            {
                CudaBfp8BufferView first = AllocateBfp8StateBuffer(
                    accelerator,
                    _firstHost);
                own(first.Payload);
                own(first.Scales);
                CudaBfp8BufferView second = AllocateBfp8StateBuffer(
                    accelerator,
                    _secondHost);
                own(second.Payload);
                own(second.Scales);
                var created = new Buffers(
                    first,
                    second);
                _buffers.Add(deviceIndex, created);
                return created;
            });
            return buffers;
        }

        internal void Execute(
            Tensor parameter,
            int deviceIndex,
            NativeCudaBuffer<int> finiteStatus,
            AdamWBfp8DeviceScratch scratch,
            float beta1,
            float beta2,
            float learningRate,
            float weightDecay,
            float updateScale,
            float scaledEpsilon,
            bool applyWeightDecay)
        {
            Buffers buffers = GetOrCreate(deviceIndex);
            CudaBfp8BufferView data =
                parameter.EnsureCudaBfp8Buffer(deviceIndex);
            CudaBfp8BufferView gradient =
                parameter.EnsureCudaBfp8GradientBuffer(deviceIndex);
            NativeCudaDevice accelerator =
                ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
            nint stream = accelerator.DefaultStream;
            AdamWBfp8DeviceScratch.Buffers workspace =
                scratch.Get(parameter.Numel);
            CudaBfp8Native.DequantizeFloat32(
                deviceIndex,
                data.Payload,
                data.Scales,
                workspace.Data,
                data.Descriptor,
                stream);
            CudaBfp8Native.DequantizeFloat32(
                deviceIndex,
                gradient.Payload,
                gradient.Scales,
                workspace.Gradient,
                gradient.Descriptor,
                stream);
            CudaBfp8Native.DequantizeFloat32(
                deviceIndex,
                buffers.First.Payload,
                buffers.First.Scales,
                workspace.First,
                buffers.First.Descriptor,
                stream);
            CudaBfp8Native.DequantizeFloat32(
                deviceIndex,
                buffers.Second.Payload,
                buffers.Second.Scales,
                workspace.Second,
                buffers.Second.Descriptor,
                stream);
            // Update moments first, then publish them to their authoritative
            // tensor-wide BFP8 representation.  The parameter update must
            // consume those quantized values, not transient precision that
            // disappears before the next optimizer step.
            CudaOptimizerNative.AdamWBfp8Moments(
                deviceIndex,
                workspace.Gradient.NativePtr,
                workspace.First.NativePtr,
                workspace.Second.NativePtr,
                parameter.Numel,
                beta1,
                beta2,
                finiteStatus.NativePtr);
            CudaBfp8GradientNative.Quantize(
                deviceIndex,
                workspace.First,
                buffers.First,
                finiteStatus,
                stream);
            CudaBfp8GradientNative.Quantize(
                deviceIndex,
                workspace.Second,
                buffers.Second,
                finiteStatus,
                stream);
            CudaOptimizerNative.AdamWBfp8Apply(
                deviceIndex,
                workspace.Data.NativePtr,
                workspace.First.NativePtr,
                workspace.Second.NativePtr,
                buffers.Second.Scales.NativePtr,
                parameter.Numel,
                learningRate,
                weightDecay,
                updateScale,
                scaledEpsilon,
                applyWeightDecay,
                finiteStatus.NativePtr);
            CudaBfp8GradientNative.Quantize(
                deviceIndex,
                workspace.Data,
                data,
                finiteStatus,
                stream);
        }

        internal void SynchronizeHost(int deviceIndex)
        {
            if (!_buffers.TryGetValue(deviceIndex, out Buffers? buffers))
                return;
            SynchronizeMoment(buffers.First, _firstHost);
            SynchronizeMoment(buffers.Second, _secondHost);
        }

        private static void SynchronizeMoment(
            CudaBfp8BufferView source,
            float[] destination)
        {
            var payload = new sbyte[destination.Length];
            var scale = new float[1];
            source.Payload.CopyToCPU(payload);
            source.Scales.CopyToCPU(scale);
            Bfp8QuantizationCodec.Default.Decode(
                payload,
                scale,
                Bfp8QuantizationDescriptor.TensorWide,
                destination);
        }

        internal CudaBfp8BufferView GetFirstMoment(int deviceIndex)
            => GetOrCreate(deviceIndex).First;

        internal CudaBfp8BufferView GetSecondMoment(int deviceIndex)
            => GetOrCreate(deviceIndex).Second;

        public void Dispose()
        {
            List<Exception>? failures = null;
            foreach (Buffers buffers in _buffers.Values)
            {
                try
                {
                    buffers.Dispose();
                }
                catch (Exception exception)
                {
                    (failures ??= []).Add(exception);
                }
            }
            _buffers.Clear();
            if (failures is not null)
            {
                throw new AggregateException(
                    "AdamW BFP8 state cleanup failed.", failures);
            }
        }

        internal sealed class Buffers(
            CudaBfp8BufferView first,
            CudaBfp8BufferView second) : IDisposable
        {
            internal CudaBfp8BufferView First { get; } = first;
            internal CudaBfp8BufferView Second { get; } = second;

            public void Dispose()
            {
                List<Exception>? failures = null;
                foreach (IDisposable resource in new IDisposable[]
                {
                    First.Payload,
                    First.Scales,
                    Second.Payload,
                    Second.Scales,
                })
                {
                    try
                    {
                        resource.Dispose();
                    }
                    catch (Exception exception)
                    {
                        (failures ??= []).Add(exception);
                    }
                }
                if (failures is not null)
                {
                    throw new AggregateException(
                        "AdamW BFP8 device buffers failed to dispose.",
                        failures);
                }
            }
        }
    }

    /// <summary>
    /// Four Float32 decode/update buffers shared by all AdamW leaves on one
    /// device. Capacity is the largest managed leaf, so native workspace
    /// usage is independent of parameter count.
    /// </summary>
    internal sealed class AdamWBfp8DeviceScratch : IDisposable
    {
        private readonly NativeCudaArena<float> _data;
        private readonly NativeCudaArena<float> _gradient;
        private readonly NativeCudaArena<float> _first;
        private readonly NativeCudaArena<float> _second;
        private readonly Dictionary<int, Buffers> _views = [];
        private int _disposed;

        internal AdamWBfp8DeviceScratch(int deviceIndex, int capacity)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(capacity);
            Capacity = capacity;
            NativeCudaDevice accelerator =
                ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
            _data = new NativeCudaArena<float>(accelerator, capacity);
            try
            {
                _gradient = new NativeCudaArena<float>(accelerator, capacity);
                try
                {
                    _first = new NativeCudaArena<float>(accelerator, capacity);
                    try
                    {
                        _second = new NativeCudaArena<float>(
                            accelerator, capacity);
                    }
                    catch
                    {
                        _first.Dispose();
                        throw;
                    }
                }
                catch
                {
                    _gradient.Dispose();
                    throw;
                }
            }
            catch
            {
                _data.Dispose();
                throw;
            }
        }

        internal int Capacity { get; }

        internal Buffers Get(int length)
        {
            ObjectDisposedException.ThrowIf(
                Volatile.Read(ref _disposed) != 0, this);
            if (length <= 0 || length > Capacity)
                throw new ArgumentOutOfRangeException(nameof(length));
            if (_views.TryGetValue(length, out Buffers? buffers))
                return buffers;
            buffers = new Buffers(
                _data.Slice(0, length),
                _gradient.Slice(0, length),
                _first.Slice(0, length),
                _second.Slice(0, length));
            _views.Add(length, buffers);
            return buffers;
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            List<Exception>? failures = null;
            foreach (Buffers buffers in _views.Values)
            {
                TryDispose(buffers, ref failures);
            }
            _views.Clear();
            TryDispose(_data, ref failures);
            TryDispose(_gradient, ref failures);
            TryDispose(_first, ref failures);
            TryDispose(_second, ref failures);
            if (failures is not null)
            {
                throw new AggregateException(
                    "AdamW BFP8 shared scratch cleanup failed.", failures);
            }
        }

        private static void TryDispose(
            IDisposable resource,
            ref List<Exception>? failures)
        {
            try
            {
                resource.Dispose();
            }
            catch (Exception exception)
            {
                (failures ??= []).Add(exception);
            }
        }

        internal sealed class Buffers(
            NativeCudaBuffer<float> data,
            NativeCudaBuffer<float> gradient,
            NativeCudaBuffer<float> first,
            NativeCudaBuffer<float> second) : IDisposable
        {
            internal NativeCudaBuffer<float> Data { get; } = data;
            internal NativeCudaBuffer<float> Gradient { get; } = gradient;
            internal NativeCudaBuffer<float> First { get; } = first;
            internal NativeCudaBuffer<float> Second { get; } = second;

            public void Dispose()
            {
                Data.Dispose();
                Gradient.Dispose();
                First.Dispose();
                Second.Dispose();
            }
        }
    }

    internal sealed record AdamWMultiTensorItem(
        Tensor Parameter,
        AdamWResidentState? FloatState,
        AdamWBFloat16ResidentState? BFloat16State,
        bool ApplyWeightDecay,
        bool PureBFloat16);

    internal sealed class AdamWMultiTensorPlan : IDisposable
    {
        private const int ElementsPerChunk = 4096;
        private readonly int _deviceIndex;
        private readonly PlanItemSignature[] _signatures;
        private readonly NativeCudaBuffer<
            CudaOptimizerNative.AdamWChunkDescriptor> _chunks;

        internal AdamWMultiTensorPlan(
            int deviceIndex,
            IReadOnlyList<AdamWMultiTensorItem> items)
        {
            _deviceIndex = deviceIndex;
            _signatures = new PlanItemSignature[items.Count];
            var descriptors = new List<
                CudaOptimizerNative.AdamWChunkDescriptor>();
            for (int itemIndex = 0; itemIndex < items.Count; itemIndex++)
            {
                AdamWMultiTensorItem item = items[itemIndex];
                Tensor parameter = item.Parameter;
                PlanItemSignature signature = CreateSignature(
                    deviceIndex,
                    item);
                _signatures[itemIndex] = signature;
                for (int offset = 0; offset < parameter.Numel;
                    offset += ElementsPerChunk)
                {
                    int length = Math.Min(
                        ElementsPerChunk, parameter.Numel - offset);
                    descriptors.Add(new CudaOptimizerNative
                        .AdamWChunkDescriptor(
                            signature.DataPointer,
                            signature.GradientPointer,
                            signature.FirstPointer,
                            signature.SecondPointer,
                            signature.ComputePointer,
                            offset,
                            length,
                            item.ApplyWeightDecay ? 1 : 0,
                            signature.PhysicalBFloat16 ? 1 : 0,
                            signature.BFloat16State ? 1 : 0,
                            signature.PureBFloat16 ? 1 : 0));
                }
            }
            if (descriptors.Count == 0)
            {
                throw new ArgumentException(
                    "AdamW multi-tensor plan requires CUDA-resident items.",
                    nameof(items));
            }
            _chunks = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex)
                .Allocate1D(System.Runtime.InteropServices.CollectionsMarshal
                    .AsSpan(descriptors));
        }

        internal void Execute(
            float beta1,
            float beta2,
            float learningRate,
            float weightDecay,
            float updateScale,
            float scaledEpsilon)
            => CudaOptimizerNative.AdamWMultiTensor(
                _deviceIndex,
                _chunks.NativePtr,
                _chunks.Length,
                beta1,
                beta2,
                learningRate,
                weightDecay,
                updateScale,
                scaledEpsilon);

        /// <summary>
        /// A plan contains raw native addresses, so object identity is part
        /// of its validity. Pointer equality alone is insufficient because
        /// the allocator may reuse an address after a dtype conversion or an
        /// arena rebind.
        /// </summary>
        internal bool Matches(IReadOnlyList<AdamWMultiTensorItem> items)
        {
            if (items.Count != _signatures.Length)
                return false;
            try
            {
                for (int index = 0; index < items.Count; index++)
                {
                    PlanItemSignature current = CreateSignature(
                        _deviceIndex,
                        items[index]);
                    if (!_signatures[index].Matches(current))
                        return false;
                }
                return true;
            }
            catch (ObjectDisposedException)
            {
                return false;
            }
        }

        private static PlanItemSignature CreateSignature(
            int deviceIndex,
            AdamWMultiTensorItem item)
        {
            Tensor parameter = item.Parameter;
            object data;
            object gradient;
            nint dataPointer;
            nint gradientPointer;
            if (item.PureBFloat16)
            {
                NativeCudaBuffer<ushort> dataBFloat16 =
                    parameter.EnsureCudaBFloat16Buffer(deviceIndex);
                if (!parameter.TryGetCudaBFloat16GradientBuffer(
                        deviceIndex,
                        out NativeCudaBuffer<ushort>? gradientBFloat16))
                {
                    throw new InvalidOperationException(
                        $"Pure BFloat16 AdamW requires a resident BF16 " +
                        $"gradient for parameter '{parameter.Name}' on " +
                        $"CUDA device {deviceIndex}.");
                }
                data = dataBFloat16;
                gradient = gradientBFloat16!;
                dataPointer = dataBFloat16.NativePtr;
                gradientPointer = gradientBFloat16!.NativePtr;
            }
            else
            {
                NativeCudaBuffer<float> dataFloat =
                    parameter.EnsureCudaMasterFloat32Buffer(deviceIndex);
                NativeCudaBuffer<float> gradientFloat =
                    parameter.EnsureCudaGradientBuffer(deviceIndex);
                data = dataFloat;
                gradient = gradientFloat;
                dataPointer = dataFloat.NativePtr;
                gradientPointer = gradientFloat.NativePtr;
            }
            object firstOwner;
            object secondOwner;
            nint firstPointer;
            nint secondPointer;
            bool bfloat16State;
            object stateOwner;
            if (item.FloatState is not null)
            {
                AdamWResidentState.Buffers state =
                    item.FloatState.GetOrCreate(deviceIndex);
                stateOwner = item.FloatState;
                firstOwner = state.First;
                secondOwner = state.Second;
                firstPointer = state.First.NativePtr;
                secondPointer = state.Second.NativePtr;
                bfloat16State = false;
            }
            else if (item.BFloat16State is not null)
            {
                AdamWBFloat16ResidentState.Buffers state =
                    item.BFloat16State.GetOrCreate(deviceIndex);
                stateOwner = item.BFloat16State;
                firstOwner = state.First;
                secondOwner = state.Second;
                firstPointer = state.First.NativePtr;
                secondPointer = state.Second.NativePtr;
                bfloat16State = true;
            }
            else
            {
                throw new ArgumentException(
                    "AdamW plan items require resident optimizer state.",
                    nameof(item));
            }

            (nint computePointer, bool physicalBFloat16, object? computeOwner) =
                item.PureBFloat16
                    ? (0, true, data)
                    : GetComputeDestinationWithOwner(parameter, deviceIndex);
            return new PlanItemSignature(
                parameter,
                parameter.DType,
                parameter.Bfp8Quantization,
                parameter.Shape,
                item.ApplyWeightDecay,
                bfloat16State,
                item.PureBFloat16,
                stateOwner,
                data,
                gradient,
                firstOwner,
                secondOwner,
                computeOwner,
                dataPointer,
                gradientPointer,
                firstPointer,
                secondPointer,
                computePointer,
                physicalBFloat16);
        }

        private readonly record struct PlanItemSignature(
            Tensor Parameter,
            TensorDType DType,
            Bfp8QuantizationDescriptor? Bfp8Descriptor,
            IReadOnlyList<int> Shape,
            bool ApplyWeightDecay,
            bool BFloat16State,
            bool PureBFloat16,
            object StateOwner,
            object Data,
            object Gradient,
            object FirstOwner,
            object SecondOwner,
            object? ComputeOwner,
            nint DataPointer,
            nint GradientPointer,
            nint FirstPointer,
            nint SecondPointer,
            nint ComputePointer,
            bool PhysicalBFloat16)
        {
            internal bool Matches(PlanItemSignature other)
                => ReferenceEquals(Parameter, other.Parameter)
                    && DType == other.DType
                    && Bfp8Descriptor == other.Bfp8Descriptor
                    && ShapesEqual(Shape, other.Shape)
                    && ApplyWeightDecay == other.ApplyWeightDecay
                    && BFloat16State == other.BFloat16State
                    && PureBFloat16 == other.PureBFloat16
                    && ReferenceEquals(StateOwner, other.StateOwner)
                    && ReferenceEquals(Data, other.Data)
                    && ReferenceEquals(Gradient, other.Gradient)
                    && ReferenceEquals(FirstOwner, other.FirstOwner)
                    && ReferenceEquals(SecondOwner, other.SecondOwner)
                    && ReferenceEquals(ComputeOwner, other.ComputeOwner)
                    && DataPointer == other.DataPointer
                    && GradientPointer == other.GradientPointer
                    && FirstPointer == other.FirstPointer
                    && SecondPointer == other.SecondPointer
                    && ComputePointer == other.ComputePointer
                    && PhysicalBFloat16 == other.PhysicalBFloat16;

            private static bool ShapesEqual(
                IReadOnlyList<int> left,
                IReadOnlyList<int> right)
            {
                if (left.Count != right.Count)
                    return false;
                for (int index = 0; index < left.Count; index++)
                {
                    if (left[index] != right[index])
                        return false;
                }
                return true;
            }
        }

        public void Dispose() => _chunks.Dispose();
    }

    /// <summary>
    /// Newton-Schulz work memory shared by every NekoMuon parameter on one
    /// CUDA device. Parameter updates are queued in-order on that device's
    /// default stream, so two parameters never use this storage concurrently.
    /// </summary>
    internal sealed class NekoMuonDeviceScratch : IDisposable
    {
        private readonly int _maximumLength;
        private NativeCudaArena<float>? _bfp8Data;
        private NativeCudaArena<float>? _bfp8Gradient;
        private NativeCudaArena<float>? _bfp8Fast;
        private NativeCudaArena<float>? _bfp8Slow;
        private readonly Dictionary<int, Bfp8Buffers> _bfp8Views = [];
        private int _disposed;

        internal NekoMuonDeviceScratch(
            int deviceIndex,
            int maximumLength,
            int maximumGramLength,
            int batchCapacity,
            bool useBFloat16TensorCores = true)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maximumLength);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(
                maximumGramLength);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batchCapacity);
            NativeCudaDevice accelerator =
                ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
            _maximumLength = maximumLength;
            BatchCapacity = batchCapacity;
            UseBFloat16TensorCores = useBFloat16TensorCores;
            int vectorCapacity = checked(maximumLength * batchCapacity);
            int gramCapacity = checked(maximumGramLength * batchCapacity);
            (X, Next, Gram, GramSquared) = AllocateCudaResources(own =>
            {
                NativeCudaBuffer<float> x =
                    accelerator.Allocate1D<float>(vectorCapacity);
                own(x);
                NativeCudaBuffer<float> next =
                    accelerator.Allocate1D<float>(vectorCapacity);
                own(next);
                NativeCudaBuffer<float> gram =
                    accelerator.Allocate1D<float>(gramCapacity);
                own(gram);
                NativeCudaBuffer<float> gramSquared =
                    accelerator.Allocate1D<float>(gramCapacity);
                own(gramSquared);
                return (x, next, gram, gramSquared);
            });
        }

        internal int BatchCapacity { get; }
        internal bool UseBFloat16TensorCores { get; }
        internal NativeCudaBuffer<float> X { get; }
        internal NativeCudaBuffer<float> Next { get; }
        internal NativeCudaBuffer<float> Gram { get; }
        internal NativeCudaBuffer<float> GramSquared { get; }

        internal Bfp8Buffers GetBfp8Buffers(int length)
        {
            ObjectDisposedException.ThrowIf(
                Volatile.Read(ref _disposed) != 0, this);
            if (length <= 0 || length > _maximumLength)
                throw new ArgumentOutOfRangeException(nameof(length));
            if (_bfp8Views.TryGetValue(length, out Bfp8Buffers? buffers))
                return buffers;
            EnsureBfp8Scratch();
            buffers = new Bfp8Buffers(
                _bfp8Data!.Slice(0, length),
                _bfp8Gradient!.Slice(0, length),
                _bfp8Fast!.Slice(0, length),
                _bfp8Slow!.Slice(0, length));
            _bfp8Views.Add(length, buffers);
            return buffers;
        }

        private void EnsureBfp8Scratch()
        {
            if (_bfp8Data is not null)
                return;
            NativeCudaDevice accelerator = X.Device;
            (_bfp8Data, _bfp8Gradient, _bfp8Fast, _bfp8Slow) =
                AllocateCudaResources(own =>
                {
                    var data = new NativeCudaArena<float>(
                        accelerator, _maximumLength);
                    own(data);
                    var gradient = new NativeCudaArena<float>(
                        accelerator, _maximumLength);
                    own(gradient);
                    var fast = new NativeCudaArena<float>(
                        accelerator, _maximumLength);
                    own(fast);
                    var slow = new NativeCudaArena<float>(
                        accelerator, _maximumLength);
                    own(slow);
                    return (data, gradient, fast, slow);
                });
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            List<Exception>? failures = null;
            foreach (Bfp8Buffers buffers in _bfp8Views.Values)
                TryDispose(buffers, ref failures);
            _bfp8Views.Clear();
            foreach (IDisposable? resource in new IDisposable?[]
            {
                X,
                Next,
                Gram,
                GramSquared,
                _bfp8Data,
                _bfp8Gradient,
                _bfp8Fast,
                _bfp8Slow,
            })
            {
                if (resource is not null)
                    TryDispose(resource, ref failures);
            }
            if (failures is not null)
            {
                throw new AggregateException(
                    "NekoMuon shared scratch cleanup failed.", failures);
            }
        }

        private static void TryDispose(
            IDisposable resource,
            ref List<Exception>? failures)
        {
            try
            {
                resource.Dispose();
            }
            catch (Exception exception)
            {
                (failures ??= []).Add(exception);
            }
        }

        internal sealed class Bfp8Buffers(
            NativeCudaBuffer<float> data,
            NativeCudaBuffer<float> gradient,
            NativeCudaBuffer<float> fast,
            NativeCudaBuffer<float> slow) : IDisposable
        {
            internal NativeCudaBuffer<float> Data { get; } = data;
            internal NativeCudaBuffer<float> Gradient { get; } = gradient;
            internal NativeCudaBuffer<float> Fast { get; } = fast;
            internal NativeCudaBuffer<float> Slow { get; } = slow;

            public void Dispose()
            {
                Data.Dispose();
                Gradient.Dispose();
                Fast.Dispose();
                Slow.Dispose();
            }
        }
    }

    internal sealed class NekoMuonResidentState : IDisposable
    {
        private readonly float[] _fastHost;
        private readonly float[] _slowHost;
        private readonly float _initialConfidence;
        private readonly Dictionary<int, NekoBuffers> _buffers = [];
        private int _deviceConfidenceAuthoritative;

        internal bool IsDeviceConfidenceAuthoritative
            => Volatile.Read(ref _deviceConfidenceAuthoritative) != 0;

        internal void MarkDeviceConfidenceAuthoritative()
            => Volatile.Write(ref _deviceConfidenceAuthoritative, 1);

        internal NekoMuonResidentState(
            float[] fastHost,
            float[] slowHost,
            float initialConfidence)
        {
            _fastHost = fastHost;
            _slowHost = slowHost;
            _initialConfidence = initialConfidence;
        }

        internal NekoBuffers GetOrCreate(int deviceIndex)
        {
            if (_buffers.TryGetValue(deviceIndex, out NekoBuffers? buffers))
                return buffers;
            NativeCudaDevice accelerator =
                ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
            var statsHost = new float[4];
            buffers = AllocateCudaResources(own =>
            {
                NativeCudaBuffer<float> fast =
                    AllocateStateBuffer(accelerator, _fastHost);
                own(fast);
                NativeCudaBuffer<float> slow =
                    AllocateStateBuffer(accelerator, _slowHost);
                own(slow);
                NativeCudaBuffer<float> stats =
                    accelerator.Allocate1D<float>(statsHost.Length);
                own(stats);
                NativeCudaBuffer<float> confidence =
                    AllocateStateBuffer(accelerator, [_initialConfidence]);
                own(confidence);
                var created = new NekoBuffers(
                    fast, slow, stats, confidence, statsHost);
                _buffers.Add(deviceIndex, created);
                return created;
            });
            return buffers;
        }

        internal void SynchronizeHost(int deviceIndex)
        {
            if (!_buffers.TryGetValue(deviceIndex, out NekoBuffers? buffers))
                return;
            buffers.Fast.CopyToCPU(_fastHost);
            buffers.Slow.CopyToCPU(_slowHost);
        }

        internal float SynchronizeConfidence(int deviceIndex)
        {
            if (!_buffers.TryGetValue(deviceIndex, out NekoBuffers? buffers))
                return _initialConfidence;
            Span<float> confidence = stackalloc float[1];
            buffers.Confidence.CopyToCPU(confidence);
            return confidence[0];
        }

        public void Dispose()
        {
            foreach (NekoBuffers buffers in _buffers.Values)
                buffers.Dispose();
            _buffers.Clear();
        }

        internal sealed class NekoBuffers(
            NativeCudaBuffer<float> fast,
            NativeCudaBuffer<float> slow,
            NativeCudaBuffer<float> stats,
            NativeCudaBuffer<float> confidence,
            float[] statsHost) : IDisposable
        {
            internal NativeCudaBuffer<float> Fast { get; } = fast;
            internal NativeCudaBuffer<float> Slow { get; } = slow;
            internal NativeCudaBuffer<float> Stats { get; } = stats;
            internal NativeCudaBuffer<float> Confidence { get; } = confidence;
            internal float[] StatsHost { get; } = statsHost;

            public void Dispose()
            {
                Fast.Dispose();
                Slow.Dispose();
                Stats.Dispose();
                Confidence.Dispose();
            }
        }
    }

    internal sealed class NekoMuonBfp8ResidentState : IDisposable
    {
        // Checkpoint shadows only. Device payloads/scales are authoritative
        // between explicit CaptureState/streaming checkpoint operations.
        private readonly float[] _fastHost;
        private readonly float[] _slowHost;
        private readonly float _initialConfidence;
        private readonly Dictionary<int, NekoBuffers> _buffers = [];

        internal NekoMuonBfp8ResidentState(
            float[] fastHost,
            float[] slowHost,
            float initialConfidence)
        {
            _fastHost = fastHost;
            _slowHost = slowHost;
            _initialConfidence = initialConfidence;
        }

        internal NekoBuffers GetOrCreate(int deviceIndex)
        {
            if (_buffers.TryGetValue(deviceIndex, out NekoBuffers? buffers))
                return buffers;
            NativeCudaDevice accelerator =
                ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
            var statsHost = new float[4];
            buffers = AllocateCudaResources(own =>
            {
                CudaBfp8BufferView fast = AllocateBfp8StateBuffer(
                    accelerator,
                    _fastHost);
                own(fast.Payload);
                own(fast.Scales);
                CudaBfp8BufferView slow = AllocateBfp8StateBuffer(
                    accelerator,
                    _slowHost);
                own(slow.Payload);
                own(slow.Scales);
                NativeCudaBuffer<float> stats =
                    accelerator.Allocate1D<float>(4);
                own(stats);
                NativeCudaBuffer<float> confidence =
                    AllocateStateBuffer(accelerator, [_initialConfidence]);
                own(confidence);
                var created = new NekoBuffers(
                    fast,
                    slow,
                    stats,
                    confidence,
                    statsHost);
                _buffers.Add(deviceIndex, created);
                return created;
            });
            return buffers;
        }

        internal void SynchronizeHost(int deviceIndex)
        {
            if (!_buffers.TryGetValue(deviceIndex, out NekoBuffers? buffers))
                return;
            SynchronizeMoment(buffers.Fast, _fastHost);
            SynchronizeMoment(buffers.Slow, _slowHost);
        }

        private static void SynchronizeMoment(
            CudaBfp8BufferView source,
            float[] destination)
        {
            var payload = new sbyte[destination.Length];
            var scale = new float[1];
            source.Payload.CopyToCPU(payload);
            source.Scales.CopyToCPU(scale);
            Bfp8QuantizationCodec.Default.Decode(
                payload,
                scale,
                Bfp8QuantizationDescriptor.TensorWide,
                destination);
        }

        internal CudaBfp8BufferView GetFast(int deviceIndex)
            => GetOrCreate(deviceIndex).Fast;

        internal CudaBfp8BufferView GetSlow(int deviceIndex)
            => GetOrCreate(deviceIndex).Slow;

        internal float SynchronizeConfidence(int deviceIndex)
        {
            if (!_buffers.TryGetValue(deviceIndex, out NekoBuffers? buffers))
                return _initialConfidence;
            var confidence = new float[1];
            buffers.Confidence.CopyToCPU(confidence);
            return confidence[0];
        }

        public void Dispose()
        {
            List<Exception>? failures = null;
            foreach (NekoBuffers buffers in _buffers.Values)
            {
                try
                {
                    buffers.Dispose();
                }
                catch (Exception exception)
                {
                    (failures ??= []).Add(exception);
                }
            }
            _buffers.Clear();
            if (failures is not null)
            {
                throw new AggregateException(
                    "NekoMuon BFP8 state cleanup failed.", failures);
            }
        }

        internal sealed class NekoBuffers(
            CudaBfp8BufferView fast,
            CudaBfp8BufferView slow,
            NativeCudaBuffer<float> stats,
            NativeCudaBuffer<float> confidence,
            float[] statsHost) : IDisposable
        {
            internal CudaBfp8BufferView Fast { get; } = fast;
            internal CudaBfp8BufferView Slow { get; } = slow;
            internal NativeCudaBuffer<float> Stats { get; } = stats;
            internal NativeCudaBuffer<float> Confidence { get; } = confidence;
            internal float[] StatsHost { get; } = statsHost;

            public void Dispose()
            {
                List<Exception>? failures = null;
                foreach (IDisposable resource in new IDisposable[]
                {
                    Fast.Payload,
                    Fast.Scales,
                    Slow.Payload,
                    Slow.Scales,
                    Stats,
                    Confidence,
                })
                {
                    try
                    {
                        resource.Dispose();
                    }
                    catch (Exception exception)
                    {
                        (failures ??= []).Add(exception);
                    }
                }
                if (failures is not null)
                {
                    throw new AggregateException(
                        "NekoMuon BFP8 device state cleanup failed.",
                        failures);
                }
            }
        }
    }

    /// <summary>
    /// Gathers every four-float NekoMuon statistic record on-device, then
    /// performs one D2H transfer per GPU instead of one per parameter.
    /// </summary>
    internal sealed class NekoMuonStatsBatch : IDisposable
    {
        private readonly int _deviceIndex;
        private readonly NekoMuonResidentState.NekoBuffers[] _states;
        private readonly NativeCudaBuffer<nint> _sourcePointers;
        private readonly NativeCudaBuffer<float> _packed;
        private readonly float[] _host;

        internal NekoMuonStatsBatch(
            int deviceIndex,
            IReadOnlyList<NekoMuonResidentState> states)
        {
            _deviceIndex = deviceIndex;
            NativeCudaDevice accelerator =
                ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
            _states = states
                .Select(state => state.GetOrCreate(deviceIndex))
                .ToArray();
            nint[] pointers = _states
                .Select(state => state.Stats.NativePtr)
                .ToArray();
            int packedLength = checked(states.Count * 4);
            _host = new float[packedLength];
            (_sourcePointers, _packed) = AllocateCudaResources(own =>
            {
                NativeCudaBuffer<nint> sourcePointers =
                    accelerator.Allocate1D(pointers);
                own(sourcePointers);
                NativeCudaBuffer<float> packed =
                    accelerator.Allocate1D<float>(packedLength);
                own(packed);
                return (sourcePointers, packed);
            });
        }

        internal void GatherAndRead()
        {
            CudaOptimizerNative.GatherStats(
                _deviceIndex,
                _sourcePointers.NativePtr,
                _packed.NativePtr,
                _states.Length);
            _packed.CopyToCPU(_host);
            for (int index = 0; index < _states.Length; ++index)
            {
                Array.Copy(
                    _host,
                    index * 4,
                    _states[index].StatsHost,
                    0,
                    4);
            }
        }

        public void Dispose()
        {
            _sourcePointers.Dispose();
            _packed.Dispose();
        }
    }

    internal sealed class NekoMuonBfp8StatsBatch : IDisposable
    {
        private readonly int _deviceIndex;
        private readonly NekoMuonBfp8ResidentState.NekoBuffers[] _states;
        private readonly NativeCudaBuffer<nint> _sourcePointers;
        private readonly NativeCudaBuffer<float> _packed;
        private readonly float[] _host;

        internal NekoMuonBfp8StatsBatch(
            int deviceIndex,
            IReadOnlyList<NekoMuonBfp8ResidentState> states)
        {
            _deviceIndex = deviceIndex;
            NativeCudaDevice accelerator =
                ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
            _states = states
                .Select(state => state.GetOrCreate(deviceIndex))
                .ToArray();
            nint[] pointers = _states
                .Select(state => state.Stats.NativePtr)
                .ToArray();
            int packedLength = checked(states.Count * 4);
            _host = new float[packedLength];
            (_sourcePointers, _packed) = AllocateCudaResources(own =>
            {
                NativeCudaBuffer<nint> sourcePointers =
                    accelerator.Allocate1D(pointers);
                own(sourcePointers);
                NativeCudaBuffer<float> packed =
                    accelerator.Allocate1D<float>(packedLength);
                own(packed);
                return (sourcePointers, packed);
            });
        }

        internal void GatherAndRead()
        {
            CudaOptimizerNative.GatherStats(
                _deviceIndex,
                _sourcePointers.NativePtr,
                _packed.NativePtr,
                _states.Length);
            _packed.CopyToCPU(_host);
            for (int index = 0; index < _states.Length; index++)
            {
                Array.Copy(
                    _host,
                    index * 4,
                    _states[index].StatsHost,
                    0,
                    4);
            }
        }

        public void Dispose()
        {
            List<Exception>? failures = null;
            try
            {
                _sourcePointers.Dispose();
            }
            catch (Exception exception)
            {
                (failures ??= []).Add(exception);
            }
            try
            {
                _packed.Dispose();
            }
            catch (Exception exception)
            {
                (failures ??= []).Add(exception);
            }
            if (failures is not null)
            {
                throw new AggregateException(
                    "NekoMuon BFP8 statistics batch cleanup failed.",
                    failures);
            }
        }
    }

    internal sealed record NekoMuonBatchItem(
        Tensor Parameter,
        NekoMuonResidentState State,
        int OriginalRows,
        int OriginalColumns,
        float PreviousConfidence,
        bool ApplyWeightDecay);

    private sealed record PreparedNekoMuonBatchItem(
        int Index,
        NekoMuonBatchItem Item,
        NekoMuonResidentState.NekoBuffers Buffers,
        float Confidence,
        float InverseNorm,
        int Rows,
        int Columns,
        int WholeSteps,
        float Fraction,
        bool UseBFloat16TensorCores);

    internal static void NekoMuonPrepareStatsResident(
        Tensor parameter,
        int deviceIndex,
        NekoMuonResidentState state,
        float betaFast,
        float betaSlow,
        float fastCorrection,
        float slowCorrection)
    {
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        var gradientBuffer = parameter.EnsureCudaGradientBuffer(deviceIndex);
        NekoMuonResidentState.NekoBuffers buffers =
            state.GetOrCreate(deviceIndex);
        buffers.Stats.MemSetToZero();
        if (!CudaNekoMuon.TryMomentsAndStatsCompact(
            accelerator,
            gradientBuffer,
            buffers.Fast,
            buffers.Slow,
            buffers.Stats,
            parameter.Numel,
            betaFast,
            betaSlow,
            fastCorrection,
            slowCorrection))
        {
            throw new InvalidOperationException(
                "The native CUDA NekoMuon statistics kernel is required.");
        }
    }

    /// <summary>
    /// Updates standard Float32/BF16 moments while leaving the statistics and
    /// confidence needed by fixed NS5 entirely device resident.
    /// </summary>
    internal static void NekoMuonPrepareFixedFiveStatsResident(
        Tensor parameter,
        int deviceIndex,
        NekoMuonResidentState state,
        NativeCudaBuffer<int> finiteStatus,
        float betaFast,
        float betaSlow,
        float fastCorrection,
        float slowCorrection,
        float epsilon,
        float rho)
    {
        NekoMuonPrepareStatsResident(
            parameter,
            deviceIndex,
            state,
            betaFast,
            betaSlow,
            fastCorrection,
            slowCorrection);
        NekoMuonResidentState.NekoBuffers buffers =
            state.GetOrCreate(deviceIndex);
        CudaOptimizerNative.NekoUpdateDeviceControl(
            deviceIndex,
            buffers.Stats.NativePtr,
            buffers.Confidence.NativePtr,
            finiteStatus.NativePtr,
            epsilon,
            rho);
        state.MarkDeviceConfidenceAuthoritative();
    }

    internal static void NekoMuonPrepareMix8StatsResident(
        Tensor parameter,
        int deviceIndex,
        NekoMuonResidentState state,
        NativeCudaBuffer<int> finiteStatus,
        float betaFast,
        float betaSlow,
        float fastCorrection,
        float slowCorrection,
        float epsilon,
        float rho)
    {
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        NativeCudaBuffer<float> gradient =
            parameter.EnsureCudaGradientBuffer(deviceIndex);
        NekoMuonResidentState.NekoBuffers buffers =
            state.GetOrCreate(deviceIndex);
        buffers.Stats.MemSetToZero();
        if (!CudaNekoMuon.TryMomentsAndStatsCompactFinite(
            accelerator,
            gradient,
            buffers.Fast,
            buffers.Slow,
            buffers.Stats,
            finiteStatus,
            parameter.Numel,
            betaFast,
            betaSlow,
            fastCorrection,
            slowCorrection))
        {
            throw new InvalidOperationException(
                "The ABI 1.7 finite-aware CUDA NekoMuon statistics " +
                "kernel is required for mix8_32.");
        }
        CudaOptimizerNative.NekoUpdateDeviceControl(
            deviceIndex,
            buffers.Stats.NativePtr,
            buffers.Confidence.NativePtr,
            finiteStatus.NativePtr,
            epsilon,
            rho);
        state.MarkDeviceConfidenceAuthoritative();
    }

    internal static void NekoMuonPrepareBfp8StatsResident(
        Tensor parameter,
        int deviceIndex,
        NekoMuonBfp8ResidentState state,
        NekoMuonDeviceScratch scratch,
        NativeCudaBuffer<int> finiteStatus,
        float betaFast,
        float betaSlow,
        float fastCorrection,
        float slowCorrection,
        float epsilon,
        float rho)
    {
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        NekoMuonDeviceScratch.Bfp8Buffers workspace =
            scratch.GetBfp8Buffers(parameter.Numel);
        CudaBfp8BufferView gradient =
            parameter.EnsureCudaBfp8GradientBuffer(deviceIndex);
        NekoMuonBfp8ResidentState.NekoBuffers buffers =
            state.GetOrCreate(deviceIndex);
        nint stream = accelerator.DefaultStream;
        CudaBfp8Native.DequantizeFloat32(
            deviceIndex,
            gradient.Payload,
            gradient.Scales,
            workspace.Gradient,
            gradient.Descriptor,
            stream);
        CudaBfp8Native.DequantizeFloat32(
            deviceIndex,
            buffers.Fast.Payload,
            buffers.Fast.Scales,
            workspace.Fast,
            buffers.Fast.Descriptor,
            stream);
        CudaBfp8Native.DequantizeFloat32(
            deviceIndex,
            buffers.Slow.Payload,
            buffers.Slow.Scales,
            workspace.Slow,
            buffers.Slow.Descriptor,
            stream);
        buffers.Stats.MemSetToZero();
        if (!CudaNekoMuon.TryMomentsAndStatsCompact(
            accelerator,
            workspace.Gradient,
            workspace.Fast,
            workspace.Slow,
            buffers.Stats,
            parameter.Numel,
            betaFast,
            betaSlow,
            fastCorrection,
            slowCorrection))
        {
            throw new InvalidOperationException(
                "The native CUDA NekoMuon statistics kernel is required.");
        }
        CudaBfp8GradientNative.Quantize(
            deviceIndex,
            workspace.Fast,
            buffers.Fast,
            finiteStatus,
            stream);
        CudaBfp8GradientNative.Quantize(
            deviceIndex,
            workspace.Slow,
            buffers.Slow,
            finiteStatus,
            stream);
        // Quantization is the persistence boundary for pure BFP8.  Rebuild
        // alignment/norm statistics from the dequantized authoritative
        // moments (the quantizer writes those values back into workspace)
        // instead of normalizing a quantized fast moment with statistics from
        // a transient FP32 value that no longer exists.
        buffers.Stats.MemSetToZero();
        if (!CudaNekoMuon.TryMomentsAndStatsCompactFinite(
            accelerator,
            workspace.Gradient,
            workspace.Fast,
            workspace.Slow,
            buffers.Stats,
            finiteStatus,
            parameter.Numel,
            betaFast: 1f,
            betaSlow: 1f,
            fastCorrection,
            slowCorrection))
        {
            throw new InvalidOperationException(
                "The finite-aware CUDA NekoMuon statistics kernel is " +
                "required after pure BFP8 moment publication.");
        }
        CudaOptimizerNative.NekoUpdateDeviceControl(
            deviceIndex,
            buffers.Stats.NativePtr,
            buffers.Confidence.NativePtr,
            finiteStatus.NativePtr,
            epsilon,
            rho);
    }

    internal static void NekoMuonReadStatsResident(
        int deviceIndex,
        NekoMuonResidentState state)
    {
        NekoMuonResidentState.NekoBuffers buffers =
            state.GetOrCreate(deviceIndex);
        buffers.Stats.CopyToCPU(buffers.StatsHost);
    }

    internal static float NekoMuonFinishStepResident(
        Tensor parameter,
        int deviceIndex,
        NekoMuonResidentState state,
        NekoMuonDeviceScratch scratch,
        int originalRows,
        int originalColumns,
        float betaFast,
        float betaSlow,
        float fastCorrection,
        float slowCorrection,
        float epsilon,
        float previousConfidence,
        float rho,
        int maxNewtonSchulzSteps,
        NekoMuonNewtonSchulzDepthMode newtonSchulzDepthMode,
        float configuredNewtonSchulzDepth,
        bool runNewtonSchulz,
        float coefficientA,
        float coefficientB,
        float coefficientC,
        float learningRate,
        float weightDecay,
        bool applyWeightDecay,
        bool forceFullNewtonSchulz = false,
        NativeCudaBuffer<int>? finiteStatus = null)
    {
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        var dataBuffer = parameter.EnsureCudaMasterFloat32Buffer(deviceIndex);
        NekoMuonResidentState.NekoBuffers buffers =
            state.GetOrCreate(deviceIndex);
        float[] stats = buffers.StatsHost;
        double alignmentDenominator =
            Math.Sqrt(stats[1]) * Math.Sqrt(stats[2]) + epsilon;
        double alignment = Math.Max(0d, stats[0] / alignmentDenominator);
        double persistence = stats[2] / (stats[2] + stats[3] + epsilon);
        float confidenceRaw =
            (float)Math.Clamp(alignment * persistence, 0d, 1d);
        float confidence = Math.Clamp(
            rho * previousConfidence + (1f - rho) * confidenceRaw,
            0f,
            1f);

        bool transpose = originalRows > originalColumns;
        int rows = Math.Min(originalRows, originalColumns);
        int columns = Math.Max(originalRows, originalColumns);
        float inverseNorm = 1f / ((float)Math.Sqrt(stats[1]) + epsilon);
        CudaOptimizerNative.NekoInitialize(
            deviceIndex,
            buffers.Fast.NativePtr,
            scratch.X.NativePtr,
            parameter.Numel,
            originalRows,
            originalColumns,
            transpose,
            1f / fastCorrection,
            inverseNorm);

        float depth = forceFullNewtonSchulz && runNewtonSchulz
            ? maxNewtonSchulzSteps
            : NekoMuon.ResolveNewtonSchulzDepth(
                maxNewtonSchulzSteps,
                newtonSchulzDepthMode,
                configuredNewtonSchulzDepth,
                confidence,
                runNewtonSchulz);
        bool useBFloat16TensorCores = parameter.DType is
                TensorDType.BFloat16 or TensorDType.Bfp8
            && scratch.UseBFloat16TensorCores;
        int wholeSteps = Math.Min(
            maxNewtonSchulzSteps,
            (int)MathF.Floor(depth));
        float fraction = depth - wholeSteps;
        NativeCudaBuffer<float> x = scratch.X;
        NativeCudaBuffer<float> next = scratch.Next;
        for (int step = 0; step < wholeSteps; step++)
        {
            NekoMuonNewtonSchulzResident(
                accelerator, deviceIndex, x, next,
                scratch.Gram, scratch.GramSquared,
                rows, columns, coefficientA, coefficientB, coefficientC,
                useBFloat16TensorCores);
            (x, next) = (next, x);
        }
        if (fraction > 0f)
        {
            NekoMuonNewtonSchulzResident(
                accelerator, deviceIndex, x, next,
                scratch.Gram, scratch.GramSquared,
                rows, columns, coefficientA, coefficientB, coefficientC,
                useBFloat16TensorCores);
            CudaOptimizerNative.NekoInterpolate(
                deviceIndex, x.NativePtr, next.NativePtr,
                parameter.Numel, fraction);
        }

        NativeCudaBuffer<float> update = x;
        if (transpose)
        {
            CudaOptimizerNative.NekoTransposeBack(
                deviceIndex,
                x.NativePtr,
                next.NativePtr,
                parameter.Numel,
                originalRows,
                originalColumns);
            update = next;
        }
        if (finiteStatus is not null)
        {
            CudaOptimizerNative.AccumulateFiniteStatus(
                deviceIndex,
                update.NativePtr,
                parameter.Numel,
                finiteStatus.NativePtr);
        }
        float finalScale = MathF.Sqrt(MathF.Max(
            1f,
            (float)originalRows / originalColumns));
        CudaOptimizerNative.NekoApply(
            deviceIndex,
            dataBuffer.NativePtr,
            update.NativePtr,
            parameter.Numel,
            learningRate,
            finalScale,
            weightDecay,
            applyWeightDecay);
        PublishMaster(parameter, accelerator, deviceIndex, dataBuffer);
        return confidence;
    }

    /// <summary>
    /// Fixed NS5 mixed-precision update whose statistics, confidence, and
    /// finite checks remain device-resident. The caller reads only the shared
    /// one-int status after every parameter on the device has been queued.
    /// </summary>
    internal static void NekoMuonFinishFixedFiveDeviceResident(
        Tensor parameter,
        int deviceIndex,
        NekoMuonResidentState state,
        NekoMuonDeviceScratch scratch,
        NativeCudaBuffer<int> finiteStatus,
        int originalRows,
        int originalColumns,
        float fastCorrection,
        float epsilon,
        float coefficientA,
        float coefficientB,
        float coefficientC,
        float learningRate,
        float weightDecay,
        bool applyWeightDecay,
        bool publishMix8)
    {
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        NativeCudaBuffer<float> data =
            parameter.EnsureCudaMasterFloat32Buffer(deviceIndex);
        NekoMuonResidentState.NekoBuffers buffers =
            state.GetOrCreate(deviceIndex);
        bool transpose = originalRows > originalColumns;
        int rows = Math.Min(originalRows, originalColumns);
        int columns = Math.Max(originalRows, originalColumns);
        NekoMuonFixedNs5Telemetry.RecordScalar(rows);
        CudaOptimizerNative.NekoInitializeFromDeviceStats(
            deviceIndex,
            buffers.Fast.NativePtr,
            scratch.X.NativePtr,
            parameter.Numel,
            originalRows,
            originalColumns,
            transpose,
            1f / fastCorrection,
            buffers.Stats.NativePtr,
            epsilon,
            finiteStatus.NativePtr);

        bool useBFloat16TensorCores = scratch.UseBFloat16TensorCores;
        NativeCudaBuffer<float> x = scratch.X;
        NativeCudaBuffer<float> next = scratch.Next;
        for (int step = 0; step < 5; step++)
        {
            NekoMuonNewtonSchulzResident(
                accelerator,
                deviceIndex,
                x,
                next,
                scratch.Gram,
                scratch.GramSquared,
                rows,
                columns,
                coefficientA,
                coefficientB,
                coefficientC,
                useBFloat16TensorCores);
            (x, next) = (next, x);
        }

        NativeCudaBuffer<float> update = x;
        if (transpose)
        {
            CudaOptimizerNative.NekoTransposeBack(
                deviceIndex,
                x.NativePtr,
                next.NativePtr,
                parameter.Numel,
                originalRows,
                originalColumns);
            update = next;
        }
        CudaOptimizerNative.AccumulateFiniteStatus(
            deviceIndex,
            update.NativePtr,
            parameter.Numel,
            finiteStatus.NativePtr);
        float finalScale = MathF.Sqrt(MathF.Max(
            1f,
            (float)originalRows / originalColumns));
        CudaOptimizerNative.NekoApply(
            deviceIndex,
            data.NativePtr,
            update.NativePtr,
            parameter.Numel,
            learningRate,
            finalScale,
            weightDecay,
            applyWeightDecay);
        if (publishMix8)
            PublishMix8Master(parameter, deviceIndex, finiteStatus);
        else
            PublishMaster(parameter, accelerator, deviceIndex, data);
    }

    /// <summary>
    /// Executes fixed NS5 while leaving statistics and confidence on device,
    /// grouping equal normalized matrix shapes into strided-batched cuBLAS
    /// calls. A group of N matrices therefore issues fifteen GEMMs rather
    /// than fifteen GEMMs per matrix. Small matrices retain the fused direct
    /// kernels, which are faster than cuBLAS at that size.
    /// </summary>
    internal static void NekoMuonFinishFixedFiveGroupedDeviceResident(
        int deviceIndex,
        IReadOnlyList<NekoMuonBatchItem> items,
        NekoMuonDeviceScratch scratch,
        NativeCudaBuffer<int> finiteStatus,
        float fastCorrection,
        float epsilon,
        float coefficientA,
        float coefficientB,
        float coefficientC,
        float learningRate,
        float weightDecay,
        bool publishMix8)
    {
        ArgumentNullException.ThrowIfNull(items);
        if (items.Count == 0)
            return;

        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        bool useBFloat16TensorCores = scratch.UseBFloat16TensorCores;
        foreach (IGrouping<(int Rows, int Columns), NekoMuonBatchItem> group
            in items.GroupBy(item => (
                Math.Min(item.OriginalRows, item.OriginalColumns),
                Math.Max(item.OriginalRows, item.OriginalColumns))))
        {
            NekoMuonBatchItem[] grouped = group.ToArray();
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
                        NekoMuonBatchItem item = grouped[offset + slot];
                        NekoMuonFinishFixedFiveDeviceResident(
                            item.Parameter,
                            deviceIndex,
                            item.State,
                            scratch,
                            finiteStatus,
                            item.OriginalRows,
                            item.OriginalColumns,
                            fastCorrection,
                            epsilon,
                            coefficientA,
                            coefficientB,
                            coefficientC,
                            learningRate,
                            weightDecay,
                            item.ApplyWeightDecay,
                            publishMix8);
                    }
                    continue;
                }

                FinishNekoMuonFixedFiveBatch(
                    accelerator,
                    deviceIndex,
                    grouped.AsSpan(offset, count),
                    scratch,
                    finiteStatus,
                    1f / fastCorrection,
                    epsilon,
                    coefficientA,
                    coefficientB,
                    coefficientC,
                    learningRate,
                    weightDecay,
                    useBFloat16TensorCores,
                    publishMix8);
            }
        }
    }

    private static void FinishNekoMuonFixedFiveBatch(
        NativeCudaDevice accelerator,
        int deviceIndex,
        ReadOnlySpan<NekoMuonBatchItem> items,
        NekoMuonDeviceScratch scratch,
        NativeCudaBuffer<int> finiteStatus,
        float inverseFastCorrection,
        float epsilon,
        float coefficientA,
        float coefficientB,
        float coefficientC,
        float learningRate,
        float weightDecay,
        bool useBFloat16TensorCores,
        bool publishMix8)
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
            NekoMuonBatchItem item = items[slot];
            NekoMuonResidentState.NekoBuffers buffers =
                item.State.GetOrCreate(deviceIndex);
            CudaOptimizerNative.NekoInitializeFromDeviceStats(
                deviceIndex,
                buffers.Fast.NativePtr,
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
                useBFloat16TensorCores);
            (x, next) = (next, x);
        }

        CudaOptimizerNative.AccumulateFiniteStatus(
            deviceIndex,
            x,
            checked(length * count),
            finiteStatus.NativePtr);
        for (int slot = 0; slot < count; slot++)
        {
            NekoMuonBatchItem item = items[slot];
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
            NativeCudaBuffer<float> data =
                item.Parameter.EnsureCudaMasterFloat32Buffer(deviceIndex);
            float finalScale = MathF.Sqrt(MathF.Max(
                1f,
                (float)item.OriginalRows / item.OriginalColumns));
            CudaOptimizerNative.NekoApply(
                deviceIndex,
                data.NativePtr,
                update,
                length,
                learningRate,
                finalScale,
                weightDecay,
                item.ApplyWeightDecay);
            if (publishMix8)
                PublishMix8Master(item.Parameter, deviceIndex, finiteStatus);
            else
                PublishMaster(item.Parameter, accelerator, deviceIndex, data);
        }
    }

    internal static float NekoMuonFinishBfp8StepResident(
        Tensor parameter,
        int deviceIndex,
        NekoMuonBfp8ResidentState state,
        NekoMuonDeviceScratch scratch,
        NativeCudaBuffer<int> finiteStatus,
        int originalRows,
        int originalColumns,
        float fastCorrection,
        float epsilon,
        float previousConfidence,
        float rho,
        int maxNewtonSchulzSteps,
        NekoMuonNewtonSchulzDepthMode newtonSchulzDepthMode,
        float configuredNewtonSchulzDepth,
        bool runNewtonSchulz,
        float coefficientA,
        float coefficientB,
        float coefficientC,
        float learningRate,
        float weightDecay,
        bool applyWeightDecay,
        bool deviceOnlyFixedFive,
        bool forceFullNewtonSchulz = false)
    {
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        NekoMuonDeviceScratch.Bfp8Buffers workspace =
            scratch.GetBfp8Buffers(parameter.Numel);
        NekoMuonBfp8ResidentState.NekoBuffers buffers =
            state.GetOrCreate(deviceIndex);
        CudaBfp8BufferView data = parameter.EnsureCudaBfp8Buffer(deviceIndex);
        nint stream = accelerator.DefaultStream;
        CudaBfp8Native.DequantizeFloat32(
            deviceIndex,
            data.Payload,
            data.Scales,
            workspace.Data,
            data.Descriptor,
            stream);
        CudaBfp8Native.DequantizeFloat32(
            deviceIndex,
            buffers.Fast.Payload,
            buffers.Fast.Scales,
            workspace.Fast,
            buffers.Fast.Descriptor,
            stream);

        float[] stats = buffers.StatsHost;
        float confidence = deviceOnlyFixedFive
            ? previousConfidence
            : CalculateNekoMuonConfidence(
                stats, epsilon, previousConfidence, rho);
        bool transpose = originalRows > originalColumns;
        int rows = Math.Min(originalRows, originalColumns);
        int columns = Math.Max(originalRows, originalColumns);
        if (deviceOnlyFixedFive)
        {
            CudaOptimizerNative.NekoInitializeFromDeviceStats(
                deviceIndex,
                workspace.Fast.NativePtr,
                scratch.X.NativePtr,
                parameter.Numel,
                originalRows,
                originalColumns,
                transpose,
                1f / fastCorrection,
                buffers.Stats.NativePtr,
                epsilon,
                finiteStatus.NativePtr);
        }
        else
        {
            float inverseNorm = 1f
                / ((float)Math.Sqrt(stats[1]) + epsilon);
            CudaOptimizerNative.NekoInitialize(
                deviceIndex,
                workspace.Fast.NativePtr,
                scratch.X.NativePtr,
                parameter.Numel,
                originalRows,
                originalColumns,
                transpose,
                1f / fastCorrection,
                inverseNorm);
        }

        float depth = deviceOnlyFixedFive
            ? 5f
            : forceFullNewtonSchulz && runNewtonSchulz
            ? maxNewtonSchulzSteps
            : NekoMuon.ResolveNewtonSchulzDepth(
                maxNewtonSchulzSteps,
                newtonSchulzDepthMode,
                configuredNewtonSchulzDepth,
                confidence,
                runNewtonSchulz);
        int wholeSteps = Math.Min(
            maxNewtonSchulzSteps,
            (int)MathF.Floor(depth));
        float fraction = depth - wholeSteps;
        bool useBFloat16TensorCores = scratch.UseBFloat16TensorCores;
        NativeCudaBuffer<float> x = scratch.X;
        NativeCudaBuffer<float> next = scratch.Next;
        for (int step = 0; step < wholeSteps; step++)
        {
            NekoMuonNewtonSchulzResident(
                accelerator,
                deviceIndex,
                x,
                next,
                scratch.Gram,
                scratch.GramSquared,
                rows,
                columns,
                coefficientA,
                coefficientB,
                coefficientC,
                useBFloat16TensorCores);
            (x, next) = (next, x);
        }
        if (fraction > 0f)
        {
            NekoMuonNewtonSchulzResident(
                accelerator,
                deviceIndex,
                x,
                next,
                scratch.Gram,
                scratch.GramSquared,
                rows,
                columns,
                coefficientA,
                coefficientB,
                coefficientC,
                useBFloat16TensorCores);
            CudaOptimizerNative.NekoInterpolate(
                deviceIndex,
                x.NativePtr,
                next.NativePtr,
                parameter.Numel,
                fraction);
        }

        NativeCudaBuffer<float> update = x;
        if (transpose)
        {
            CudaOptimizerNative.NekoTransposeBack(
                deviceIndex,
                x.NativePtr,
                next.NativePtr,
                parameter.Numel,
                originalRows,
                originalColumns);
            update = next;
        }
        float finalScale = MathF.Sqrt(MathF.Max(
            1f,
            (float)originalRows / originalColumns));
        CudaOptimizerNative.NekoApply(
            deviceIndex,
            workspace.Data.NativePtr,
            update.NativePtr,
            parameter.Numel,
            learningRate,
            finalScale,
            weightDecay,
            applyWeightDecay);
        CudaBfp8GradientNative.Quantize(
            deviceIndex,
            workspace.Data,
            data,
            finiteStatus,
            stream);
        return confidence;
    }

    internal static float[] NekoMuonFinishStepGrouped(
        int deviceIndex,
        IReadOnlyList<NekoMuonBatchItem> items,
        NekoMuonDeviceScratch scratch,
        float fastCorrection,
        float slowCorrection,
        float epsilon,
        float rho,
        int maxNewtonSchulzSteps,
        NekoMuonNewtonSchulzDepthMode newtonSchulzDepthMode,
        float configuredNewtonSchulzDepth,
        bool runNewtonSchulz,
        float coefficientA,
        float coefficientB,
        float coefficientC,
        float learningRate,
        float weightDecay,
        bool forceFullNewtonSchulz = false,
        NativeCudaBuffer<int>? finiteStatus = null)
    {
        var confidences = new float[items.Count];
        if (scratch.BatchCapacity <= 1 || items.Count <= 1)
        {
            for (int index = 0; index < items.Count; index++)
            {
                NekoMuonBatchItem item = items[index];
                confidences[index] = NekoMuonFinishStepResident(
                    item.Parameter,
                    deviceIndex,
                    item.State,
                    scratch,
                    item.OriginalRows,
                    item.OriginalColumns,
                    betaFast: 0f,
                    betaSlow: 0f,
                    fastCorrection,
                    slowCorrection,
                    epsilon,
                    item.PreviousConfidence,
                    rho,
                    maxNewtonSchulzSteps,
                    newtonSchulzDepthMode,
                    configuredNewtonSchulzDepth,
                    runNewtonSchulz,
                    coefficientA,
                    coefficientB,
                    coefficientC,
                    learningRate,
                    weightDecay,
                    item.ApplyWeightDecay,
                    forceFullNewtonSchulz,
                    finiteStatus);
            }
            return confidences;
        }

        var prepared = new PreparedNekoMuonBatchItem[items.Count];
        for (int index = 0; index < items.Count; index++)
        {
            NekoMuonBatchItem item = items[index];
            NekoMuonResidentState.NekoBuffers buffers =
                item.State.GetOrCreate(deviceIndex);
            float[] stats = buffers.StatsHost;
            float confidence = CalculateNekoMuonConfidence(
                stats, epsilon, item.PreviousConfidence, rho);
            float depth = forceFullNewtonSchulz && runNewtonSchulz
                ? maxNewtonSchulzSteps
                : NekoMuon.ResolveNewtonSchulzDepth(
                    maxNewtonSchulzSteps,
                    newtonSchulzDepthMode,
                    configuredNewtonSchulzDepth,
                    confidence,
                    runNewtonSchulz);
            int wholeSteps = Math.Min(
                maxNewtonSchulzSteps,
                (int)MathF.Floor(depth));
            int rows = Math.Min(item.OriginalRows, item.OriginalColumns);
            int columns = Math.Max(item.OriginalRows, item.OriginalColumns);
            bool useBFloat16TensorCores =
                item.Parameter.DType is
                    TensorDType.BFloat16 or TensorDType.Bfp8
                && scratch.UseBFloat16TensorCores;
            prepared[index] = new PreparedNekoMuonBatchItem(
                index,
                item,
                buffers,
                confidence,
                1f / ((float)Math.Sqrt(stats[1]) + epsilon),
                rows,
                columns,
                wholeSteps,
                depth - wholeSteps,
                useBFloat16TensorCores);
            confidences[index] = confidence;
        }

        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        foreach (IGrouping<
            (int Rows, int Columns, int WholeSteps, bool HasFraction,
                bool UseBFloat16TensorCores),
            PreparedNekoMuonBatchItem> group in prepared.GroupBy(item => (
                item.Rows,
                item.Columns,
                item.WholeSteps,
                item.Fraction > 0f,
                item.UseBFloat16TensorCores)))
        {
            PreparedNekoMuonBatchItem[] grouped = group.ToArray();
            for (int offset = 0; offset < grouped.Length;
                offset += scratch.BatchCapacity)
            {
                int count = Math.Min(
                    scratch.BatchCapacity,
                    grouped.Length - offset);
                if (count == 1 || group.Key.Rows <= DirectNewtonSchulzMaxRows)
                {
                    for (int slot = 0; slot < count; slot++)
                    {
                        PreparedNekoMuonBatchItem preparedItem =
                            grouped[offset + slot];
                        NekoMuonBatchItem item = preparedItem.Item;
                        confidences[preparedItem.Index] =
                            NekoMuonFinishStepResident(
                                item.Parameter,
                                deviceIndex,
                                item.State,
                                scratch,
                                item.OriginalRows,
                                item.OriginalColumns,
                                betaFast: 0f,
                                betaSlow: 0f,
                                fastCorrection,
                                slowCorrection,
                                epsilon,
                                item.PreviousConfidence,
                                rho,
                                maxNewtonSchulzSteps,
                                newtonSchulzDepthMode,
                                configuredNewtonSchulzDepth,
                                runNewtonSchulz,
                                coefficientA,
                                coefficientB,
                                coefficientC,
                                learningRate,
                                weightDecay,
                                item.ApplyWeightDecay,
                                forceFullNewtonSchulz,
                                finiteStatus);
                    }
                    continue;
                }

                FinishNekoMuonBatch(
                    accelerator,
                    deviceIndex,
                    grouped.AsSpan(offset, count),
                    scratch,
                    1f / fastCorrection,
                    coefficientA,
                    coefficientB,
                    coefficientC,
                    learningRate,
                    weightDecay,
                    finiteStatus);
            }
        }
        return confidences;
    }

    private static float CalculateNekoMuonConfidence(
        float[] stats,
        float epsilon,
        float previousConfidence,
        float rho)
    {
        double alignmentDenominator =
            Math.Sqrt(stats[1]) * Math.Sqrt(stats[2]) + epsilon;
        double alignment = Math.Max(0d, stats[0] / alignmentDenominator);
        double persistence = stats[2] / (stats[2] + stats[3] + epsilon);
        float confidenceRaw =
            (float)Math.Clamp(alignment * persistence, 0d, 1d);
        return Math.Clamp(
            rho * previousConfidence + (1f - rho) * confidenceRaw,
            0f,
            1f);
    }

    private static void FinishNekoMuonBatch(
        NativeCudaDevice accelerator,
        int deviceIndex,
        ReadOnlySpan<PreparedNekoMuonBatchItem> items,
        NekoMuonDeviceScratch scratch,
        float inverseFastCorrection,
        float coefficientA,
        float coefficientB,
        float coefficientC,
        float learningRate,
        float weightDecay,
        NativeCudaBuffer<int>? finiteStatus)
    {
        int count = items.Length;
        int rows = items[0].Rows;
        int columns = items[0].Columns;
        int length = checked(rows * columns);
        int gramLength = checked(rows * rows);
        for (int slot = 0; slot < count; slot++)
        {
            PreparedNekoMuonBatchItem prepared = items[slot];
            NekoMuonBatchItem item = prepared.Item;
            CudaOptimizerNative.NekoInitialize(
                deviceIndex,
                prepared.Buffers.Fast.NativePtr,
                AddFloatOffset(scratch.X.NativePtr, slot * length),
                length,
                item.OriginalRows,
                item.OriginalColumns,
                item.OriginalRows > item.OriginalColumns,
                inverseFastCorrection,
                prepared.InverseNorm);
        }

        nint x = scratch.X.NativePtr;
        nint next = scratch.Next.NativePtr;
        for (int step = 0; step < items[0].WholeSteps; step++)
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
                items[0].UseBFloat16TensorCores);
            (x, next) = (next, x);
        }
        if (items[0].Fraction > 0f)
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
                items[0].UseBFloat16TensorCores);
            for (int slot = 0; slot < count; slot++)
            {
                CudaOptimizerNative.NekoInterpolate(
                    deviceIndex,
                    AddFloatOffset(x, slot * length),
                    AddFloatOffset(next, slot * length),
                    length,
                    items[slot].Fraction);
            }
        }

        if (finiteStatus is not null)
        {
            CudaOptimizerNative.AccumulateFiniteStatus(
                deviceIndex,
                x,
                checked(length * count),
                finiteStatus.NativePtr);
        }
        for (int slot = 0; slot < count; slot++)
        {
            PreparedNekoMuonBatchItem prepared = items[slot];
            NekoMuonBatchItem item = prepared.Item;
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
            NativeCudaBuffer<float> data =
                item.Parameter.EnsureCudaMasterFloat32Buffer(deviceIndex);
            float finalScale = MathF.Sqrt(MathF.Max(
                1f,
                (float)item.OriginalRows / item.OriginalColumns));
            CudaOptimizerNative.NekoApply(
                deviceIndex,
                data.NativePtr,
                update,
                length,
                learningRate,
                finalScale,
                weightDecay,
                item.ApplyWeightDecay);
            PublishMaster(item.Parameter, accelerator, deviceIndex, data);
        }
    }

    private static void NekoMuonNewtonSchulzBatched(
        NativeCudaDevice accelerator,
        int deviceIndex,
        nint source,
        nint destination,
        nint gram,
        nint gramSquared,
        int rows,
        int columns,
        int batch,
        float coefficientA,
        float coefficientB,
        float coefficientC,
        bool useBFloat16TensorCores)
    {
        CudaBlas.MuonGramBatched(
            accelerator,
            deviceIndex,
            source,
            gram,
            rows,
            columns,
            batch,
            useBFloat16TensorCores);
        CudaBlas.MuonGramBatched(
            accelerator,
            deviceIndex,
            gram,
            gramSquared,
            rows,
            rows,
            batch,
            useBFloat16TensorCores);
        CudaOptimizerNative.NekoCombineBatched(
            deviceIndex,
            gram,
            gramSquared,
            checked(rows * rows),
            batch,
            rows,
            coefficientA,
            coefficientB,
            coefficientC);
        CudaBlas.MuonPolynomialUpdateBatched(
            accelerator,
            deviceIndex,
            source,
            gramSquared,
            destination,
            rows,
            columns,
            batch,
            useBFloat16TensorCores);
    }

    private static nint AddFloatOffset(nint pointer, int elementOffset)
        => pointer + checked(elementOffset * sizeof(float));

    private static void PublishMaster(
        Tensor parameter,
        NativeCudaDevice accelerator,
        int deviceIndex,
        NativeCudaBuffer<float> master)
    {
        if (parameter.DType == TensorDType.Float32)
            return;
        if (parameter.DType == TensorDType.BFloat16)
        {
            var bfloat16Compute = parameter.EnsureCudaBFloat16Buffer(deviceIndex);
            CudaOptimizerNative.PublishBFloat16(
                deviceIndex, master.NativePtr, bfloat16Compute.NativePtr,
                parameter.Numel, physical: true);
            return;
        }
        if (parameter.DType == TensorDType.Bfp8)
        {
            // The mix8_32 caller publishes all block-scaled parameters with
            // one shared finite-status scalar after the grouped update.
            return;
        }
        var compute = parameter.EnsureCudaFloat32Buffer(deviceIndex);
        CudaOptimizerNative.PublishBFloat16(
            deviceIndex, master.NativePtr, compute.NativePtr,
            parameter.Numel, physical: false);
    }

    private static void NekoMuonNewtonSchulzResident(
        NativeCudaDevice accelerator,
        int deviceIndex,
        NativeCudaBuffer<float> source,
        NativeCudaBuffer<float> destination,
        NativeCudaBuffer<float> gram,
        NativeCudaBuffer<float> gramSquared,
        int rows,
        int columns,
        float coefficientA,
        float coefficientB,
        float coefficientC,
        bool useBFloat16TensorCores)
    {
        if (rows <= DirectNewtonSchulzMaxRows)
        {
            if (useBFloat16TensorCores)
            {
                CudaOptimizerNative.SymmetricGramBFloat16Operands(
                    deviceIndex, source.NativePtr, gram.NativePtr,
                    rows, columns);
                CudaOptimizerNative.SymmetricGramBFloat16Operands(
                    deviceIndex, gram.NativePtr, gramSquared.NativePtr,
                    rows, rows);
                CudaOptimizerNative.NewtonSchulzBFloat16Operands(
                    deviceIndex, source.NativePtr, gram.NativePtr,
                    gramSquared.NativePtr, destination.NativePtr,
                    rows, columns, coefficientA, coefficientB, coefficientC);
            }
            else
            {
                CudaOptimizerNative.SymmetricGram(
                    deviceIndex, source.NativePtr, gram.NativePtr,
                    rows, columns);
                CudaOptimizerNative.SymmetricGram(
                    deviceIndex, gram.NativePtr, gramSquared.NativePtr,
                    rows, rows);
                CudaOptimizerNative.NewtonSchulz(
                    deviceIndex, source.NativePtr, gram.NativePtr,
                    gramSquared.NativePtr, destination.NativePtr,
                    rows, columns, coefficientA, coefficientB, coefficientC);
            }
            return;
        }

        CudaBlas.MuonGram(
            accelerator, deviceIndex, source, gram, rows, columns,
            useBFloat16TensorCores);
        CudaBlas.MuonGram(
            accelerator, deviceIndex, gram, gramSquared, rows, rows,
            useBFloat16TensorCores);
        CudaOptimizerNative.NekoCombine(
            deviceIndex,
            gram.NativePtr,
            gramSquared.NativePtr,
            checked(rows * rows),
            rows,
            coefficientA,
            coefficientB,
            coefficientC);
        CudaBlas.MuonPolynomialUpdate(
            accelerator,
            deviceIndex,
            source,
            gramSquared,
            destination,
            rows,
            columns,
            useBFloat16TensorCores);
    }

    internal static void AdamWUpdateResident(
        Tensor parameter,
        int deviceIndex,
        AdamWResidentState state,
        float beta1,
        float beta2,
        float learningRate,
        float weightDecay,
        float updateScale,
        float scaledEpsilon,
        bool applyWeightDecay)
    {
        var dataBuffer = parameter.EnsureCudaMasterFloat32Buffer(deviceIndex);
        var gradientBuffer = parameter.EnsureCudaGradientBuffer(deviceIndex);
        AdamWResidentState.Buffers stateBuffers =
            state.GetOrCreate(deviceIndex);
        (nint compute, bool physicalBFloat16) = GetComputeDestination(
            parameter, deviceIndex);
        CudaOptimizerNative.AdamWAndPublish(
            deviceIndex,
            dataBuffer.NativePtr,
            gradientBuffer.NativePtr,
            stateBuffers.First.NativePtr,
            stateBuffers.Second.NativePtr,
            compute,
            parameter.Numel,
            beta1,
            beta2,
            learningRate,
            weightDecay,
            updateScale,
            scaledEpsilon,
            applyWeightDecay,
            bfloat16State: false,
            physicalBFloat16);
    }

    internal static void AdamWUpdateBFloat16Resident(
        Tensor parameter,
        int deviceIndex,
        AdamWBFloat16ResidentState state,
        float beta1,
        float beta2,
        float learningRate,
        float weightDecay,
        float updateScale,
        float scaledEpsilon,
        bool applyWeightDecay)
    {
        bool pureBFloat16 =
            TensorExecutionContext.ActivePrecisionPolicy?.OptimizerState
                == NNtrain.Runtime.Execution.NumericFormat.BFloat16
            && parameter.DType == TensorDType.BFloat16;
        if (pureBFloat16)
        {
            NativeCudaBuffer<ushort> data =
                parameter.EnsureCudaBFloat16Buffer(deviceIndex);
            if (!parameter.TryGetCudaBFloat16GradientBuffer(
                    deviceIndex,
                    out NativeCudaBuffer<ushort>? gradient))
            {
                throw new InvalidOperationException(
                    $"Pure BFloat16 AdamW requires a resident BF16 " +
                    $"gradient for parameter '{parameter.Name}' on " +
                    $"CUDA device {deviceIndex}.");
            }
            AdamWBFloat16ResidentState.Buffers pureState =
                state.GetOrCreate(deviceIndex);
            CudaOptimizerNative.AdamWPureBFloat16(
                deviceIndex,
                data.NativePtr,
                gradient!.NativePtr,
                pureState.First.NativePtr,
                pureState.Second.NativePtr,
                parameter.Numel,
                beta1,
                beta2,
                learningRate,
                weightDecay,
                updateScale,
                scaledEpsilon,
                applyWeightDecay);
            return;
        }

        var dataBuffer = parameter.EnsureCudaMasterFloat32Buffer(deviceIndex);
        var gradientBuffer = parameter.EnsureCudaGradientBuffer(deviceIndex);
        AdamWBFloat16ResidentState.Buffers stateBuffers =
            state.GetOrCreate(deviceIndex);
        (nint compute, bool physicalBFloat16) = GetComputeDestination(
            parameter, deviceIndex);
        CudaOptimizerNative.AdamWAndPublish(
            deviceIndex,
            dataBuffer.NativePtr,
            gradientBuffer.NativePtr,
            stateBuffers.First.NativePtr,
            stateBuffers.Second.NativePtr,
            compute,
            parameter.Numel,
            beta1,
            beta2,
            learningRate,
            weightDecay,
            updateScale,
            scaledEpsilon,
            applyWeightDecay,
            bfloat16State: true,
            physicalBFloat16);
    }

    private static (nint Compute, bool PhysicalBFloat16) GetComputeDestination(
        Tensor parameter,
        int deviceIndex)
    {
        (nint compute, bool physicalBFloat16, _) =
            GetComputeDestinationWithOwner(parameter, deviceIndex);
        return (compute, physicalBFloat16);
    }

    private static (nint Compute, bool PhysicalBFloat16, object? Owner)
        GetComputeDestinationWithOwner(
            Tensor parameter,
            int deviceIndex)
    {
        if (parameter.DType == TensorDType.Float32)
            return (0, false, null);
        if (parameter.DType == TensorDType.BFloat16)
        {
            NativeCudaBuffer<ushort> buffer =
                parameter.EnsureCudaBFloat16Buffer(deviceIndex);
            return (
                buffer.NativePtr,
                true,
                buffer);
        }
        if (parameter.DType == TensorDType.Bfp8)
            return (0, false, null);
        NativeCudaBuffer<float> compute =
            parameter.EnsureCudaFloat32Buffer(deviceIndex);
        return (compute.NativePtr, false, compute);
    }

    internal static void SynchronizeDevices(
        IReadOnlyList<int> deviceIndices,
        string operation)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(operation);
        foreach (int deviceIndex in deviceIndices)
        {
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex).Synchronize(
                $"{operation} (CUDA device {deviceIndex})");
        }
    }

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
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        using var dataBuffer = accelerator.Allocate1D(data);
        using var gradientBuffer = accelerator.Allocate1D(gradient);
        using var firstMomentBuffer = accelerator.Allocate1D(firstMoment);
        using var secondMomentBuffer = accelerator.Allocate1D(secondMoment);
        CudaOptimizerNative.AdamW(
            Tensor.CudaDeviceIndex,
            dataBuffer.NativePtr,
            gradientBuffer.NativePtr,
            firstMomentBuffer.NativePtr,
            secondMomentBuffer.NativePtr,
            data.Length,
            beta1,
            beta2,
            learningRate,
            weightDecay,
            updateScale,
            scaledEpsilon,
            applyWeightDecay,
            bfloat16State: false);
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
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        using var gradientBuffer = accelerator.Allocate1D(gradient);
        using var fastBuffer = accelerator.Allocate1D(fast);
        using var slowBuffer = accelerator.Allocate1D(slow);
        using var fastHatBuffer = accelerator.Allocate1D<float>(fastHat.Length);
        using var slowHatBuffer = accelerator.Allocate1D<float>(slowHat.Length);
        CudaOptimizerNative.NekoMoments(
            Tensor.CudaDeviceIndex,
            gradientBuffer.NativePtr,
            fastBuffer.NativePtr,
            slowBuffer.NativePtr,
            fastHatBuffer.NativePtr,
            slowHatBuffer.NativePtr,
            fast.Length,
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
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        using var dataBuffer = accelerator.Allocate1D(data);
        using var updateBuffer = accelerator.Allocate1D(update);
        CudaOptimizerNative.NekoApply(
            Tensor.CudaDeviceIndex,
            dataBuffer.NativePtr,
            updateBuffer.NativePtr,
            data.Length,
            learningRate,
            finalScale,
            weightDecay,
            applyWeightDecay);
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
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        using var sourceBuffer = accelerator.Allocate1D(source);
        using var destinationBuffer =
            accelerator.Allocate1D<float>(destination.Length);
        using var gramBuffer = accelerator.Allocate1D<float>(gram.Length);
        using var gramSquaredBuffer =
            accelerator.Allocate1D<float>(gramSquared.Length);
        CudaOptimizerNative.SymmetricGram(
            Tensor.CudaDeviceIndex,
            sourceBuffer.NativePtr,
            gramBuffer.NativePtr,
            rows,
            columns);
        CudaOptimizerNative.SymmetricGram(
            Tensor.CudaDeviceIndex,
            gramBuffer.NativePtr,
            gramSquaredBuffer.NativePtr,
            rows,
            rows);
        CudaOptimizerNative.NewtonSchulz(
            Tensor.CudaDeviceIndex,
            sourceBuffer.NativePtr,
            gramBuffer.NativePtr,
            gramSquaredBuffer.NativePtr,
            destinationBuffer.NativePtr,
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
}
