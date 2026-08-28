namespace NNtrain;

public partial class Tensor
{
    /// <summary>
    /// Converts persistent parameter storage without making the host replica
    /// authoritative. The caller holds <see cref="_deviceSync"/>.
    /// </summary>
    /// <remarks>
    /// One authoritative CUDA replica is converted, then copied peer-to-peer
    /// to stale or differently-typed replicas. This both avoids a full-model
    /// D2H/H2D round trip and gives every configured device the same generation.
    /// Mixed policies retain their FP32 master as a persistent device buffer;
    /// the host storage remains a lazy placeholder until an explicit read.
    /// </remarks>
    private bool TryConvertCudaStorageInPlaceLocked(
        TensorDType targetDType,
        Bfp8QuantizationDescriptor? targetBfp8,
        bool retainFloat32Master)
    {
        if (_device != TensorDevice.Cuda
            || targetDType is not (TensorDType.Float32
                or TensorDType.BFloat16
                or TensorDType.Bfp8))
        {
            return false;
        }

        int[] devices = _cudaBuffers.Keys
            .Concat(_cudaBFloat16Buffers.Keys)
            .Concat(_cudaBfp8Buffers.Keys)
            .Concat(_cudaMasterBuffers.Keys)
            .Distinct()
            .Order()
            .ToArray();
        if (devices.Length == 0)
            return false;

        bool samePhysicalContract = DType == targetDType
            && (targetDType != TensorDType.Bfp8
                || Equals(Bfp8Quantization, targetBfp8));
        bool currentContract = samePhysicalContract
            && devices.All(deviceIndex =>
                HasCurrentCudaPhysicalLocked(targetDType, deviceIndex))
            && (retainFloat32Master
                ? devices.All(deviceIndex =>
                    _cudaMasterBuffers.TryGetValue(
                        deviceIndex,
                        out DeviceBuffer? master)
                    && master.Version == _dataVersion)
                : _cudaMasterBuffers.Count == 0);
        if (currentContract)
        {
            // A prior explicit host read is only a cache. It is safe to leave
            // it intact on an exact no-op conversion.
            return true;
        }

        int primaryDevice = SelectCudaConversionPrimaryLocked(devices);
        long sourceVersion = _dataVersion;
        var created = new HashSet<IDisposable>(
            ReferenceEqualityComparer.Instance);
        var touchedDevices = new HashSet<int>();
        var plans = new Dictionary<int, CudaStorageReplicaPlan>();

        try
        {
            CudaStorageReplicaPlan primary = BuildPrimaryCudaStoragePlanLocked(
                primaryDevice,
                targetDType,
                targetBfp8,
                retainFloat32Master,
                samePhysicalContract,
                created,
                touchedDevices);
            plans.Add(primaryDevice, primary);

            // Conversion kernels on the source lane must complete before its
            // new storage can be the source of peer copies.
            if (touchedDevices.Remove(primaryDevice))
                NativeCudaRuntime.SynchronizeDeviceComputeStream(primaryDevice);

            foreach (int deviceIndex in devices)
            {
                if (deviceIndex == primaryDevice)
                    continue;
                plans.Add(
                    deviceIndex,
                    BuildReplicaCudaStoragePlanLocked(
                        deviceIndex,
                        primary,
                        targetDType,
                        targetBfp8,
                        retainFloat32Master,
                        samePhysicalContract,
                        sourceVersion,
                        created,
                        touchedDevices));
            }

            foreach (int deviceIndex in touchedDevices)
                NativeCudaRuntime.SynchronizeDeviceComputeStream(deviceIndex);

            CommitCudaStoragePlansLocked(
                plans,
                targetDType,
                targetBfp8,
                primaryDevice,
                created);
            return true;
        }
        catch (Exception conversionFailure)
        {
            List<Exception>? cleanupFailures = null;
            foreach (int deviceIndex in touchedDevices)
            {
                try
                {
                    NativeCudaRuntime.SynchronizeDeviceComputeStream(deviceIndex);
                }
                catch (Exception cleanupFailure)
                {
                    (cleanupFailures ??= []).Add(cleanupFailure);
                }
            }
            foreach (IDisposable buffer in created)
                TryDisposeCudaConversionBuffer(buffer, ref cleanupFailures);
            if (cleanupFailures is not null)
            {
                cleanupFailures.Insert(0, conversionFailure);
                throw new AggregateException(
                    "CUDA storage conversion and rollback failed.",
                    cleanupFailures);
            }
            System.Runtime.ExceptionServices.ExceptionDispatchInfo
                .Capture(conversionFailure)
                .Throw();
            throw;
        }
    }

    private CudaStorageReplicaPlan BuildPrimaryCudaStoragePlanLocked(
        int deviceIndex,
        TensorDType targetDType,
        Bfp8QuantizationDescriptor? targetBfp8,
        bool retainFloat32Master,
        bool samePhysicalContract,
        HashSet<IDisposable> created,
        HashSet<int> touchedDevices)
    {
        CudaStorageSource source = GetCudaStorageSourceLocked(deviceIndex);
        DeviceBuffer? targetMaster = null;
        if (retainFloat32Master)
        {
            if (_cudaMasterBuffers.TryGetValue(
                    deviceIndex,
                    out DeviceBuffer? residentMaster))
            {
                targetMaster = residentMaster;
            }
            else if (source.Kind == CudaStorageSourceKind.Float32)
            {
                targetMaster = source.Float32;
            }
            else
            {
                targetMaster = CreateCudaFloat32FromSource(
                    deviceIndex, source, created, touchedDevices);
            }
        }

        DeviceBuffer? float32 = null;
        BFloat16DeviceBuffer? bfloat16 = null;
        Bfp8DeviceBuffer? bfp8 = null;
        if (samePhysicalContract
            && TryGetCudaPhysicalLocked(
                targetDType,
                deviceIndex,
                out DeviceBuffer? currentFloat32,
                out BFloat16DeviceBuffer? currentBFloat16,
                out Bfp8DeviceBuffer? currentBfp8))
        {
            float32 = currentFloat32;
            bfloat16 = currentBFloat16;
            bfp8 = currentBfp8;
        }
        else
        {
            CudaStorageSource conversionSource = targetMaster is not null
                ? CudaStorageSource.FromFloat32(targetMaster)
                : source;
            switch (targetDType)
            {
                case TensorDType.Float32:
                    float32 = conversionSource.Kind
                        == CudaStorageSourceKind.Float32
                            ? conversionSource.Float32
                            : CreateCudaFloat32FromSource(
                                deviceIndex,
                                conversionSource,
                                created,
                                touchedDevices);
                    break;
                case TensorDType.BFloat16:
                    bfloat16 = CreateCudaBFloat16FromSource(
                        deviceIndex,
                        conversionSource,
                        created,
                        touchedDevices);
                    break;
                case TensorDType.Bfp8:
                    bfp8 = CreateCudaBfp8FromSource(
                        deviceIndex,
                        conversionSource,
                        targetBfp8!,
                        created,
                        touchedDevices);
                    break;
                default:
                    throw new InvalidOperationException(
                        $"CUDA storage conversion to {targetDType} is unsupported.");
            }
        }

        return new CudaStorageReplicaPlan(
            deviceIndex, float32, bfloat16, bfp8, targetMaster);
    }

    private CudaStorageReplicaPlan BuildReplicaCudaStoragePlanLocked(
        int deviceIndex,
        CudaStorageReplicaPlan primary,
        TensorDType targetDType,
        Bfp8QuantizationDescriptor? targetBfp8,
        bool retainFloat32Master,
        bool samePhysicalContract,
        long sourceVersion,
        HashSet<IDisposable> created,
        HashSet<int> touchedDevices)
    {
        DeviceBuffer? float32 = null;
        BFloat16DeviceBuffer? bfloat16 = null;
        Bfp8DeviceBuffer? bfp8 = null;
        bool reuseCurrentPhysical = samePhysicalContract
            && HasCurrentCudaPhysicalLocked(targetDType, deviceIndex);
        if (reuseCurrentPhysical)
        {
            _ = TryGetCudaPhysicalLocked(
                targetDType,
                deviceIndex,
                out float32,
                out bfloat16,
                out bfp8);
        }
        else
        {
            NativeCudaDevice accelerator =
                ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
            accelerator.Bind();
            nint stream = accelerator.DefaultStream;
            switch (targetDType)
            {
                case TensorDType.Float32:
                    float32 = AllocateCudaFloat32(
                        deviceIndex, created, sourceVersion);
                    primary.Float32!.Buffer.View.CopyTo(
                        stream, float32.Buffer.View);
                    break;
                case TensorDType.BFloat16:
                    bfloat16 = AllocateCudaBFloat16(
                        deviceIndex, created, sourceVersion);
                    primary.BFloat16!.Buffer.View.CopyTo(
                        stream, bfloat16.Buffer.View);
                    break;
                case TensorDType.Bfp8:
                    bfp8 = AllocateCudaBfp8(
                        deviceIndex,
                        targetBfp8!,
                        created,
                        sourceVersion);
                    primary.Bfp8!.Payload.View.CopyTo(
                        stream, bfp8.Payload.View);
                    primary.Bfp8.Scales.View.CopyTo(
                        stream, bfp8.Scales.View);
                    break;
                default:
                    throw new InvalidOperationException(
                        $"CUDA storage conversion to {targetDType} is unsupported.");
            }
            touchedDevices.Add(deviceIndex);
        }

        DeviceBuffer? master = null;
        if (retainFloat32Master)
        {
            if (_cudaMasterBuffers.TryGetValue(
                    deviceIndex,
                    out DeviceBuffer? currentMaster)
                && currentMaster.Version == sourceVersion)
            {
                master = currentMaster;
            }
            else
            {
                NativeCudaDevice accelerator =
                    ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
                accelerator.Bind();
                nint stream = accelerator.DefaultStream;
                master = AllocateCudaFloat32(
                    deviceIndex, created, sourceVersion);
                primary.Master!.Buffer.View.CopyTo(stream, master.Buffer.View);
                touchedDevices.Add(deviceIndex);
            }
        }

        return new CudaStorageReplicaPlan(
            deviceIndex, float32, bfloat16, bfp8, master);
    }

    private CudaStorageSource GetCudaStorageSourceLocked(int deviceIndex)
    {
        bool hasCurrentMaster = _cudaMasterBuffers.TryGetValue(
                deviceIndex,
                out DeviceBuffer? master)
            && master.Version == _dataVersion;
        if (hasCurrentMaster)
            return CudaStorageSource.FromFloat32(master!);

        if (TryGetCudaPhysicalLocked(
                DType,
                deviceIndex,
                out DeviceBuffer? float32,
                out BFloat16DeviceBuffer? bfloat16,
                out Bfp8DeviceBuffer? bfp8,
                requireCurrent: true))
        {
            return CudaStorageSource.FromPhysical(
                DType, float32, bfloat16, bfp8);
        }

        // A data-parallel optimizer may have updated every replica but only
        // advanced the designated authority's managed generation. Prefer the
        // local master in that case; otherwise use its resident physical value.
        if (master is not null)
            return CudaStorageSource.FromFloat32(master);
        if (TryGetCudaPhysicalLocked(
                DType,
                deviceIndex,
                out float32,
                out bfloat16,
                out bfp8))
        {
            return CudaStorageSource.FromPhysical(
                DType, float32, bfloat16, bfp8);
        }

        throw new InvalidOperationException(
            $"Tensor '{Name}' has no CUDA storage replica on device {deviceIndex}.");
    }

    private int SelectCudaConversionPrimaryLocked(IReadOnlyList<int> devices)
    {
        int selected = devices[0];
        int selectedScore = int.MinValue;
        foreach (int deviceIndex in devices)
        {
            int score = deviceIndex == _cudaDeviceIndex ? 4 : 0;
            if (_cudaMasterBuffers.TryGetValue(
                    deviceIndex,
                    out DeviceBuffer? master))
            {
                score += master.Version == _dataVersion ? 100 : 10;
            }
            if (HasCurrentCudaPhysicalLocked(DType, deviceIndex))
                score += 50;
            else if (HasAnyCudaPhysicalLocked(DType, deviceIndex))
                score += 5;
            if (score > selectedScore)
            {
                selected = deviceIndex;
                selectedScore = score;
            }
        }
        return selected;
    }

    private DeviceBuffer CreateCudaFloat32FromSource(
        int deviceIndex,
        CudaStorageSource source,
        HashSet<IDisposable> created,
        HashSet<int> touchedDevices)
    {
        if (source.Kind == CudaStorageSourceKind.Float32)
            return source.Float32!;

        DeviceBuffer destination = AllocateCudaFloat32(
            deviceIndex, created, _dataVersion);
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        switch (source.Kind)
        {
            case CudaStorageSourceKind.BFloat16:
                CudaTensorNative.DecodeBFloat16(
                    deviceIndex,
                    source.BFloat16!.Buffer.NativePtr,
                    destination.Buffer.NativePtr,
                    Numel);
                break;
            case CudaStorageSourceKind.Bfp8:
                CudaBfp8Native.DequantizeFloat32(
                    deviceIndex,
                    source.Bfp8!.Payload,
                    source.Bfp8.Scales,
                    destination.Buffer,
                    source.Bfp8.Descriptor,
                    accelerator.DefaultStream);
                break;
            default:
                throw new InvalidOperationException(
                    "Unknown CUDA source storage kind.");
        }
        touchedDevices.Add(deviceIndex);
        return destination;
    }

    private BFloat16DeviceBuffer CreateCudaBFloat16FromSource(
        int deviceIndex,
        CudaStorageSource source,
        HashSet<IDisposable> created,
        HashSet<int> touchedDevices)
    {
        if (source.Kind == CudaStorageSourceKind.BFloat16)
            return source.BFloat16!;

        BFloat16DeviceBuffer destination = AllocateCudaBFloat16(
            deviceIndex, created, _dataVersion);
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        switch (source.Kind)
        {
            case CudaStorageSourceKind.Float32:
                CudaTensorNative.EncodeBFloat16(
                    deviceIndex,
                    source.Float32!.Buffer.NativePtr,
                    destination.Buffer.NativePtr,
                    Numel);
                break;
            case CudaStorageSourceKind.Bfp8:
                CudaBfp8Native.DequantizeBFloat16(
                    deviceIndex,
                    source.Bfp8!.Payload,
                    source.Bfp8.Scales,
                    destination.Buffer,
                    source.Bfp8.Descriptor,
                    accelerator.DefaultStream);
                break;
            default:
                throw new InvalidOperationException(
                    "Unknown CUDA source storage kind.");
        }
        touchedDevices.Add(deviceIndex);
        return destination;
    }

    private Bfp8DeviceBuffer CreateCudaBfp8FromSource(
        int deviceIndex,
        CudaStorageSource source,
        Bfp8QuantizationDescriptor descriptor,
        HashSet<IDisposable> created,
        HashSet<int> touchedDevices)
    {
        if (source.Kind == CudaStorageSourceKind.Bfp8
            && source.Bfp8!.Descriptor == descriptor)
        {
            return source.Bfp8;
        }

        Bfp8DeviceBuffer destination = AllocateCudaBfp8(
            deviceIndex, descriptor, created, _dataVersion);
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        nint stream = accelerator.DefaultStream;
        switch (source.Kind)
        {
            case CudaStorageSourceKind.Float32:
                CudaBfp8Native.QuantizeFloat32(
                    deviceIndex,
                    source.Float32!.Buffer,
                    destination.Payload,
                    destination.Scales,
                    descriptor,
                    stream);
                break;
            case CudaStorageSourceKind.BFloat16:
                CudaBfp8Native.QuantizeBFloat16(
                    deviceIndex,
                    source.BFloat16!.Buffer,
                    destination.Payload,
                    destination.Scales,
                    descriptor,
                    stream);
                break;
            case CudaStorageSourceKind.Bfp8:
                DeviceBuffer decoded = CreateCudaFloat32FromSource(
                    deviceIndex, source, created, touchedDevices);
                CudaBfp8Native.QuantizeFloat32(
                    deviceIndex,
                    decoded.Buffer,
                    destination.Payload,
                    destination.Scales,
                    descriptor,
                    stream);
                break;
            default:
                throw new InvalidOperationException(
                    "Unknown CUDA source storage kind.");
        }
        touchedDevices.Add(deviceIndex);
        return destination;
    }

    private DeviceBuffer AllocateCudaFloat32(
        int deviceIndex,
        HashSet<IDisposable> created,
        long version)
    {
        NativeCudaBuffer<float> raw =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex)
                .Allocate1D<float>(Numel);
        var owner = new DeviceBuffer(raw, version, deviceIndex);
        created.Add(owner);
        return owner;
    }

    private BFloat16DeviceBuffer AllocateCudaBFloat16(
        int deviceIndex,
        HashSet<IDisposable> created,
        long version)
    {
        NativeCudaBuffer<ushort> raw =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex)
                .Allocate1D<ushort>(Numel);
        var owner = new BFloat16DeviceBuffer(raw, version, deviceIndex);
        created.Add(owner);
        return owner;
    }

    private Bfp8DeviceBuffer AllocateCudaBfp8(
        int deviceIndex,
        Bfp8QuantizationDescriptor descriptor,
        HashSet<IDisposable> created,
        long version)
    {
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        NativeCudaBuffer<sbyte>? payload = null;
        NativeCudaBuffer<float>? scales = null;
        try
        {
            payload = accelerator.Allocate1D<sbyte>(Numel);
            scales = accelerator.Allocate1D<float>(
                descriptor.GetScaleCount(Numel));
            var owner = new Bfp8DeviceBuffer(
                payload, scales, descriptor, version);
            created.Add(owner);
            return owner;
        }
        catch
        {
            payload?.Dispose();
            scales?.Dispose();
            throw;
        }
    }

    private void CommitCudaStoragePlansLocked(
        IReadOnlyDictionary<int, CudaStorageReplicaPlan> plans,
        TensorDType targetDType,
        Bfp8QuantizationDescriptor? targetBfp8,
        int primaryDevice,
        HashSet<IDisposable> created)
    {
        DeviceBuffer[] oldFloat32 = _cudaBuffers.Values.ToArray();
        BFloat16DeviceBuffer[] oldBFloat16 =
            _cudaBFloat16Buffers.Values.ToArray();
        Bfp8DeviceBuffer[] oldBfp8 = _cudaBfp8Buffers.Values.ToArray();
        DeviceBuffer[] oldMaster = _cudaMasterBuffers.Values.ToArray();

        unchecked
        {
            _dataVersion++;
        }
        foreach (CudaStorageReplicaPlan plan in plans.Values)
        {
            if (plan.Float32 is not null)
                plan.Float32.Version = _dataVersion;
            if (plan.BFloat16 is not null)
                plan.BFloat16.Version = _dataVersion;
            if (plan.Bfp8 is not null)
                plan.Bfp8.Version = _dataVersion;
            if (plan.Master is not null)
                plan.Master.Version = _dataVersion;
        }

        _cudaBuffers.Clear();
        _cudaBFloat16Buffers.Clear();
        _cudaBfp8Buffers.Clear();
        _cudaMasterBuffers.Clear();
        foreach (CudaStorageReplicaPlan plan in plans.Values)
        {
            if (plan.Float32 is not null)
            {
                _cudaBuffers.Add(plan.DeviceIndex, plan.Float32);
                RegisterSessionReplicaLocked(plan.Float32.Buffer);
            }
            if (plan.BFloat16 is not null)
            {
                _cudaBFloat16Buffers.Add(plan.DeviceIndex, plan.BFloat16);
                RegisterSessionReplicaLocked(plan.BFloat16.Buffer);
            }
            if (plan.Bfp8 is not null)
            {
                _cudaBfp8Buffers.Add(plan.DeviceIndex, plan.Bfp8);
                RegisterSessionReplicaLocked(plan.Bfp8.Payload);
            }
            if (plan.Master is not null)
            {
                _cudaMasterBuffers.Add(plan.DeviceIndex, plan.Master);
                RegisterSessionReplicaLocked(plan.Master.Buffer);
            }
        }

        if (_hostDataCurrent
            && DType == TensorDType.Float32
            && _data.TryGetFloat32Buffer(out float[] previousHostValues))
        {
            CudaResidentArrayCache.Invalidate(previousHostValues);
        }
        CudaResidentArrayCache.Invalidate(_physicalFloat32Cache);
        _data = targetDType == TensorDType.Bfp8
            ? TensorStorage.CreateDeviceBfp8Placeholder(Numel, targetBfp8!)
            : TensorStorage.CreateDevicePlaceholder(Numel, targetDType);
        _masterData = null;
        _hostDataCurrent = false;
        _device = TensorDevice.Cuda;
        _cudaDeviceIndex = plans.ContainsKey(_cudaDeviceIndex)
            ? _cudaDeviceIndex
            : primaryDevice;
        _physicalFloat32Cache = null;
        _physicalFloat32CacheDataVersion = -1;
        _transposedDataCache = null;
        _transposedDataVersion = -1;

        var retained = new HashSet<IDisposable>(
            plans.Values.SelectMany(static plan => plan.Owners),
            ReferenceEqualityComparer.Instance);
        List<Exception>? cleanupFailures = null;
        foreach (IDisposable buffer in oldFloat32)
        {
            if (!retained.Contains(buffer))
                TryDisposeCudaConversionBuffer(buffer, ref cleanupFailures);
        }
        foreach (IDisposable buffer in oldBFloat16)
        {
            if (!retained.Contains(buffer))
                TryDisposeCudaConversionBuffer(buffer, ref cleanupFailures);
        }
        foreach (IDisposable buffer in oldBfp8)
        {
            if (!retained.Contains(buffer))
                TryDisposeCudaConversionBuffer(buffer, ref cleanupFailures);
        }
        foreach (IDisposable buffer in oldMaster)
        {
            if (!retained.Contains(buffer))
                TryDisposeCudaConversionBuffer(buffer, ref cleanupFailures);
        }
        foreach (IDisposable buffer in created)
        {
            if (!retained.Contains(buffer))
                TryDisposeCudaConversionBuffer(buffer, ref cleanupFailures);
        }
        created.Clear();
        if (cleanupFailures is not null)
        {
            throw new AggregateException(
                "One or more superseded CUDA storage replicas failed to dispose.",
                cleanupFailures);
        }
    }

    private bool HasCurrentCudaPhysicalLocked(
        TensorDType dtype,
        int deviceIndex)
        => TryGetCudaPhysicalLocked(
            dtype,
            deviceIndex,
            out _,
            out _,
            out _,
            requireCurrent: true);

    private bool HasAnyCudaPhysicalLocked(
        TensorDType dtype,
        int deviceIndex)
        => TryGetCudaPhysicalLocked(
            dtype,
            deviceIndex,
            out _,
            out _,
            out _);

    private bool TryGetCudaPhysicalLocked(
        TensorDType dtype,
        int deviceIndex,
        out DeviceBuffer? float32,
        out BFloat16DeviceBuffer? bfloat16,
        out Bfp8DeviceBuffer? bfp8,
        bool requireCurrent = false)
    {
        float32 = null;
        bfloat16 = null;
        bfp8 = null;
        switch (dtype)
        {
            case TensorDType.Float32:
            case TensorDType.Float16:
                if (!_cudaBuffers.TryGetValue(deviceIndex, out float32))
                    return false;
                return IsReplicaUsableInCurrentSession(float32.Buffer)
                    && (!requireCurrent || float32.Version == _dataVersion);
            case TensorDType.BFloat16:
                if (!_cudaBFloat16Buffers.TryGetValue(deviceIndex, out bfloat16))
                    return false;
                return IsReplicaUsableInCurrentSession(bfloat16.Buffer)
                    && (!requireCurrent || bfloat16.Version == _dataVersion);
            case TensorDType.Bfp8:
                if (!_cudaBfp8Buffers.TryGetValue(deviceIndex, out bfp8))
                    return false;
                return IsReplicaUsableInCurrentSession(bfp8.Payload)
                    && IsReplicaUsableInCurrentSession(bfp8.Scales)
                    && (!requireCurrent || bfp8.Version == _dataVersion);
            default:
                return false;
        }
    }

    private static void TryDisposeCudaConversionBuffer(
        IDisposable buffer,
        ref List<Exception>? failures)
    {
        try
        {
            buffer.Dispose();
        }
        catch (Exception failure)
        {
            (failures ??= []).Add(failure);
        }
    }

    private enum CudaStorageSourceKind
    {
        Float32,
        BFloat16,
        Bfp8,
    }

    private readonly record struct CudaStorageSource(
        CudaStorageSourceKind Kind,
        DeviceBuffer? Float32,
        BFloat16DeviceBuffer? BFloat16,
        Bfp8DeviceBuffer? Bfp8)
    {
        internal static CudaStorageSource FromFloat32(DeviceBuffer buffer)
            => new(CudaStorageSourceKind.Float32, buffer, null, null);

        internal static CudaStorageSource FromPhysical(
            TensorDType dtype,
            DeviceBuffer? float32,
            BFloat16DeviceBuffer? bfloat16,
            Bfp8DeviceBuffer? bfp8)
            => dtype switch
            {
                TensorDType.Float32 or TensorDType.Float16 =>
                    FromFloat32(float32!),
                TensorDType.BFloat16 => new(
                    CudaStorageSourceKind.BFloat16,
                    null,
                    bfloat16,
                    null),
                TensorDType.Bfp8 => new(
                    CudaStorageSourceKind.Bfp8,
                    null,
                    null,
                    bfp8),
                _ => throw new InvalidOperationException(
                    $"CUDA storage source '{dtype}' is unsupported."),
            };
    }

    private sealed record CudaStorageReplicaPlan(
        int DeviceIndex,
        DeviceBuffer? Float32,
        BFloat16DeviceBuffer? BFloat16,
        Bfp8DeviceBuffer? Bfp8,
        DeviceBuffer? Master)
    {
        internal IEnumerable<IDisposable> Owners
        {
            get
            {
                if (Float32 is not null)
                    yield return Float32;
                if (BFloat16 is not null)
                    yield return BFloat16;
                if (Bfp8 is not null)
                    yield return Bfp8;
                if (Master is not null)
                    yield return Master;
            }
        }
    }
}
