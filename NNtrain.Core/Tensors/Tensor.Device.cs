using NNtrain.Runtime.Execution;

namespace NNtrain;

public partial class Tensor
{
    internal enum GradientStorageAuthority
    {
        Host,
        CudaFloat32,
        CudaBFloat16,
        CudaBfp8,
    }

    private object _deviceSync => _value.Replicas.Sync;
    private TensorDevice _device
    {
        get => _value.Replicas.Device;
        set => _value.Replicas.Device = value;
    }
    private int _cudaDeviceIndex
    {
        get => _value.Replicas.CudaDeviceIndex;
        set => _value.Replicas.CudaDeviceIndex = value;
    }
    private Dictionary<int, DeviceBuffer> _cudaBuffers
        => _value.Replicas.Float32Data;
    private Dictionary<int, BFloat16DeviceBuffer> _cudaBFloat16Buffers
        => _value.Replicas.BFloat16Data;
    private Dictionary<int, Bfp8DeviceBuffer> _cudaBfp8Buffers
        => _value.Replicas.Bfp8Data;
    private Dictionary<int, DeviceBuffer> _cudaMasterBuffers
        => _value.Replicas.Float32Masters;
    private Dictionary<int, GradientDeviceBuffer> _cudaGradientBuffers
        => _value.Replicas.Float32Gradients;
    private Dictionary<int, BFloat16GradientDeviceBuffer>
        _cudaBFloat16GradientBuffers
        => _value.Replicas.BFloat16Gradients;
    private Dictionary<int, Bfp8GradientDeviceBuffer>
        _cudaBfp8GradientBuffers
        => _value.Replicas.Bfp8Gradients;
    private Dictionary<int, DeviceBuffer> _cudaStagingBuffers
        => _value.Replicas.Staging;
    private bool _hostDataCurrent
    {
        get => _value.Replicas.HostDataCurrent;
        set => _value.Replicas.HostDataCurrent = value;
    }
    private long _gradientVersion
    {
        get => _value.Replicas.GradientVersion;
        set => _value.Replicas.GradientVersion = value;
    }
    private bool _hostGradientCurrent
    {
        get => _value.Replicas.HostGradientCurrent;
        set => _value.Replicas.HostGradientCurrent = value;
    }
    private GradientStorageAuthority _gradientAuthority
    {
        get => _value.Replicas.GradientAuthority;
        set => _value.Replicas.GradientAuthority = value;
    }
    private int _gradientAuthorityDeviceIndex
    {
        get => _value.Replicas.GradientAuthorityDeviceIndex;
        set => _value.Replicas.GradientAuthorityDeviceIndex = value;
    }
    private CudaGradientCoherenceKind _gradientCoherenceKind
    {
        get => _value.Replicas.GradientCoherenceKind;
        set => _value.Replicas.GradientCoherenceKind = value;
    }
    private int _gradientLocalDeviceIndex
    {
        get => _value.Replicas.GradientLocalDeviceIndex;
        set => _value.Replicas.GradientLocalDeviceIndex = value;
    }
    private int[] _gradientReducedDevices
    {
        get => _value.Replicas.GradientReducedDevices;
        set => _value.Replicas.GradientReducedDevices = value;
    }
    private CudaGradientReductionStamp _gradientReductionStamp
    {
        get => _value.Replicas.GradientReductionStamp;
        set => _value.Replicas.GradientReductionStamp = value;
    }
    private int[] _pendingGradientReductionDevices
    {
        get => _value.Replicas.PendingGradientReductionDevices;
        set => _value.Replicas.PendingGradientReductionDevices = value;
    }
    private CudaGradientReductionStamp _pendingGradientReductionStamp
    {
        get => _value.Replicas.PendingGradientReductionStamp;
        set => _value.Replicas.PendingGradientReductionStamp = value;
    }
    private long _registeredGradientReducerGeneration
    {
        get => _value.Replicas.RegisteredGradientReducerGeneration;
        set => _value.Replicas.RegisteredGradientReducerGeneration = value;
    }
    private long _gradientZeroOwnerGeneration
    {
        get => _value.Replicas.GradientZeroOwnerGeneration;
        set => _value.Replicas.GradientZeroOwnerGeneration = value;
    }
    private int[] _gradientZeroOwnerDevices
    {
        get => _value.Replicas.GradientZeroOwnerDevices;
        set => _value.Replicas.GradientZeroOwnerDevices = value;
    }
    private ulong _reducerOwnedGradientZeroPendingMask
    {
        get => _value.Replicas.ReducerOwnedGradientZeroPendingMask;
        set => _value.Replicas.ReducerOwnedGradientZeroPendingMask = value;
    }
    private long _optimizerConsumedGradientVersion
    {
        get => _value.Replicas.OptimizerConsumedGradientVersion;
        set => _value.Replicas.OptimizerConsumedGradientVersion = value;
    }
    private CudaGradientReductionStamp _optimizerConsumedReductionStamp
    {
        get => _value.Replicas.OptimizerConsumedReductionStamp;
        set => _value.Replicas.OptimizerConsumedReductionStamp = value;
    }

    public TensorDevice Device => _device;

    public TorchDevice device
        => new(
            _device,
            _device == TensorDevice.Cuda ? _cudaDeviceIndex : _device == TensorDevice.Arc ? _arcDeviceIndex : 0);

    internal int[] GetResidentCudaDeviceIndices()
    {
        lock (_deviceSync)
        {
            return _cudaBuffers.Keys
                .Concat(_cudaBFloat16Buffers.Keys)
                .Concat(_cudaBfp8Buffers.Keys)
                .Concat(_cudaMasterBuffers.Keys)
                .Distinct()
                .Order()
                .ToArray();
        }
    }

    public Tensor To(TensorDevice device)
        => to(new TorchDevice(
            device,
            device == TensorDevice.Cuda ? CudaDeviceIndex : 0));

    public Tensor to(TorchDevice device)
    {
        if (!device.IsArc && _arcReplica is not null) ReleaseArcReplica(preserve: true);
        if (device.IsArc)
        {
            if (!IsArcAvailable(device.Index))
                throw new InvalidOperationException($"Intel Arc device {device.Index} is unavailable; no CPU fallback.");
            EnsureHostDataCurrent();
            _device = TensorDevice.Arc;
            _arcDeviceIndex = device.Index;
            return this;
        }
        if (device.IsCuda)
        {
            if (DType == TensorDType.BFloat16)
                EnsureCudaBFloat16Buffer(device.Index);
            else if (DType == TensorDType.Bfp8)
                EnsureCudaBfp8Buffer(device.Index);
            else
                EnsureCudaFloat32Buffer(device.Index);
            _device = TensorDevice.Cuda;
            _cudaDeviceIndex = device.Index;
            return this;
        }

        if (_device == TensorDevice.Cuda)
            SynchronizeHostFromCuda();
        _device = TensorDevice.Cpu;
        return this;
    }

    public Tensor to(TensorDevice device) => To(device);

    private int ResolveCudaDeviceIndex(int requestedDeviceIndex)
    {
        if (requestedDeviceIndex >= 0)
            return requestedDeviceIndex;
        return ExecutionDevice == TensorDevice.Cuda
            ? CudaDeviceIndex
            : _cudaDeviceIndex;
    }


}
