using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace NNtrain;

public partial class Tensor
{
    private readonly object _deviceSync = new();
    private TensorDevice _device;
    private readonly Dictionary<int, DeviceBuffer> _cudaBuffers = [];
    private readonly Dictionary<int, DeviceBuffer> _cudaMasterBuffers = [];
    private long _cudaBufferDataVersion = -1;
    private bool _hostDataCurrent = true;

    public TensorDevice Device => _device;

    public Tensor To(TensorDevice device)
    {
        if (!Enum.IsDefined(device))
            throw new ArgumentOutOfRangeException(nameof(device));

        if (device == TensorDevice.Cuda)
        {
            EnsureCudaFloat32Buffer();
            _device = TensorDevice.Cuda;
            return this;
        }

        if (_device == TensorDevice.Cuda)
            SynchronizeHostFromCuda();
        _device = TensorDevice.Cpu;
        return this;
    }

    public Tensor to(TensorDevice device) => To(device);

    internal MemoryBuffer1D<float, Stride1D.Dense> EnsureCudaFloat32Buffer(
        int deviceIndex = -1)
    {
        int resolvedDeviceIndex = deviceIndex < 0
            ? CudaDeviceIndex
            : deviceIndex;
        CudaAccelerator accelerator =
            NNtrain.ForgetMemoryV2Cuda.GetAccelerator(resolvedDeviceIndex);
        if (!_hostDataCurrent
            && !_cudaBuffers.ContainsKey(resolvedDeviceIndex))
        {
            SynchronizeHostFromCuda();
        }
        lock (_deviceSync)
        {
            if (!_cudaBuffers.TryGetValue(
                resolvedDeviceIndex,
                out DeviceBuffer? buffer)
                || buffer.Buffer.Length != Numel)
            {
                buffer?.Dispose();
                buffer = new DeviceBuffer(
                    accelerator.Allocate1D(GetPhysicalFloat32ComputeCache()));
                _cudaBuffers[resolvedDeviceIndex] = buffer;
                _cudaBufferDataVersion = _dataVersion;
                return buffer.Buffer;
            }

            if (_cudaBufferDataVersion != _dataVersion)
            {
                buffer.Buffer.CopyFromCPU(GetPhysicalFloat32ComputeCache());
                _cudaBufferDataVersion = _dataVersion;
            }
            return buffer.Buffer;
        }
    }

    internal MemoryBuffer1D<float, Stride1D.Dense> EnsureCudaMasterFloat32Buffer(
        int deviceIndex = -1)
    {
        int resolvedDeviceIndex = deviceIndex < 0
            ? CudaDeviceIndex
            : deviceIndex;
        if (DType == TensorDType.Float32)
            return EnsureCudaFloat32Buffer(resolvedDeviceIndex);
        CudaAccelerator accelerator =
            NNtrain.ForgetMemoryV2Cuda.GetAccelerator(resolvedDeviceIndex);
        if (!_hostDataCurrent
            && !_cudaMasterBuffers.ContainsKey(resolvedDeviceIndex))
        {
            SynchronizeHostFromCuda();
        }
        lock (_deviceSync)
        {
            if (!_cudaMasterBuffers.TryGetValue(
                resolvedDeviceIndex,
                out DeviceBuffer? buffer)
                || buffer.Buffer.Length != Numel)
            {
                buffer?.Dispose();
                buffer = new DeviceBuffer(
                    accelerator.Allocate1D(DataBuffer));
                _cudaMasterBuffers[resolvedDeviceIndex] = buffer;
            }
            return buffer.Buffer;
        }
    }

    internal void SynchronizeHostFromCuda(int deviceIndex = -1)
    {
        int resolvedDeviceIndex = deviceIndex < 0
            ? CudaDeviceIndex
            : deviceIndex;
        lock (_deviceSync)
        {
            if (!_cudaBuffers.TryGetValue(
                resolvedDeviceIndex,
                out DeviceBuffer? buffer))
            {
                return;
            }

            if (_hostDataCurrent)
                return;

            float[] data = DataBuffer;
            if (_cudaMasterBuffers.TryGetValue(
                resolvedDeviceIndex,
                out DeviceBuffer? masterBuffer))
            {
                masterBuffer.Buffer.CopyToCPU(data);
            }
            else
            {
                buffer.Buffer.CopyToCPU(data);
            }
            if (DType != TensorDType.Float32)
                _data.CopyFrom(data);
            _physicalFloat32CacheDataVersion = -1;
            _hostDataCurrent = true;
            _cudaBufferDataVersion = _dataVersion;
        }
    }

    internal void MarkCudaDataMutated(int deviceIndex = -1)
    {
        int resolvedDeviceIndex = deviceIndex < 0
            ? CudaDeviceIndex
            : deviceIndex;
        lock (_deviceSync)
        {
            if (!_cudaBuffers.ContainsKey(resolvedDeviceIndex))
            {
                throw new InvalidOperationException(
                    "Cannot mark CUDA data modified before allocating its buffer.");
            }
            unchecked
            {
                _dataVersion++;
            }
            _cudaBufferDataVersion = _dataVersion;
            _physicalFloat32CacheDataVersion = -1;
            _hostDataCurrent = false;
            _device = TensorDevice.Cuda;
        }
    }

    private void EnsureHostDataCurrent()
    {
        if (!_hostDataCurrent)
            SynchronizeHostFromCuda();
    }

    internal void InvalidateCudaBuffers()
    {
        lock (_deviceSync)
        {
            foreach (DeviceBuffer buffer in _cudaBuffers.Values)
                buffer.Dispose();
            foreach (DeviceBuffer buffer in _cudaMasterBuffers.Values)
                buffer.Dispose();
            _cudaBuffers.Clear();
            _cudaMasterBuffers.Clear();
            _cudaBufferDataVersion = -1;
            _hostDataCurrent = true;
            _device = TensorDevice.Cpu;
        }
    }

    private sealed class DeviceBuffer : IDisposable
    {
        internal DeviceBuffer(
            MemoryBuffer1D<float, Stride1D.Dense> buffer)
        {
            Buffer = buffer;
        }

        internal MemoryBuffer1D<float, Stride1D.Dense> Buffer { get; }

        public void Dispose() => Buffer.Dispose();
    }
}
