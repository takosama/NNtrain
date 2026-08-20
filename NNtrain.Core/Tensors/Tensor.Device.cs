using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace NNtrain;

public partial class Tensor
{
    private readonly object _deviceSync = new();
    private TensorDevice _device;
    private readonly Dictionary<int, DeviceBuffer> _cudaBuffers = [];
    private long _cudaBufferDataVersion = -1;

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

            float[] data = DataBuffer;
            buffer.Buffer.CopyToCPU(data);
            SynchronizeStorageFromMaster();
            _cudaBufferDataVersion = _dataVersion;
        }
    }

    internal void InvalidateCudaBuffers()
    {
        lock (_deviceSync)
        {
            foreach (DeviceBuffer buffer in _cudaBuffers.Values)
                buffer.Dispose();
            _cudaBuffers.Clear();
            _cudaBufferDataVersion = -1;
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
