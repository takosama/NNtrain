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
    private readonly Dictionary<int, GradientDeviceBuffer> _cudaGradientBuffers = [];
    private long _cudaBufferDataVersion = -1;
    private bool _hostDataCurrent = true;
    private long _gradientVersion;
    private bool _hostGradientCurrent = true;

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

    internal MemoryBuffer1D<float, Stride1D.Dense> EnsureCudaGradientBuffer(
        int deviceIndex = -1)
    {
        int resolvedDeviceIndex = deviceIndex < 0
            ? CudaDeviceIndex
            : deviceIndex;
        CudaAccelerator accelerator =
            NNtrain.ForgetMemoryV2Cuda.GetAccelerator(resolvedDeviceIndex);
        if (!_hostGradientCurrent
            && !_cudaGradientBuffers.ContainsKey(resolvedDeviceIndex))
        {
            SynchronizeHostGradientFromCuda();
        }
        lock (_deviceSync)
        {
            if (!_cudaGradientBuffers.TryGetValue(
                resolvedDeviceIndex,
                out GradientDeviceBuffer? buffer)
                || buffer.Buffer.Length != Numel)
            {
                buffer?.Dispose();
                MemoryBuffer1D<float, Stride1D.Dense> gradientBuffer;
                if (_grad.Length == 0)
                {
                    gradientBuffer = accelerator.Allocate1D<float>(Numel);
                    gradientBuffer.MemSetToZero();
                }
                else
                {
                    gradientBuffer = accelerator.Allocate1D(_grad);
                }
                buffer = new GradientDeviceBuffer(
                    gradientBuffer,
                    _gradientVersion);
                _cudaGradientBuffers[resolvedDeviceIndex] = buffer;
                return buffer.Buffer;
            }

            if (buffer.Version != _gradientVersion)
            {
                // A different CUDA adapter may have produced another local
                // gradient for the same data-parallel step. Keep this
                // adapter's local buffer until the explicit all-reduce.
                if (!_hostGradientCurrent)
                    return buffer.Buffer;
                buffer.Buffer.CopyFromCPU(_grad);
                buffer.Version = _gradientVersion;
            }
            return buffer.Buffer;
        }
    }

    internal void MarkCudaGradientMutated(int deviceIndex = -1)
    {
        int resolvedDeviceIndex = deviceIndex < 0
            ? CudaDeviceIndex
            : deviceIndex;
        lock (_deviceSync)
        {
            if (!_cudaGradientBuffers.TryGetValue(
                resolvedDeviceIndex,
                out GradientDeviceBuffer? buffer))
            {
                throw new InvalidOperationException(
                    "Cannot mark a CUDA gradient modified before allocating it.");
            }
            unchecked
            {
                _gradientVersion++;
            }
            buffer.Version = _gradientVersion;
            _hostGradientCurrent = false;
        }
    }

    internal void PrepareCudaGradientBuffers(IReadOnlyList<int> deviceIndices)
    {
        ArgumentNullException.ThrowIfNull(deviceIndices);
        foreach (int deviceIndex in deviceIndices)
            EnsureCudaGradientBuffer(deviceIndex);
    }

    internal void MarkCudaGradientsSynchronized(IReadOnlyList<int> deviceIndices)
    {
        lock (_deviceSync)
        {
            unchecked
            {
                _gradientVersion++;
            }
            foreach (int deviceIndex in deviceIndices)
            {
                if (_cudaGradientBuffers.TryGetValue(
                    deviceIndex,
                    out GradientDeviceBuffer? buffer))
                {
                    buffer.Version = _gradientVersion;
                }
            }
            _hostGradientCurrent = false;
        }
    }

    internal void SetCudaGradient(float[] values, int deviceIndex = -1)
    {
        ArgumentNullException.ThrowIfNull(values);
        if (values.Length != Numel)
            throw new ArgumentException("Gradient length must match the tensor.", nameof(values));
        int resolvedDeviceIndex = deviceIndex < 0
            ? CudaDeviceIndex
            : deviceIndex;
        MemoryBuffer1D<float, Stride1D.Dense> buffer =
            EnsureCudaGradientBuffer(resolvedDeviceIndex);
        buffer.CopyFromCPU(values);
        MarkCudaGradientMutated(resolvedDeviceIndex);
    }

    internal void AdoptCudaFloat32Buffer(
        MemoryBuffer1D<float, Stride1D.Dense> buffer,
        int deviceIndex)
    {
        ArgumentNullException.ThrowIfNull(buffer);
        if (buffer.Length != Numel)
            throw new ArgumentException("CUDA buffer length must match the tensor.", nameof(buffer));
        lock (_deviceSync)
        {
            if (_cudaBuffers.Remove(deviceIndex, out DeviceBuffer? previous))
                previous.Dispose();
            _cudaBuffers[deviceIndex] = new DeviceBuffer(buffer);
            _cudaBufferDataVersion = _dataVersion;
            _hostDataCurrent = false;
            _device = TensorDevice.Cuda;
        }
    }

    private static Tensor FromCudaResult(
        MemoryBuffer1D<float, Stride1D.Dense> buffer,
        int deviceIndex,
        int[] shape,
        Tensor[] parents,
        TensorDType? dtype = null)
    {
        TensorDType resultDType = dtype ?? TensorDTypeContract.Promote(parents);
        var result = new Tensor(
            TensorStorage.CreateDevicePlaceholder(
                checked((int)buffer.Length),
                resultDType),
            shape,
            parents);
        result.AdoptCudaFloat32Buffer(buffer, deviceIndex);
        return result;
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

    private void EnsureHostGradientCurrent()
    {
        if (!_hostGradientCurrent)
            SynchronizeHostGradientFromCuda();
    }

    private void SynchronizeHostGradientFromCuda(int deviceIndex = -1)
    {
        lock (_deviceSync)
            SynchronizeHostGradientFromCudaLocked(deviceIndex);
    }

    private void SynchronizeHostGradientFromCudaLocked(int deviceIndex = -1)
    {
        if (_hostGradientCurrent)
            return;
        int resolvedDeviceIndex = deviceIndex < 0
            ? CudaDeviceIndex
            : deviceIndex;
        if (!_cudaGradientBuffers.TryGetValue(
            resolvedDeviceIndex,
            out GradientDeviceBuffer? buffer))
        {
            buffer = _cudaGradientBuffers.Values.FirstOrDefault(
                candidate => candidate.Version == _gradientVersion);
        }
        if (buffer is null)
            return;
        if (_grad.Length == 0)
            _grad = new float[Numel];
        buffer.Buffer.CopyToCPU(_grad);
        _hostGradientCurrent = true;
    }

    private void MarkHostGradientMutable()
    {
        unchecked
        {
            _gradientVersion++;
        }
        _hostGradientCurrent = true;
    }

    private void ClearCudaGradients()
    {
        lock (_deviceSync)
        {
            unchecked
            {
                _gradientVersion++;
            }
            foreach (GradientDeviceBuffer buffer in _cudaGradientBuffers.Values)
            {
                buffer.Buffer.MemSetToZero();
                buffer.Version = _gradientVersion;
            }
            _hostGradientCurrent = true;
        }
    }

    internal void InvalidateCudaBuffers()
    {
        lock (_deviceSync)
        {
            foreach (DeviceBuffer buffer in _cudaBuffers.Values)
                buffer.Dispose();
            foreach (DeviceBuffer buffer in _cudaMasterBuffers.Values)
                buffer.Dispose();
            foreach (GradientDeviceBuffer buffer in _cudaGradientBuffers.Values)
                buffer.Dispose();
            _cudaBuffers.Clear();
            _cudaMasterBuffers.Clear();
            _cudaGradientBuffers.Clear();
            _cudaBufferDataVersion = -1;
            _hostDataCurrent = true;
            _device = TensorDevice.Cpu;
        }
    }

    internal void ReleaseCudaGraphBuffers()
    {
        lock (_deviceSync)
        {
            foreach (DeviceBuffer buffer in _cudaBuffers.Values)
                buffer.Dispose();
            foreach (DeviceBuffer buffer in _cudaMasterBuffers.Values)
                buffer.Dispose();
            foreach (GradientDeviceBuffer buffer in _cudaGradientBuffers.Values)
                buffer.Dispose();
            _cudaBuffers.Clear();
            _cudaMasterBuffers.Clear();
            _cudaGradientBuffers.Clear();
            _hostDataCurrent = true;
            _hostGradientCurrent = true;
            _device = TensorDevice.Cpu;
        }
    }

    private sealed class DeviceBuffer : IDisposable
    {
        private int _disposed;

        internal DeviceBuffer(
            MemoryBuffer1D<float, Stride1D.Dense> buffer)
        {
            Buffer = buffer;
        }

        internal MemoryBuffer1D<float, Stride1D.Dense> Buffer { get; }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            Buffer.Dispose();
            GC.SuppressFinalize(this);
        }

        ~DeviceBuffer()
        {
            if (Interlocked.Exchange(ref _disposed, 1) == 0)
                Buffer.Dispose();
        }
    }

    private sealed class GradientDeviceBuffer(
        MemoryBuffer1D<float, Stride1D.Dense> buffer,
        long version) : IDisposable
    {
        private int _disposed;
        internal MemoryBuffer1D<float, Stride1D.Dense> Buffer { get; } = buffer;
        internal long Version { get; set; } = version;

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            Buffer.Dispose();
            GC.SuppressFinalize(this);
        }

        ~GradientDeviceBuffer()
        {
            if (Interlocked.Exchange(ref _disposed, 1) == 0)
                Buffer.Dispose();
        }
    }
}
