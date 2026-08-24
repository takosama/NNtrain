namespace NNtrain;

internal interface ITensorBackend
{
    TensorDevice DeviceType { get; }
    string GetName(int deviceIndex);
    bool IsAvailable(int deviceIndex);
}

internal static class TensorBackends
{
    private static readonly ITensorBackend Cpu = new CpuTensorBackend();
    private static readonly ITensorBackend Cuda = new CudaTensorBackend();

    internal static ITensorBackend Get(TensorDevice device)
        => device switch
        {
            TensorDevice.Cpu => Cpu,
            TensorDevice.Cuda => Cuda,
            _ => throw new ArgumentOutOfRangeException(nameof(device)),
        };

    private sealed class CpuTensorBackend : ITensorBackend
    {
        public TensorDevice DeviceType => TensorDevice.Cpu;
        public string GetName(int deviceIndex) => "CPU";
        public bool IsAvailable(int deviceIndex) => deviceIndex == 0;
    }

    private sealed class CudaTensorBackend : ITensorBackend
    {
        public TensorDevice DeviceType => TensorDevice.Cuda;

        public string GetName(int deviceIndex)
        {
            if (!IsAvailable(deviceIndex))
                return $"CUDA:{deviceIndex} (unavailable)";
            return ForgetMemoryV2Cuda.GetAccelerator(deviceIndex).Name;
        }

        public bool IsAvailable(int deviceIndex)
            => ForgetMemoryV2Cuda.IsAvailable(deviceIndex);
    }
}
