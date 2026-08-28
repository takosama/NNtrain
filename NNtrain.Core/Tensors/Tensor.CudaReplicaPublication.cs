namespace NNtrain;

public partial class Tensor
{
    /// <summary>
    /// Publishes a parameter update which was executed on every listed CUDA
    /// replica.  The logical generation advances once for the update, not once
    /// per device, so no already-updated peer is made stale and rebuilt through
    /// the host on the following optimizer step.
    /// </summary>
    internal void MarkCudaDataReplicasSynchronized(
        IReadOnlyList<int> deviceIndices)
    {
        ArgumentNullException.ThrowIfNull(deviceIndices);
        if (deviceIndices.Count == 0
            || deviceIndices.Any(static deviceIndex => deviceIndex < 0)
            || deviceIndices.Distinct().Count() != deviceIndices.Count)
        {
            throw new ArgumentException(
                "CUDA data publication requires unique, non-negative devices.",
                nameof(deviceIndices));
        }
        if (DType == TensorDType.Bfp8)
        {
            MarkCudaBfp8DataReplicasSynchronized(deviceIndices);
            return;
        }

        lock (_deviceSync)
        {
            bool publishesMasters = deviceIndices.Any(
                deviceIndex => _cudaMasterBuffers.ContainsKey(deviceIndex));
            foreach (int deviceIndex in deviceIndices)
            {
                bool hasPhysicalReplica = DType == TensorDType.BFloat16
                    ? _cudaBFloat16Buffers.ContainsKey(deviceIndex)
                    : _cudaBuffers.ContainsKey(deviceIndex);
                if (!hasPhysicalReplica)
                {
                    throw new InvalidOperationException(
                        $"CUDA device {deviceIndex} has no {DType} data " +
                        "replica to publish.");
                }
                bool usablePhysicalReplica = DType == TensorDType.BFloat16
                    ? IsReplicaUsableInCurrentSession(
                        _cudaBFloat16Buffers[deviceIndex].Buffer)
                    : IsReplicaUsableInCurrentSession(
                        _cudaBuffers[deviceIndex].Buffer);
                if (!usablePhysicalReplica)
                {
                    throw new InvalidOperationException(
                        $"CUDA device {deviceIndex} data replica belongs to " +
                        "a closed or different execution session.");
                }
                if (publishesMasters
                    && !_cudaMasterBuffers.ContainsKey(deviceIndex))
                {
                    throw new InvalidOperationException(
                        $"CUDA device {deviceIndex} has no float32 master " +
                        "replica to publish.");
                }
                if (publishesMasters
                    && !IsReplicaUsableInCurrentSession(
                        _cudaMasterBuffers[deviceIndex].Buffer))
                {
                    throw new InvalidOperationException(
                        $"CUDA device {deviceIndex} master replica belongs " +
                        "to a closed or different execution session.");
                }
            }

            unchecked
            {
                _dataVersion++;
            }
            foreach (int deviceIndex in deviceIndices)
            {
                if (_cudaBuffers.TryGetValue(
                        deviceIndex,
                        out DeviceBuffer? float32))
                {
                    float32.Version = _dataVersion;
                }
                if (_cudaBFloat16Buffers.TryGetValue(
                        deviceIndex,
                        out BFloat16DeviceBuffer? bfloat16))
                {
                    bfloat16.Version = _dataVersion;
                }
                if (_cudaMasterBuffers.TryGetValue(
                        deviceIndex,
                        out DeviceBuffer? master))
                {
                    master.Version = _dataVersion;
                }
            }

            _physicalFloat32CacheDataVersion = -1;
            _hostDataCurrent = false;
            _device = TensorDevice.Cuda;
            _cudaDeviceIndex = deviceIndices[0];
        }
    }
}
