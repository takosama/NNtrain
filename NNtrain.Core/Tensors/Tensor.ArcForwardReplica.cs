using NNtrain.Arc;
using NNtrain.Runtime.Execution;

namespace NNtrain;

partial class Tensor
{
    /// <summary>
    /// Copies the published compute weights to another Arc lane. This is for a
    /// forward/backward-only data-parallel replica: its FP32 optimizer master
    /// is deliberately discarded and never used by that worker.
    /// </summary>
    internal void CopyArcForwardReplicaFrom(
        Tensor source, int sourceDevice, int targetDevice,
        ref Array? payload, ref float[]? scales)
    {
        ArgumentNullException.ThrowIfNull(source);
        if (ReferenceEquals(this, source) || sourceDevice == targetDevice
            || DType != source.DType || Numel != source.Numel
            || !Shape.SequenceEqual(source.Shape)
            || Bfp8Quantization != source.Bfp8Quantization)
            throw new ArgumentException("Arc forward replica requires matching tensors on different devices.", nameof(source));

        using (TensorExecutionContext.Push(new TorchDevice(TensorDevice.Arc, sourceDevice)))
        {
            source.EnsureArcPacked();
            ArcReplica primary = source._arcReplica!;
            if (primary.Lane.DeviceIndex != sourceDevice || primary.Value is null)
                throw new InvalidOperationException("The primary Arc parameter has no published compute weights.");
            if (DType == TensorDType.Bfp8)
            {
                sbyte[] bytes = payload as sbyte[] ?? [];
                if (bytes.Length != Numel) bytes = new sbyte[Numel];
                payload = bytes;
                int scaleCount = Bfp8Quantization!.GetScaleCount(Numel);
                if (scales is null || scales.Length != scaleCount)
                    scales = new float[scaleCount];
                if (primary.Scales is null)
                    throw new InvalidOperationException("The primary Arc BFP8 parameter has no scales.");
                primary.Lane.ReadRaw(primary.Value, bytes);
                primary.Lane.Read(primary.Scales, scales);
            }
            else if (DType == TensorDType.BFloat16)
            {
                ushort[] bytes = payload as ushort[] ?? [];
                if (bytes.Length != Numel) bytes = new ushort[Numel];
                payload = bytes;
                primary.Lane.ReadRaw(primary.Value, bytes);
            }
            else
            {
                float[] values = payload as float[] ?? [];
                if (values.Length != Numel) values = new float[Numel];
                payload = values;
                primary.Lane.ReadRaw(primary.Value, values);
            }
        }

        using (TensorExecutionContext.Push(new TorchDevice(TensorDevice.Arc, targetDevice)))
        {
            ArcReplica replica = ArcOwner();
            if (replica.Lane.DeviceIndex != targetDevice)
                throw new InvalidOperationException("The secondary Arc parameter belongs to another device.");
            replica.MatrixPanels?.Clear();
            if (replica.Value is { } value && value.ByteLength == Math.Max(4, Buffer.ByteLength(payload)))
                replica.Lane.WriteRaw(value, payload);
            else
            {
                replica.Value?.Dispose();
                replica.Value = replica.Lane.UploadRaw(payload);
            }
            if (scales is not null)
            {
                if (replica.Scales is { } currentScales && currentScales.ByteLength == Math.Max(4, Buffer.ByteLength(scales)))
                    replica.Lane.Write(currentScales, scales);
                else
                {
                    replica.Scales?.Dispose();
                    replica.Scales = replica.Lane.Upload(scales);
                }
            }
            else
            {
                replica.Scales?.Dispose();
                replica.Scales = null;
            }
            replica.Master?.Dispose();
            replica.Master = null;
            replica.DataDirty = true;
            replica.MasterDirty = false;
            _masterData = null;
            _device = TensorDevice.Arc;
            _arcDeviceIndex = targetDevice;
            unchecked { _dataVersion++; }
            _physicalFloat32CacheDataVersion = -1;
            _transposedDataVersion = -1;
        }
    }
}
