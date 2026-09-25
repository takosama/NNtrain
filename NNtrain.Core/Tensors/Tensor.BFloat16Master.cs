namespace NNtrain;

public partial class Tensor
{
    internal int HostMasterByteLength => _bfloat16MasterData is not null
        ? checked(_bfloat16MasterData.Length * sizeof(ushort))
        : _masterData is not null ? checked(_masterData.Length * sizeof(float)) : 0;

    internal bool HasBFloat16HostMaster => _bfloat16MasterData?.Length == Numel;
    internal bool HasFloat32HostMaster => _masterData is not null;

    /// <summary>
    /// Replaces the temporary FP32 conversion/restore master with physical
    /// BF16 host authority. The BFP8 compute payload remains independent.
    /// </summary>
    internal void PackBFloat16MasterInPlace()
    {
        if (DType != TensorDType.Bfp8
            || Bfp8Quantization?.Granularity != Bfp8ScaleGranularity.Block)
            throw new InvalidOperationException("A mix8_16 master requires block BFP8 storage.");
        lock (_deviceSync)
        {
            if (_bfloat16MasterData is not null)
                return;
            float[] source = _masterData ?? _data.ToFloat32Array();
            var packed = new ushort[Numel];
            for (int index = 0; index < packed.Length; index++)
            {
                ushort bits = TensorStorageCodec.EncodeBFloat16(source[index]);
                packed[index] = bits;
                source[index] = TensorStorageCodec.DecodeBFloat16(bits);
            }
            // The first published BFP8 forward payload must come from the
            // same BF16 master that subsequent optimizer updates will use.
            _data.CopyFrom(source);
            _bfloat16MasterData = packed;
            _masterData = null;
            MarkDataMutated();
        }
    }

    private ushort[] GetOrCreateHostBFloat16Master()
    {
        if (_bfloat16MasterData is not null)
            return _bfloat16MasterData;
        PackBFloat16MasterInPlace();
        return _bfloat16MasterData!;
    }

    private void AdoptHostBFloat16Master(ushort[] packed)
    {
        ArgumentNullException.ThrowIfNull(packed);
        if (packed.Length != Numel)
            throw new ArgumentException("BF16 master length must match the tensor.", nameof(packed));
        _bfloat16MasterData = packed;
        _masterData = null;
    }

    private float[] DecodeHostBFloat16Master()
    {
        ushort[] packed = _bfloat16MasterData
            ?? throw new InvalidOperationException("The tensor has no BF16 host master.");
        var decoded = new float[packed.Length];
        for (int index = 0; index < packed.Length; index++)
            decoded[index] = TensorStorageCodec.DecodeBFloat16(packed[index]);
        return decoded;
    }

    /// <summary>
    /// Publishes the packed host master to BFP8 without materializing an
    /// additional full-size FP32 weight array during checkpoint restore.
    /// </summary>
    private void PublishBfp8FromHostBFloat16Master()
    {
        ushort[] packed = _bfloat16MasterData
            ?? throw new InvalidOperationException("The tensor has no BF16 host master.");
        Bfp8QuantizationDescriptor descriptor = Bfp8Quantization
            ?? throw new InvalidOperationException("The tensor has no BFP8 quantization descriptor.");
        int blockSize = descriptor.GetEffectiveBlockSize(Numel);
        var payload = new sbyte[Numel];
        var scales = new float[descriptor.GetScaleCount(Numel)];
        for (int block = 0; block < scales.Length; block++)
        {
            int start = checked(block * blockSize);
            int end = Math.Min(Numel, checked(start + blockSize));
            float maximum = 0f;
            for (int index = start; index < end; index++)
            {
                float value = TensorStorageCodec.DecodeBFloat16(packed[index]);
                if (!float.IsFinite(value))
                    throw new ArgumentException("BFP8 encoding requires finite input values.");
                maximum = MathF.Max(maximum, MathF.Abs(value));
            }
            float scale = maximum == 0f ? 1f : maximum / 127f;
            scales[block] = scale;
            for (int index = start; index < end; index++)
            {
                float value = TensorStorageCodec.DecodeBFloat16(packed[index]);
                float rounded = MathF.Round(value / scale, MidpointRounding.ToEven);
                payload[index] = (sbyte)Math.Clamp((int)rounded, -127, 127);
            }
        }
        _data.CopyFromBfp8Encoded(payload, scales);
    }
}
