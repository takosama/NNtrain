using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

/// <summary>
/// Borrowed resident matrix storage. Slicing preserves the original BFP8 scale
/// origin. There is no persistent panel cache or hidden host materialization.
/// </summary>
internal sealed class ArcXmxStorageOperand : IDisposable
{
    internal ArcExecutionLane Lane { get; }
    internal ArcBuffer Value { get; }
    internal ArcBuffer? Scales { get; }
    internal TensorDType DType { get; }
    internal int Numel { get; }
    internal int BlockSize { get; }
    internal int Offset { get; }
    private readonly ArcBuffer _sourceValue;
    private readonly ArcBuffer? _sourceScales;
    private bool _disposed;

    internal ArcXmxStorageOperand(ArcExecutionLane lane, ArcBuffer value, ArcBuffer? scales,
        TensorDType dtype, int numel, int blockSize = 1, int offset = 0)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(numel);
        ArgumentOutOfRangeException.ThrowIfNegative(offset);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(blockSize);
        if (dtype is not (TensorDType.Float32 or TensorDType.BFloat16 or TensorDType.Bfp8))
            throw new NotSupportedException($"Unsupported Arc XMX storage {dtype}.");
        if (dtype == TensorDType.Bfp8 && scales is null) throw new ArgumentNullException(nameof(scales));
        (Lane, DType, Numel, BlockSize, Offset) = (lane, dtype, numel, blockSize, offset);
        (_sourceValue, _sourceScales) = (value, scales);
        Value = value.Borrow();
        try { Scales = scales?.Borrow(); }
        catch { Value.Dispose(); throw; }
    }

    internal ArcXmxStorageOperand Slice(int offset, int length)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentOutOfRangeException.ThrowIfNegative(offset);
        ArgumentOutOfRangeException.ThrowIfNegative(length);
        if ((long)offset + length > Numel) throw new ArgumentOutOfRangeException(nameof(length));
        return new(Lane, _sourceValue, _sourceScales, DType, length, BlockSize, checked(Offset + offset));
    }

    internal ArcBuffer PackA(int m, int k, bool transpose, ArcBuffer? gate = null, int gateOffset = 0)
        => Pack(m, k, transpose, false, gate, gateOffset);

    internal ArcBuffer PackB(int n, int k, bool transpose, ArcBuffer? gate = null, int gateOffset = 0)
        => Pack(n, k, transpose, true, gate, gateOffset);

    private ArcBuffer Pack(int outer, int k, bool transpose, bool right, ArcBuffer? gate, int gateOffset)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(outer);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(k);
        ArgumentOutOfRangeException.ThrowIfNegative(gateOffset);
        if (checked((long)outer * k) != Numel) throw new ArgumentException("Matrix view length does not match its shape.");
        int panels = checked((int)(((outer + (right ? 15L : 7L)) / (right ? 16 : 8)) * ((k + 15L) / 16) * 128));
        ArcBuffer result = Lane.AllocateBytes(checked(panels * (right ? 4 : 2)));
        try
        {
            string suffix = DType switch { TensorDType.Float32 => "f32", TensorDType.BFloat16 => "bf16", _ => "bfp8" };
            if (!right && !transpose)
            {
                Lane.Run($"xmx_storage_pack_a_linear_vec4_{suffix}", panels / 4, 0,
                    Value, Scales ?? Value, gate ?? Value, result, outer, k, 0, BlockSize,
                    gate is null ? 0 : 1, Offset, gateOffset);
                return result;
            }
            if (transpose)
                Lane.Run2D($"xmx_storage_pack_{(right ? 'b' : 'a')}_transpose_{suffix}",
                    ((outer + 31L) / 32) * 256, (k + 31L) / 32, 256, 1,
                    Value, Scales ?? Value, gate ?? Value, result, outer, k, 1, BlockSize,
                    gate is null ? 0 : 1, Offset, gateOffset);
            else
                Lane.Run($"xmx_storage_pack_{(right ? 'b' : 'a')}_{suffix}", panels, 0,
                    Value, Scales ?? Value, gate ?? Value, result, outer, k, 0, BlockSize,
                    gate is null ? 0 : 1, Offset, gateOffset);
            return result;
        }
        catch { result.Dispose(); throw; }
    }

    internal static bool CanRun(ArcExecutionLane lane, int m, int n, int k)
    {
        if (!lane.Options.DirectXmxMatrices || !lane.Options.XmxMatrices || !lane.Device.SupportsXmx
            || lane.Device.MinimumSubgroupSize != 16 || m < 128 || n < 64 || k < 64) return false;
        long aCount = ((m + 7L) / 8) * ((k + 15L) / 16) * 128;
        long bCount = ((n + 15L) / 16) * ((k + 15L) / 16) * 128;
        const long budget = 96L * 1024 * 1024;
        return aCount <= budget / 2 && bCount <= budget / 4 && aCount * 2 + bCount * 4 <= budget;
    }

    /// <summary>BF16 operand GEMM only; callers retain their FP32 fallback policy.</summary>
    internal static bool TryGemm(ArcExecutionLane lane, ArcXmxStorageOperand a, ArcXmxStorageOperand b,
        ArcBuffer output, int m, int n, int k, bool ta = false, bool tb = false, bool accumulate = false,
        ArcBuffer? bias = null, bool relu = false, ArcBuffer? gate = null, int gateOperand = 0, int gateOffset = 0)
    {
        if (!ReferenceEquals(lane, a.Lane) || !ReferenceEquals(lane, b.Lane))
            throw new ArgumentException("Matrix operands must belong to the executing Arc lane.");
        if (gateOperand is < 0 or > 2 || (gateOperand != 0 && gate is null)) throw new ArgumentException("Invalid matrix gate.");
        if (a.Numel != (long)m * k || b.Numel != (long)n * k) throw new ArgumentException("Matrix operands do not match GEMM dimensions.");
        if (!CanRun(lane, m, n, k)) return false;
        using var packedA = a.PackA(m, k, ta, gateOperand == 1 ? gate : null, gateOffset);
        using var packedB = b.PackB(n, k, tb, gateOperand == 2 ? gate : null, gateOffset);
        GemmPanels(lane, packedA, packedB, output, m, n, k, ta, tb, accumulate, bias, relu);
        return true;
    }

    /// <summary>
    /// Execute already-packed operands produced by PackA/PackB. The caller owns
    /// their lifetime, enabling reuse inside one forward/backward phase without
    /// a persistent cache or a weight-generation invalidation protocol.
    /// </summary>
    internal static void GemmPanels(ArcExecutionLane lane, ArcBuffer packedA, ArcBuffer packedB,
        ArcBuffer output, int m, int n, int k, bool ta = false, bool tb = false, bool accumulate = false,
        ArcBuffer? bias = null, bool relu = false)
    {
        if (!CanRun(lane, m, n, k)) throw new ArgumentException("Packed GEMM shape or backend is unsupported.");
        bool longReduction = k >= 4096 && !ta;
        string kernel = longReduction ? "gemm_xmx_direct_block_8x32_wg16" : "gemm_xmx_direct_block_16x32_wg16";
        int tileRows = longReduction ? 128 : 256;
        long gx = ((n + 31L) / 32) * 16, gy = ((m + tileRows - 1L) / tileRows) * 16;
        int slices = (k + 2047) / 2048;
        if (lane.Options.ParallelWeightGradients && ta && !tb && accumulate && bias is null && !relu
            && k >= 4096 && (long)m * n * slices * 4 <= 64 * 1024 * 1024)
        {
            using var partials = lane.Allocate(checked(m * n * slices));
            lane.Run3D(kernel, gx, gy, slices, 16, 16, 1, packedA, packedB, partials, packedA, packedA,
                m, n, k, ta ? 1 : 0, tb ? 1 : 0, 3, 0, 0, 0, 0, 0, 2048);
            lane.Run("gemm_split_finish", checked(m * n), 0, partials, output, checked(m * n), slices);
            return;
        }
        for (int start = 0; start < k; start += 2048)
            lane.Run2D(kernel, gx, gy, 16, 16, packedA, packedB, output, bias ?? packedA, packedA,
                m, n, k, ta ? 1 : 0, tb ? 1 : 0, 3, accumulate || start != 0 ? 1 : 0,
                bias is not null && start == 0 ? 1 : 0, relu && start + 2048 >= k ? 1 : 0,
                0, start, Math.Min(2048, k - start));
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        try { Value.Dispose(); } finally { Scales?.Dispose(); }
    }
}
