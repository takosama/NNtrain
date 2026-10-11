using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

public sealed partial class Qwen35QuantizedModel
{
    internal static int Iq2PackedDecodeBytes(int inputWidth, int outputWidth)
    {
        if (inputWidth <= 0 || inputWidth % 256 != 0)
            throw new ArgumentOutOfRangeException(nameof(inputWidth), "IQ2_S rows require a positive multiple of 256 weights.");
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(outputWidth);
        return checked(inputWidth / 256 * outputWidth * 82);
    }

    /// <summary>
    /// Rearrange original GGUF bytes without quantization or decoded weights.
    /// The returned single allocation is the same size as the original tensor.
    /// </summary>
    internal static byte[] PackIq2DecodeBytes(ReadOnlySpan<byte> raw, int inputWidth, int outputWidth)
    {
        int bytes = Iq2PackedDecodeBytes(inputWidth, outputWidth);
        if (raw.Length != bytes) throw new ArgumentException("IQ2_S tensor byte length does not match its shape.", nameof(raw));
        byte[] packed = GC.AllocateUninitializedArray<byte>(bytes);
        int blocks = bytes / 82, extras = checked(blocks * 64), ds = checked(blocks * 80);
        for (int block = 0; block < blocks; block++)
        {
            int source = block * 82, codeOffset = block * 64, extraOffset = extras + block * 16;
            for (int pair = 0; pair < 16; pair++)
            {
                int octet = pair * 2, group = pair >> 1, shift = (pair & 1) * 4, code = codeOffset + pair * 4;
                packed[code] = raw[source + 2 + octet];
                packed[code + 1] = raw[source + 34 + octet];
                packed[code + 2] = raw[source + 3 + octet];
                packed[code + 3] = raw[source + 35 + octet];
                packed[extraOffset + pair] = (byte)(((raw[source + 66 + group] >> shift) & 15)
                    | (((raw[source + 74 + group] >> shift) & 15) << 4));
            }
            packed[ds + block * 2] = raw[source];
            packed[ds + block * 2 + 1] = raw[source + 1];
        }
        return packed;
    }

    /// <summary>Restore every source byte, including signed zero/NaN half encodings.</summary>
    internal static byte[] UnpackIq2DecodeBytes(ReadOnlySpan<byte> packed, int inputWidth, int outputWidth)
    {
        int bytes = Iq2PackedDecodeBytes(inputWidth, outputWidth);
        if (packed.Length != bytes) throw new ArgumentException("Packed IQ2_S byte length does not match its shape.", nameof(packed));
        byte[] raw = GC.AllocateUninitializedArray<byte>(bytes);
        int blocks = bytes / 82, extras = checked(blocks * 64), ds = checked(blocks * 80);
        for (int block = 0; block < blocks; block++)
        {
            int target = block * 82, codeOffset = block * 64, extraOffset = extras + block * 16;
            for (int octet = 0; octet < 32; octet++)
            {
                raw[target + 2 + octet] = packed[codeOffset + octet * 2];
                raw[target + 34 + octet] = packed[codeOffset + octet * 2 + 1];
            }
            for (int group = 0; group < 8; group++)
            {
                int low = packed[extraOffset + group * 2], high = packed[extraOffset + group * 2 + 1];
                raw[target + 66 + group] = (byte)((low & 15) | ((high & 15) << 4));
                raw[target + 74 + group] = (byte)((low >> 4) | (high & 240));
            }
            raw[target] = packed[ds + block * 2];
            raw[target + 1] = packed[ds + block * 2 + 1];
        }
        return raw;
    }

    internal static bool SupportsIq2PackedDecode(ArcExecutionLane lane)
        => lane.Options.Qwen35InferenceKernelsOnly && !lane.Options.Qwen35VisionKernelsOnly
            && !lane.Options.AsrKernelsOnly && !lane.Options.Qwen35TrainingKernels
            && lane.Options.XmxMatrices && lane.Device.SupportsXmx && lane.Device.MinimumSubgroupSize == 16
            && lane.Device.Extensions.Split(' ').Contains("cl_intel_subgroups");

    /// <summary>
    /// Allocate and enqueue a same-size device repack. The caller owns the
    /// returned buffer and must retain the source until the queue completes.
    /// No hidden synchronization, permanent duplication or weight expansion.
    /// </summary>
    internal static ArcBuffer PackIq2Decode(ArcExecutionLane lane, ArcBuffer raw, int inputWidth, int outputWidth)
    {
        ArgumentNullException.ThrowIfNull(lane);
        ArgumentNullException.ThrowIfNull(raw);
        if (!SupportsIq2PackedDecode(lane)) throw new NotSupportedException("Packed IQ2_S decode requires an inference SG16 lane.");
        int bytes = Iq2PackedDecodeBytes(inputWidth, outputWidth);
        if (raw.ByteLength != bytes) throw new ArgumentException("IQ2_S tensor byte length does not match its shape.", nameof(raw));
        ArcBuffer packed = lane.AllocateBytes(bytes);
        try
        {
            int blocks = bytes / 82;
            lane.Run("q35l_iq2_s_pack82", (long)blocks * 16, 0, raw, packed, blocks);
            return packed;
        }
        catch { packed.Dispose(); throw; }
    }

    internal static void RunIq2PackedDecode(ArcExecutionLane lane, ArcBuffer input, ArcBuffer packed,
        ArcBuffer bias, ArcBuffer output, int rows, int inputWidth, int outputWidth, bool paired)
    {
        ArgumentNullException.ThrowIfNull(lane);
        if (!SupportsIq2PackedDecode(lane)) throw new NotSupportedException("Packed IQ2_S decode requires an inference SG16 lane.");
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        int bytes = Iq2PackedDecodeBytes(inputWidth, outputWidth);
        int outputElements = checked(rows * outputWidth);
        if (packed.ByteLength != bytes || input.ByteLength < checked(4L * rows * inputWidth)
            || bias.ByteLength < checked(4L * outputWidth) || output.ByteLength < checked(4L * outputElements))
            throw new ArgumentException("Packed IQ2_S projection buffer shape mismatch.");
        int group = paired ? 32 : lane.Options.Qwen35ProjectionWorkgroupSize;
        long workItems = paired ? ((long)rows * ((outputWidth + 1) / 2) + 1) / 2 * 32
            : ((long)outputElements + group / 16 - 1) / (group / 16) * group;
        lane.Run(paired ? "q35l_iq2_s_packed82_sg16_pair" : "q35l_iq2_s_packed82_sg16",
            workItems, group, input, packed, bias, output, rows, inputWidth, outputWidth);
    }

    internal static ArcBuffer UnpackIq2Decode(ArcExecutionLane lane, ArcBuffer packed, int inputWidth, int outputWidth)
    {
        int bytes = Iq2PackedDecodeBytes(inputWidth, outputWidth);
        if (packed.ByteLength != bytes) throw new ArgumentException("Packed IQ2_S tensor byte length does not match its shape.", nameof(packed));
        ArcBuffer raw = lane.AllocateBytes(bytes);
        try
        {
            int blocks = bytes / 82;
            lane.Run("q35l_iq2_s_unpack82", (long)blocks * 8, 0, packed, raw, blocks);
            return raw;
        }
        catch { raw.Dispose(); throw; }
    }
}
