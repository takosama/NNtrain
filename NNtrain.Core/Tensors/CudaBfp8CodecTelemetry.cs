namespace NNtrain;

/// <summary>Managed codec launches, not hardware instructions or graph replays.</summary>
internal readonly record struct CudaBfp8CodecTelemetrySnapshot(
    long EncodeBFloat16Launches, long EncodeBFloat16Elements,
    long DecodeBFloat16Launches, long DecodeBFloat16Elements)
{
    public static CudaBfp8CodecTelemetrySnapshot operator -(
        CudaBfp8CodecTelemetrySnapshot a, CudaBfp8CodecTelemetrySnapshot b)
        => new(a.EncodeBFloat16Launches - b.EncodeBFloat16Launches,
            a.EncodeBFloat16Elements - b.EncodeBFloat16Elements,
            a.DecodeBFloat16Launches - b.DecodeBFloat16Launches,
            a.DecodeBFloat16Elements - b.DecodeBFloat16Elements);
}

internal static class CudaBfp8CodecTelemetry
{
    private static long _encodes, _encodedElements, _decodes, _decodedElements;
    internal static CudaBfp8CodecTelemetrySnapshot Snapshot => new(
        Interlocked.Read(ref _encodes), Interlocked.Read(ref _encodedElements),
        Interlocked.Read(ref _decodes), Interlocked.Read(ref _decodedElements));
    internal static void Encoded(long elements)
    { Interlocked.Increment(ref _encodes); Interlocked.Add(ref _encodedElements, elements); }
    internal static void Decoded(long elements)
    { Interlocked.Increment(ref _decodes); Interlocked.Add(ref _decodedElements, elements); }
}
