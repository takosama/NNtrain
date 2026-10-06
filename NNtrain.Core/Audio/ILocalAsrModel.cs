namespace NNtrain.Audio;
public interface ILocalAsrModel : IDisposable
{
    long HostWeightBytes { get; }
    long PeakDeviceBufferBytes { get; }
    long ResidentDeviceBytes { get; }
    string ExecutionDevice { get; }
    void EnableArc(int deviceIndex, CancellationToken ct = default);
    ILocalAsrStream CreateStream();
}
public interface ILocalAsrStream
{
    long CacheBytes { get; }
    string Append(ReadOnlySpan<float> samples, bool final = false, Action<string>? partial = null, CancellationToken ct = default);
}

/// <summary>Stops optional whole-prefix previews while retaining audio for final recognition.</summary>
public interface ILocalAsrPreviewControl
{
    void RequestFinalization();
}
