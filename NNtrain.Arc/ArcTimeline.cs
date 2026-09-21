using System.Diagnostics;
using System.IO.Compression;
using System.Text.Json;

namespace NNtrain.Arc;

/// <summary>Opt-in single in-order-queue trace. Lane calls are serialized; host workers have separate tracks. No tensor readbacks.</summary>
public sealed class ArcTimeline
{
    private readonly List<HostInterval> _host = new(500_000);
    private readonly List<DeviceInterval> _device = new(400_000);
    private readonly System.Collections.Concurrent.ConcurrentDictionary<string, string> _labels = new(StringComparer.Ordinal);
    private int _sequence;
    internal ClockAnchor First { get; }
    internal ClockAnchor Last { get; private set; } = null!;
    public long StartTicks { get; private set; }
    public long EndTicks { get; private set; }
    public int MissingEvents { get; private set; }
    public int OpaqueAllocationCopies { get; internal set; }
    internal ArcTimeline(ClockAnchor first) => First = first;
    public void MarkStart() => StartTicks = Stopwatch.GetTimestamp();
    public void MarkEnd()
    {
        EndTicks = Stopwatch.GetTimestamp();
        // Explicit enclosing benchmark interval: any ticks between starting/
        // closing narrower scopes belong to trace/benchmark bookkeeping.
        lock (_host) _host.Add(new(0, StartTicks, EndTicks, "benchmark-boundary", "timestamp-and-scope-bookkeeping", Environment.CurrentManagedThreadId));
    }
    internal void Complete(ClockAnchor last) => Last = last;
    private string Label(string text)
    {
        return _labels.GetOrAdd(text, text);
    }
    public HostScope Host(string category, string detail = "")
    {
        return new(this, Interlocked.Increment(ref _sequence), Stopwatch.GetTimestamp(), Label(category), Label(detail));
    }
    public readonly struct HostScope : IDisposable
    {
        private readonly ArcTimeline? _owner;
        private readonly int _id;
        private readonly int _thread;
        private readonly long _start;
        private readonly string _category, _detail;
        internal HostScope(ArcTimeline owner, int id, long start, string category, string detail)
            => (_owner, _id, _start, _category, _detail, _thread) = (owner, id, start, category, detail, Environment.CurrentManagedThreadId);
        public void Dispose()
        {
            if (_owner is null) return;
            long end = Stopwatch.GetTimestamp();
            lock (_owner._host) _owner._host.Add(new(_id, _start, end, _category, _detail, _thread));
        }
    }
    internal void Device(string name, string? label, nint evt, string kind, long bytes)
    {
        if (evt == 0
            || OpenClNative.clGetEventProfilingInfo(evt, 0x1280, 8, out ulong queued, out _) != 0
            || OpenClNative.clGetEventProfilingInfo(evt, 0x1281, 8, out ulong submitted, out _) != 0
            || OpenClNative.clGetEventProfilingInfo(evt, 0x1282, 8, out ulong start, out _) != 0
            || OpenClNative.clGetEventProfilingInfo(evt, 0x1283, 8, out ulong end, out _) != 0)
        { MissingEvents++; return; }
        _device.Add(new(Label(name), Label(label ?? name), kind, bytes, queued, submitted, start, end));
    }
    internal sealed record ClockAnchor(ulong DeviceNs, ulong OpenClHostNs, long QpcTicks, long BracketTicks);
    private readonly record struct HostInterval(int Id, long Start, long End, string Category, string Detail, int Thread);
    private readonly record struct DeviceInterval(string Name, string Label, string Kind, long Bytes,
        ulong Queued, ulong Submitted, ulong Start, ulong End);
    private double HostNs(long ticks) => (ticks - StartTicks) * (1e9 / Stopwatch.Frequency);
    private double DeviceNs(ulong ns)
    {
        double slope = (Last.QpcTicks - First.QpcTicks) * (1e9 / Stopwatch.Frequency)
            / (Last.DeviceNs - First.DeviceNs);
        return HostNs(First.QpcTicks) + ((double)ns - First.DeviceNs) * slope;
    }
    public ArcTimelineReport Export(string path)
    {
        if (StartTicks == 0 || EndTicks <= StartTicks || Last is null)
            throw new InvalidOperationException("Timeline must have a completed, calibrated wall interval.");
        var host = _host.Select(h => new ArcTimelineInterval(HostNs(h.Start), HostNs(h.End), h.Category, h.Detail, h.Id)).ToArray();
        var gpu = _device.Select(d => new ArcTimelineInterval(DeviceNs(d.Start), DeviceNs(d.End),
            d.Kind == "kernel" ? KernelCategory(d.Name) : "GPU/transfer/" + d.Kind, d.Label, 0)).ToArray();
        double duration = HostNs(EndTicks);
        var partition = ArcTimelinePartition.Build(duration, host, gpu);
        double queuedIdle = IdleWithPending(gpu, _device.Select(d => (DeviceNs(d.Queued), DeviceNs(d.Start))).ToArray(), duration);
        double scale = (Last.QpcTicks - First.QpcTicks) * (1e9 / Stopwatch.Frequency) / (Last.DeviceNs - First.DeviceNs);
        var report = new ArcTimelineReport(duration / 1e6, _host.Count, _device.Count, MissingEvents,
            OpaqueAllocationCopies, Math.Max(First.BracketTicks, Last.BracketTicks) * 500d / Stopwatch.Frequency,
            (scale - 1) * 1e6, queuedIdle / 1e6, partition, path);
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        using var file = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.Read);
        using var zip = new GZipStream(file, CompressionLevel.Fastest);
        using var writer = new Utf8JsonWriter(zip);
        writer.WriteStartObject(); writer.WritePropertyName("metadata"); JsonSerializer.Serialize(writer, report);
        writer.WritePropertyName("clockCalibration"); JsonSerializer.Serialize(writer, new { First, Last, Stopwatch.Frequency, StartTicks, EndTicks });
        writer.WritePropertyName("traceEvents"); writer.WriteStartArray();
        for (int i = 0; i < host.Length; i++) Write(host[i], _host[i].Thread, null);
        for (int i = 0; i < gpu.Length; i++) Write(gpu[i], 0, _device[i]);
        writer.WriteEndArray(); writer.WriteEndObject();
        return report;

        void Write(ArcTimelineInterval interval, int track, DeviceInterval? evt)
        {
            writer.WriteStartObject(); writer.WriteString("ph", "X"); writer.WriteNumber("pid", 1); writer.WriteNumber("tid", track);
            writer.WriteString("name", interval.Detail.Length > 0 ? interval.Detail : interval.Category);
            writer.WriteString("cat", interval.Category); writer.WriteNumber("ts", interval.StartNs / 1000);
            writer.WriteNumber("dur", (interval.EndNs - interval.StartNs) / 1000);
            if (evt is { } d)
            {
                writer.WritePropertyName("args"); writer.WriteStartObject(); writer.WriteNumber("bytes", d.Bytes);
                writer.WriteNumber("queued_us", DeviceNs(d.Queued) / 1000);
                writer.WriteNumber("submitted_us", DeviceNs(d.Submitted) / 1000);
                writer.WriteNumber("queued_device_ns", d.Queued); writer.WriteNumber("submitted_device_ns", d.Submitted);
                writer.WriteNumber("start_device_ns", d.Start); writer.WriteNumber("end_device_ns", d.End);
                writer.WriteEndObject();
            }
            writer.WriteEndObject();
        }
    }
    private static string KernelCategory(string name)
    {
        // Packed COMPUTE (norm, gradient, ReLU) is not a format conversion.
        if (name.StartsWith("xmx_storage_pack", StringComparison.Ordinal) || name.StartsWith("decode_", StringComparison.Ordinal)
            || name.Contains("_pack_", StringComparison.Ordinal) || name.StartsWith("resident_bfp8", StringComparison.Ordinal)
            || name.StartsWith("resident_bf16", StringComparison.Ordinal) || name.StartsWith("round_bf16", StringComparison.Ordinal)) return "GPU/pack-decode-publication";
        if (name.StartsWith("attention", StringComparison.Ordinal)) return "GPU/attention";
        if (name.StartsWith("gemm", StringComparison.Ordinal)) return "GPU/GEMM";
        if (name.Contains("norm", StringComparison.Ordinal) || name.StartsWith("gradient_rows", StringComparison.Ordinal)) return "GPU/normalization-reduction";
        if (name.Contains("loss", StringComparison.Ordinal) || name.Contains("cross_entropy", StringComparison.Ordinal)) return "GPU/loss";
        return "GPU/other-kernels";
    }
    // Union of queued-to-start intervals intersected with the complement of GPU execution.
    private static double IdleWithPending(ArcTimelineInterval[] gpu, (double Start, double End)[] pending, double duration)
    {
        var edges = new List<(double T, int G, int Q)>();
        foreach (var e in gpu) { edges.Add((Math.Clamp(e.StartNs, 0, duration), 1, 0)); edges.Add((Math.Clamp(e.EndNs, 0, duration), -1, 0)); }
        foreach (var e in pending) { edges.Add((Math.Clamp(e.Start, 0, duration), 0, 1)); edges.Add((Math.Clamp(e.End, 0, duration), 0, -1)); }
        edges.Sort((a, b) => a.T.CompareTo(b.T));
        double last = 0, result = 0; int activeGpu = 0, activeQueue = 0;
        foreach (var edge in edges) { if (activeGpu == 0 && activeQueue > 0) result += edge.T - last; activeGpu += edge.G; activeQueue += edge.Q; last = edge.T; }
        return result;
    }
}

public readonly record struct ArcTimelineInterval(double StartNs, double EndNs, string Category, string Detail, int NestingOrder);
public sealed record ArcTimelineShare(string Name, double Milliseconds, double Fraction);
public sealed record ArcTimelinePartitionResult(ArcTimelineShare[] WallCategories, ArcTimelineShare[] WallOperations,
    ArcTimelineShare[] HostExclusive, double CoverageFraction, double GpuOverlapMs, double UnattributedHostMs);
public sealed record ArcTimelineReport(double WallMs, int HostSpans, int DeviceEvents, int MissingEvents,
    int OpaqueAllocationCopies, double ClockBracketUncertaintyMs, double ClockDriftPpm, double GpuIdleWithQueuedCommandsMs,
    ArcTimelinePartitionResult Partition, string TracePath);

/// <summary>GPU execution takes priority; only its complement is attributed to the deepest host scope.
/// HostExclusive is a separate view of the SAME wall time, never added to WallCategories.</summary>
public static class ArcTimelinePartition
{
    public static ArcTimelinePartitionResult Build(double durationNs, IReadOnlyList<ArcTimelineInterval> host, IReadOnlyList<ArcTimelineInterval> gpu)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(durationNs);
        var edges = new List<(double T, int Id, bool Gpu, bool Start)>((host.Count + gpu.Count) * 2);
        void Add(IReadOnlyList<ArcTimelineInterval> source, bool device)
        {
            for (int i = 0; i < source.Count; i++)
            {
                var e = source[i];
                if (e.EndNs < e.StartNs) throw new InvalidOperationException("Negative timeline interval.");
                double a = Math.Clamp(e.StartNs, 0, durationNs), b = Math.Clamp(e.EndNs, 0, durationNs);
                if (b <= a) continue;
                edges.Add((a, i, device, true)); edges.Add((b, i, device, false));
            }
        }
        Add(host, false); Add(gpu, true);
        edges.Add((durationNs, -1, false, false));
        edges.Sort((a, b) => a.T.CompareTo(b.T));
        var activeHost = new SortedSet<(int Order, int Id)>(); var activeGpu = new SortedSet<int>();
        var categories = new Dictionary<string, double>(); var operations = new Dictionary<string, double>(); var hostOnly = new Dictionary<string, double>();
        double last = 0, overlap = 0, unknown = 0;
        static void Sum(Dictionary<string, double> values, string label, double ns) => values[label] = values.GetValueOrDefault(label) + ns;
        foreach (var edge in edges)
        {
            double ns = edge.T - last;
            if (ns > 0)
            {
                var h = activeHost.Count > 0 ? host[activeHost.Max.Id] : new(0, 0, "host/unattributed", "outside-host-scope", 0);
                Sum(hostOnly, h.Category, ns);
                if (activeHost.Count == 0) unknown += ns;
                if (activeGpu.Count > 0)
                {
                    var g = gpu[activeGpu.Min]; Sum(categories, g.Category, ns); Sum(operations, g.Detail, ns);
                    if (activeGpu.Count > 1) overlap += ns;
                }
                else { Sum(categories, "GPU-idle/" + h.Category, ns); Sum(operations, "GPU-idle/" + h.Category + "/" + h.Detail, ns); }
            }
            last = edge.T;
            if (edge.Id < 0) continue;
            if (edge.Gpu) { if (edge.Start) activeGpu.Add(edge.Id); else activeGpu.Remove(edge.Id); }
            else { var key = (host[edge.Id].NestingOrder, edge.Id); if (edge.Start) activeHost.Add(key); else activeHost.Remove(key); }
        }
        ArcTimelineShare[] Shares(Dictionary<string, double> values) => values.OrderByDescending(p => p.Value)
            .Select(p => new ArcTimelineShare(p.Key, p.Value / 1e6, p.Value / durationNs)).ToArray();
        return new(Shares(categories), Shares(operations), Shares(hostOnly), categories.Values.Sum() / durationNs, overlap / 1e6, unknown / 1e6);
    }
}
