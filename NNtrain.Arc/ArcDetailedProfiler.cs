namespace NNtrain.Arc;

/// <summary>Bounded aggregated telemetry; no tensor readback and no event per sample retained.</summary>
public sealed class ArcDetailedProfiler
{
    private readonly Dictionary<string, Counter> _counters = [];
    public string Phase { get; set; } = "unscoped";
    private sealed class Counter { internal long Count, Bytes; internal double Milliseconds; }
    internal void Add(string kind, string detail, double milliseconds = 0, long bytes = 0)
    {
        string key = kind + "|" + detail;
        // Diagnostic labels are shape-based, never buffer addresses or step IDs.
        if (!_counters.TryGetValue(key, out var value))
        {
            if (_counters.Count >= 8192) key = kind + "|other";
            if (!_counters.TryGetValue(key, out value)) _counters.Add(key, value = new());
        }
        value.Count++; value.Bytes += bytes; value.Milliseconds += milliseconds;
    }
    public IReadOnlyList<ArcProfileEntry> Snapshot() => _counters.Select(pair => {
        int separator = pair.Key.IndexOf('|');
        return new ArcProfileEntry(pair.Key[..separator], pair.Key[(separator + 1)..],
            pair.Value.Count, pair.Value.Bytes, pair.Value.Milliseconds);
    }).OrderByDescending(e => e.Milliseconds).ToArray();
    internal void Reset() => _counters.Clear();
    internal string KernelLabel(string name, object[] args)
    {
        string shape = name.StartsWith("gemm_xmx_bfp8_epilogue_", StringComparison.Ordinal) && args.Length >= 11
            ? $" M={args[6]} N={args[7]} K={args[8]} offset={args[10]}"
            : name.StartsWith("gemm_xmx_streamed_", StringComparison.Ordinal) && args.Length >= 13
            ? $" M={args[4]} N={args[5]} K={args[6]} offset={args[12]}"
            : name.StartsWith("gemm_", StringComparison.Ordinal) && args.Length >= 10
            ? $" M={args[5]} N={args[6]} K={args[7]} TA={args[8]} TB={args[9]}"
            : (name.StartsWith("attention_gemm", StringComparison.Ordinal) || name.StartsWith("attention_xmx", StringComparison.Ordinal)) && args.Length >= 6
                ? $" M={args[3]} N={args[4]} K={args[5]}"
                : name == "gradient_rows" && args.Length >= 8
                    ? $" rows={args[4]} width={args[5]} relu={args[6]} norm={args[7]}"
                : name == "gradient_rows_bias_bf16" && args.Length >= 6
                    ? $" rows={args[3]} width={args[4]} relu={args[5]}"
                : "";
        return Phase + "/" + name + shape;
    }
}

public sealed record ArcProfileEntry(string Kind, string Detail, long Count, long Bytes, double Milliseconds);
