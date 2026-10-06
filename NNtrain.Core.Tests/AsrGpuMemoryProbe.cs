using System.Runtime.InteropServices;

namespace NNtrain.Core.Tests;

/// <summary>Observes this test process's WDDM GPU allocations, including driver-owned memory.
/// Does not inspect other processes or alter device state.</summary>
internal sealed class AsrGpuMemoryProbe : IDisposable
{
    private readonly object _gate = new();
    private readonly CancellationTokenSource _cancellation = new();
    private readonly Task _task;
    private nint _query, _dedicated, _shared;
    internal long? DedicatedPeakBytes { get; private set; }
    internal long? SharedPeakBytes { get; private set; }
    internal int Samples { get; private set; }
    internal string? Error { get; private set; }

    internal AsrGpuMemoryProbe()
    {
        try
        {
            Check(PdhOpenQuery(null, 0, out _query));
            Check(PdhAddEnglishCounter(_query, @"\GPU Process Memory(*)\Dedicated Usage", 0, out _dedicated));
            Check(PdhAddEnglishCounter(_query, @"\GPU Process Memory(*)\Shared Usage", 0, out _shared));
        }
        catch (Exception ex) { Error = ex.Message; }
        _task = Task.Run(async () =>
        {
            while (!_cancellation.IsCancellationRequested && Error is null)
            {
                try
                {
                    Check(PdhCollectQueryData(_query));
                    long? dedicated = Read(_dedicated), shared = Read(_shared);
                    lock (_gate)
                    {
                        if (dedicated is not null)
                        {
                            DedicatedPeakBytes = Math.Max(DedicatedPeakBytes ?? 0, dedicated.Value);
                            SharedPeakBytes = Math.Max(SharedPeakBytes ?? 0, shared ?? 0);
                            Samples++;
                        }
                    }
                    await Task.Delay(100, _cancellation.Token);
                }
                catch (OperationCanceledException) { break; }
                catch (Exception ex) { Error = ex.Message; break; }
            }
        });
    }

    private static long? Read(nint counter)
    {
        uint bytes = 0;
        uint result = PdhGetFormattedCounterArray(counter, 0x200, ref bytes, out _, 0);
        if (result != 0x800007D2 && result != 0) Check(result);
        if (bytes == 0) return null;
        if (bytes > 16 * 1024 * 1024) throw new InvalidOperationException("Unexpected GPU counter array size.");
        nint buffer = Marshal.AllocHGlobal((int)bytes);
        try
        {
            Check(PdhGetFormattedCounterArray(counter, 0x200, ref bytes, out uint count, buffer));
            long total = 0; bool found = false;
            string prefix = $"pid_{Environment.ProcessId}_";
            for (int i = 0; i < count; i++)
            {
                var item = Marshal.PtrToStructure<CounterItem>(buffer + i * Marshal.SizeOf<CounterItem>());
                string? name = Marshal.PtrToStringUni(item.Name);
                if (name?.StartsWith(prefix, StringComparison.Ordinal) != true || item.Value.Status > 1) continue;
                total += checked((long)item.Value.Value); found = true;
            }
            return found ? total : null;
        }
        finally { Marshal.FreeHGlobal(buffer); }
    }

    public void Dispose()
    {
        _cancellation.Cancel(); _task.GetAwaiter().GetResult();
        if (_query != 0) { PdhCloseQuery(_query); _query = 0; }
        _cancellation.Dispose();
    }
    private static void Check(uint status)
    { if (status != 0) throw new InvalidOperationException($"GPU memory counter unavailable: PDH 0x{status:X8}"); }
    [StructLayout(LayoutKind.Explicit)] private struct CounterValue
    { [FieldOffset(0)] public uint Status; [FieldOffset(8)] public double Value; }
    [StructLayout(LayoutKind.Sequential)] private struct CounterItem
    { public nint Name; public CounterValue Value; }
    [DllImport("pdh.dll", EntryPoint = "PdhOpenQueryW", CharSet = CharSet.Unicode)] private static extern uint PdhOpenQuery(string? source, nuint userData, out nint query);
    [DllImport("pdh.dll", EntryPoint = "PdhAddEnglishCounterW", CharSet = CharSet.Unicode)] private static extern uint PdhAddEnglishCounter(nint query, string path, nuint userData, out nint counter);
    [DllImport("pdh.dll")] private static extern uint PdhCollectQueryData(nint query);
    [DllImport("pdh.dll", EntryPoint = "PdhGetFormattedCounterArrayW")] private static extern uint PdhGetFormattedCounterArray(nint counter, uint format, ref uint size, out uint count, nint buffer);
    [DllImport("pdh.dll")] private static extern uint PdhCloseQuery(nint query);
}
