using System.Diagnostics;

namespace NNtrain.Arc;

public sealed partial class ArcExecutionLane
{
    public void ResetDetailedProfile()
    {
        lock (_sync)
        {
            if (_pendingEvents.Count != 0 || _queuedCopies) throw new InvalidOperationException("Synchronize before resetting the Arc profile.");
            DetailedProfiler?.Reset();
        }
    }

    /// <summary>Finish queued work and collect profiling events without transferring tensor data.</summary>
    public void Synchronize() => SynchronizeCore("explicit");

    // Keep the newer half executing while CPU queries/releases older events.
    // This is NOT a retirement fence: retired allocations and D2D events stay
    // owned until the existing full-queue synchronization / memory budget fence.
    private unsafe void DrainOldestEventHalf()
    {
        int count = _pendingEvents.Count / 2;
        nint last = _pendingEvents[count - 1].Event;
        if (last == 0) throw new InvalidOperationException("The queued kernel window has no completion event.");
        long start = Stopwatch.GetTimestamp();
        using (Timeline?.Host("queue-partial-wait", "event-half"))
        {
            OpenClNative.Check(OpenClNative.clFlush(_queue), "flush event window");
            OpenClNative.Check(OpenClNative.clWaitForEvents(1, (nint)(&last)), "wait oldest event window");
        }
        DetailedProfiler?.Add("partial-wait", DetailedProfiler.Phase, Stopwatch.GetElapsedTime(start).TotalMilliseconds);
        using (Timeline?.Host("profiler-event-collection", "completed-half"))
        {
            // Remove each owner even if profiling throws; the caller's failure
            // cleanup must never release the same native event twice.
            int removed = 0;
            try
            {
                for (; removed < count;)
                {
                    var entry = _pendingEvents[removed++];
                    try { RecordKernelTime(entry.Name, entry.Event, entry.Label); }
                    finally { if (entry.Event != 0) OpenClNative.clReleaseEvent(entry.Event); }
                }
            }
            finally { _pendingEvents.RemoveRange(0, removed); }
        }
    }

    private void SynchronizeCore(string reason)
    {
        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            long start = Stopwatch.GetTimestamp();
            try { using var wait = Timeline?.Host("queue-sync-wait", reason); OpenClNative.Check(OpenClNative.clFinish(_queue), "synchronize"); }
            finally {
                DetailedProfiler?.Add("synchronize", $"{DetailedProfiler.Phase}/{reason}", Stopwatch.GetElapsedTime(start).TotalMilliseconds);
                DrainCompletedEvents();
            }
        }
    }

    private void DrainCompletedEvents()
    {
        using var collection = Timeline?.Host("profiler-event-collection");
        foreach (var entry in _pendingEvents)
        {
            try { RecordKernelTime(entry.Name, entry.Event, entry.Label); }
            finally { if (entry.Event != 0) OpenClNative.clReleaseEvent(entry.Event); }
        }
        _pendingEvents.Clear();
        foreach (var entry in _pendingCopyEvents)
        {
            try { Timeline?.Device(entry.Kind, entry.Kind, entry.Event, entry.Kind, entry.Bytes); }
            finally { OpenClNative.clReleaseEvent(entry.Event); }
        }
        _pendingCopyEvents.Clear();
        _queuedCopies = false;
        foreach (var entry in _retired) ReleaseNative(entry.Handle, entry.Bytes);
        _retired.Clear(); RetiredBytes = 0;
    }

    private void RecordKernelTime(string name, nint evt, string? label)
    {
        if (evt == 0 || !CollectKernelEvents) return;
        Timeline?.Device(name, label, evt, "kernel", 0);
        if (OpenClNative.clGetEventProfilingInfo(evt, 0x1282, 8, out ulong start, out _) != 0
            || OpenClNative.clGetEventProfilingInfo(evt, 0x1283, 8, out ulong end, out _) != 0) return;
        double ms = (end - start) * 1e-6;
        KernelMilliseconds += ms;
        KernelTimings[name] = KernelTimings.GetValueOrDefault(name) + ms;
        if (label is not null)
        {
            DetailedProfiler!.Add("gpu-kernel", label, ms);
            if (OpenClNative.clGetEventProfilingInfo(evt, 0x1281, 8, out ulong submitted, out _) == 0 && start >= submitted)
                DetailedProfiler.Add("queue-delay-overlapping", label, (start - submitted) * 1e-6);
        }
    }
}
