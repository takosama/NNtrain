using System.Diagnostics;

namespace NNtrain.Arc;

public sealed partial class ArcExecutionLane
{
    public ArcTimeline? Timeline { get; private set; }
    public ArcTimeline BeginTimeline()
    {
        lock (_sync)
        {
            if (Timeline is not null) throw new InvalidOperationException("Timeline already active.");
            Synchronize();
            return Timeline = new(CalibrateClock());
        }
    }
    public ArcTimeline EndTimeline()
    {
        lock (_sync)
        {
            var timeline = Timeline ?? throw new InvalidOperationException("No timeline active.");
            if (_pendingEvents.Count != 0 || _queuedCopies) throw new InvalidOperationException("Finish queue before ending timeline.");
            timeline.Complete(CalibrateClock()); Timeline = null; return timeline;
        }
    }
    private ArcTimeline.ClockAnchor CalibrateClock()
    {
        ArcTimeline.ClockAnchor? best = null;
        for (int i = 0; i < 9; i++)
        {
            long a = Stopwatch.GetTimestamp();
            OpenClNative.Check(OpenClNative.clGetDeviceAndHostTimer(Device.NativeDevice, out ulong device, out ulong host), "correlate device/host clocks (required for timeline)");
            long b = Stopwatch.GetTimestamp();
            if (best is null || b - a < best.BracketTicks) best = new(device, host, a + (b - a) / 2, b - a);
        }
        return best!;
    }
}
