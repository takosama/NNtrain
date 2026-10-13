using System.Runtime.InteropServices;
using System.Diagnostics;

namespace NNtrain.Arc;

public sealed partial class ArcExecutionLane
{

    /// <summary>Driver-reported resource usage; unsupported queries remain null, not zero.</summary>
    public ArcKernelResources GetKernelResources(string name)
    {
        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            nint kernel = GetOrCreateKernel(name, profileCompilation: false);
            ulong? Info(uint field) => OpenClNative.clGetKernelWorkGroupInfo(kernel, Device.NativeDevice,
                field, 8, out ulong value, out _) == 0 ? value : null;
            return new(name, Info(0x11b0), Info(0x11b2), Info(0x11b4), Info(0x4109));
        }
    }

    public static HostArray In(Array values) => new(values, false);
    public static HostArray InOut(float[] values) => new(values, true);
    public static HostArray Out(float[] values) => new(values, true, false);
    public sealed record HostArray(Array Values, bool ReadBack, bool Upload = true);
    public readonly record struct LocalMemory(int Bytes);

    public void Run(string name, long workItems, int localSize, params object[] arguments)
    {
        if (workItems <= 0) return;
        long count = localSize == 0 ? workItems : checked((workItems + localSize - 1) / localSize * localSize);
        RunDimensions(name, [checked((nuint)count)], localSize == 0 ? null : [(nuint)localSize], arguments);
    }

    public void Run2D(string name, long globalX, long globalY, int localX, int localY, params object[] arguments)
        => RunDimensions(name, [checked((nuint)globalX), checked((nuint)globalY)], [(nuint)localX, (nuint)localY], arguments);

    public void Run3D(string name, long globalX, long globalY, long globalZ, int localX, int localY, int localZ, params object[] arguments)
        => RunDimensions(name, [checked((nuint)globalX), checked((nuint)globalY), checked((nuint)globalZ)], [(nuint)localX, (nuint)localY, (nuint)localZ], arguments);

    private unsafe void RunDimensions(string name, nuint[] global, nuint[]? localSize, object[] arguments)
    {
        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            nint kernel = GetOrCreateKernel(name, profileCompilation: true);
            List<(ArcBuffer Buffer, HostArray Array)>? temporary = null;
            CachedKernelArguments? argumentCache = GetKernelArgumentCache(kernel, arguments.Length);
            nint kernelEvent = 0;
            long submitStart = Stopwatch.GetTimestamp();
            string? label = DetailedProfiler?.KernelLabel(name, arguments);
            try
            {
                using (Timeline?.Host("queue-submit", label ?? name))
                {
                for (int i = 0; i < arguments.Length; i++)
                {
                    object argument = arguments[i];
                    if (argument is HostArray host)
                    {
                        if (host.Values is not float[] and not int[])
                            throw new ArgumentException("Only float[] and int[] staging arrays are supported.");
                        var allocation = AllocateCore(Math.Max(4, Buffer.ByteLength(host.Values)), host.Upload ? host.Values : null);
                        (temporary ??= []).Add((allocation, host));
                        argument = allocation;
                    }
                    int status;
                    switch (argument)
                    {
                        case ArcBuffer buffer:
                            if (buffer.Owner != this || buffer.Handle == 0) throw new ArgumentException("Arc buffer belongs to a different or disposed lane.");
                            nint handle = buffer.Handle;
                            status = SetKernelArgument(kernel, (uint)i, (nuint)sizeof(nint), (nint)(&handle), argumentCache); break;
                        case int value: status = SetKernelArgument(kernel, (uint)i, 4, (nint)(&value), argumentCache); break;
                        case uint value: status = SetKernelArgument(kernel, (uint)i, 4, (nint)(&value), argumentCache); break;
                        case float value: status = SetKernelArgument(kernel, (uint)i, 4, (nint)(&value), argumentCache); break;
                        case LocalMemory local: status = SetKernelArgument(kernel, (uint)i, checked((nuint)local.Bytes), 0, argumentCache); break;
                        default: throw new ArgumentException($"Unsupported Arc kernel argument {argument.GetType()}.");
                    }
                    if (status != 0) OpenClNative.Check(status, $"set {name} argument {i}");
                }
                // With timing disabled, retain one event at each half-window
                // boundary. The existing partial wait therefore fences the
                // same oldest commands without allocating an event per kernel.
                bool eventRequired = CollectKernelEvents || (Options.BatchDispatch && Options.PipelineEventCollection
                    && (_pendingEvents.Count + 1) % (Options.QueuedKernelLimit / 2) == 0);
                int launchStatus = OpenClNative.clEnqueueNDRangeKernel(_queue, kernel, (uint)global.Length, 0,
                    global, localSize, 0, 0, eventRequired ? (nint)(&kernelEvent) : 0);
                if (launchStatus != 0) OpenClNative.Check(launchStatus, $"launch {name}");
                KernelLaunchCount++;
                }
                if (label is not null) DetailedProfiler!.Add("host-submit", label, Stopwatch.GetElapsedTime(submitStart).TotalMilliseconds);
                if (temporary is not null) foreach (var (buffer, host) in temporary)
                {
                    if (!host.ReadBack || host.Values.Length == 0) continue;
                    GCHandle pin = GCHandle.Alloc(host.Values, GCHandleType.Pinned);
                    long readStart = Stopwatch.GetTimestamp();
                    nint readEvent = 0;
                    try {
                        using (Timeline?.Host("blocking-transfer-wait", "host-result"))
                            OpenClNative.Check(OpenClNative.clEnqueueReadBuffer(_queue, buffer.Handle, 1, 0,
                                (nuint)Buffer.ByteLength(host.Values), pin.AddrOfPinnedObject(), 0, 0,
                                Timeline is null ? 0 : (nint)(&readEvent)), $"read {name}");
                        Timeline?.Device("D2H", label, readEvent, "D2H", Buffer.ByteLength(host.Values));
                        D2HBytes += Buffer.ByteLength(host.Values);
                        double elapsed = Stopwatch.GetElapsedTime(readStart).TotalMilliseconds;
                        TransferMilliseconds += elapsed;
                        DetailedProfiler?.Add("D2H", $"{DetailedProfiler.Phase}/host-result/{Buffer.ByteLength(host.Values)}B", elapsed, Buffer.ByteLength(host.Values));
                    }
                    finally { if (readEvent != 0) OpenClNative.clReleaseEvent(readEvent); pin.Free(); }
                }
                _pendingEvents.Add((name, kernelEvent, label));
                kernelEvent = 0;
                // Bound queued command/event resources. No per-kernel fence is
                // necessary when all uses (including pooled reuse) share this queue.
                if (!Options.BatchDispatch || temporary?.Any(t => t.Array.ReadBack) == true)
                    SynchronizeCore(!Options.BatchDispatch ? "unbatched" : "host-result");
                else if (_pendingEvents.Count >= Options.QueuedKernelLimit)
                {
                    if (Options.PipelineEventCollection) DrainOldestEventHalf();
                    else SynchronizeCore("event-limit");
                }
            }
            catch { if (_queue != 0) OpenClNative.clFinish(_queue); DrainCompletedEvents(); throw; }
            finally {
                if (kernelEvent != 0) OpenClNative.clReleaseEvent(kernelEvent);
                if (temporary is not null) foreach (var entry in temporary) entry.Buffer.Dispose();
            }
        }
    }

    private nint GetOrCreateKernel(string name, bool profileCompilation)
    {
        if (!_kernels.TryGetValue(name, out nint kernel))
        {
            using var compilation = profileCompilation
                ? Timeline?.Host("kernel-create-compile", name)
                : null;
            kernel = OpenClNative.clCreateKernel(ProgramForRequestedKernel(name), name, out int error);
            OpenClNative.Check(error, $"create kernel {name}");
            _kernels.Add(name, kernel);
        }
        return kernel;
    }
}
