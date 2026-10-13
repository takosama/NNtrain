using System.Runtime.InteropServices;
using System.Diagnostics;

namespace NNtrain.Arc;

public sealed partial class ArcExecutionLane
{

    public void ReadRaw(ArcBuffer buffer, Array values)
    {
        lock (_sync) Transfer(buffer, values, read: true);
    }

    public void WriteRaw(ArcBuffer buffer, Array values)
    {
        lock (_sync) Transfer(buffer, values, read: false);
    }

    public void ReadFloatRange(ArcBuffer buffer, int elementOffset, float[] values)
    {
        lock (_sync) Transfer(buffer, values, read: true, checked(elementOffset * 4L));
    }

    /// <summary>Write a contiguous FP32 range, completing the transfer before returning.</summary>
    public void WriteFloatRange(ArcBuffer buffer, int elementOffset, float[] values)
    {
        lock (_sync) Transfer(buffer, values, read: false, checked(elementOffset * 4L));
    }

    public void ReadBFloat16Range(ArcBuffer buffer, int elementOffset, ushort[] values)
    {
        lock (_sync) Transfer(buffer, values, read: true, checked(elementOffset * 2L));
    }

    public unsafe void CopyBytes(ArcBuffer source, ArcBuffer target, int sourceOffset, int targetOffset, int bytes)
    {
        lock (_sync)
        {
            if (source.Owner != this || target.Owner != this || !source.IsAlive || !target.IsAlive
                || sourceOffset < 0 || targetOffset < 0 || bytes < 0
                || sourceOffset + (long)bytes > source.Bytes || targetOffset + (long)bytes > target.Bytes)
                throw new ArgumentException("Invalid Arc device copy range.");
            if (bytes == 0) return;
            nint evt = 0;
            using (Timeline?.Host("queue-submit", "D2D"))
                OpenClNative.Check(OpenClNative.clEnqueueCopyBuffer(_queue, source.Handle, target.Handle,
                    (nuint)sourceOffset, (nuint)targetOffset, (nuint)bytes, 0, 0, Timeline is null ? 0 : (nint)(&evt)), "device copy");
            if (evt != 0) _pendingCopyEvents.Add(("D2D", evt, bytes));
            _queuedCopies = true;
            if (!Options.BatchDispatch) Synchronize();
        }
    }

    public void Write(ArcBuffer buffer, float[] values)
    {
        lock (_sync) Transfer(buffer, values, read: false);
    }

    public void Read(ArcBuffer buffer, float[] values)
    {
        lock (_sync) Transfer(buffer, values, read: true);
    }

    /// <summary>
    /// Queue a readback on this lane's in-order queue without waiting for the
    /// preceding kernels. The destination stays pinned until the returned
    /// handle is completed or disposed. The source buffer must also remain
    /// undisposed until then. The caller may enqueue later kernels before
    /// waiting, provided it does not inspect the host array early.
    /// </summary>
    public unsafe PendingRead ReadAsync(ArcBuffer buffer, float[] values)
    {
        ArgumentNullException.ThrowIfNull(values);
        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (buffer.Owner != this || !buffer.IsAlive || (long)values.Length * sizeof(float) > buffer.Bytes)
                throw new ArgumentException("Arc transfer buffer is disposed, too small, or belongs to a different lane.", nameof(buffer));
            if (values.Length == 0) throw new ArgumentException("Asynchronous Arc read requires a nonempty destination.", nameof(values));
            GCHandle pin = GCHandle.Alloc(values, GCHandleType.Pinned);
            nint evt = 0;
            bool queued = false;
            try
            {
                long bytes = checked((long)values.Length * sizeof(float));
                using (Timeline?.Host("queue-submit", "D2H-async"))
                    OpenClNative.Check(OpenClNative.clEnqueueReadBuffer(_queue, buffer.Handle, 0,
                        0, (nuint)bytes, pin.AddrOfPinnedObject(), 0, 0, (nint)(&evt)), "queue asynchronous download");
                queued = true;
                if (evt == 0) throw new InvalidOperationException("OpenCL did not return a readback event.");
                _queuedCopies = true;
                D2HBytes += bytes;
                return new PendingRead(this, buffer, values, pin, evt, bytes);
            }
            catch
            {
                bool completed = !queued;
                if (queued)
                {
                    nint pending = evt;
                    if (pending != 0)
                    {
                        OpenClNative.clFlush(_queue);
                        completed = OpenClNative.clWaitForEvents(1, (nint)(&pending)) == 0;
                    }
                    if (!completed) completed = OpenClNative.clFinish(_queue) == 0;
                }
                if (completed)
                {
                    if (evt != 0) OpenClNative.clReleaseEvent(evt);
                    pin.Free();
                }
                throw;
            }
        }
    }

    public sealed class PendingRead : IDisposable
    {
        private readonly ArcExecutionLane _lane;
        private readonly ArcBuffer _source;
        // Retain the managed destination as well as its pin until the event completes.
        private readonly float[] _destination;
        private GCHandle _pin;
        private nint _event;
        private readonly long _bytes;

        internal PendingRead(ArcExecutionLane lane, ArcBuffer source, float[] destination,
            GCHandle pin, nint evt, long bytes)
            => (_lane, _source, _destination, _pin, _event, _bytes) = (lane, source, destination, pin, evt, bytes);

        public unsafe void Wait()
        {
            lock (_lane._sync)
            {
                if (_event == 0) return;
                long start = Stopwatch.GetTimestamp();
                bool completed = false;
                try
                {
                    OpenClNative.Check(OpenClNative.clFlush(_lane._queue), "flush asynchronous download");
                    nint evt = _event;
                    OpenClNative.Check(OpenClNative.clWaitForEvents(1, (nint)(&evt)), "wait asynchronous download");
                    completed = true;
                    _lane.Timeline?.Device("D2H", _lane.DetailedProfiler?.Phase, _event, "D2H", _bytes);
                    _lane.TransferMilliseconds += Stopwatch.GetElapsedTime(start).TotalMilliseconds;
                    _lane.DetailedProfiler?.Add("D2H-async-wait", $"{_lane.DetailedProfiler.Phase}/{_bytes}B",
                        Stopwatch.GetElapsedTime(start).TotalMilliseconds, _bytes);
                    GC.KeepAlive(_source);
                    GC.KeepAlive(_destination);
                }
                finally
                {
                    if (!completed && _lane._queue != 0)
                        completed = OpenClNative.clFinish(_lane._queue) == 0;
                    // If OpenCL cannot establish completion, keep the array
                    // pinned and the source referenced rather than risking a
                    // native DMA into relocated managed memory.
                    if (completed)
                    {
                        OpenClNative.clReleaseEvent(_event);
                        _event = 0;
                        _pin.Free();
                    }
                }
            }
        }

        public void Dispose() => Wait();
    }

    private unsafe void Transfer(ArcBuffer buffer, Array values, bool read, long offset = 0)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (buffer.Owner != this || buffer.Handle == 0 || offset < 0 || Buffer.ByteLength(values) + offset > buffer.Bytes)
            throw new ArgumentException("Arc transfer buffer is disposed, too small, or belongs to a different lane.");
        if (values.Length == 0) return;
        GCHandle pin = GCHandle.Alloc(values, GCHandleType.Pinned);
        long start = Stopwatch.GetTimestamp();
        nint evt = 0;
        try
        {
            nuint bytes = (nuint)Buffer.ByteLength(values);
            int result;
            using (Timeline?.Host("blocking-transfer-wait", read ? "D2H" : "H2D"))
                result = read
                    ? OpenClNative.clEnqueueReadBuffer(_queue, buffer.Handle, 1, (nuint)offset, bytes, pin.AddrOfPinnedObject(), 0, 0, Timeline is null ? 0 : (nint)(&evt))
                    : OpenClNative.clEnqueueWriteBuffer(_queue, buffer.Handle, 1, (nuint)offset, bytes, pin.AddrOfPinnedObject(), 0, 0, Timeline is null ? 0 : (nint)(&evt));
            OpenClNative.Check(result, read ? "download" : "upload");
            Timeline?.Device(read ? "D2H" : "H2D", DetailedProfiler?.Phase, evt, read ? "D2H" : "H2D", (long)bytes);
            // Blocking transfers complete all earlier commands in our in-order queue.
            DrainCompletedEvents();
            if (read) D2HBytes += (long)bytes; else H2DBytes += (long)bytes;
            double elapsed = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            TransferMilliseconds += elapsed;
            DetailedProfiler?.Add(read ? "D2H" : "H2D", $"{DetailedProfiler.Phase}/{bytes}B", elapsed, (long)bytes);
        }
        finally { if (evt != 0) OpenClNative.clReleaseEvent(evt); pin.Free(); }
    }
}
