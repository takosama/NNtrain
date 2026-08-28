
using System.Collections.Concurrent;
using NNtrain.Cuda.Interop;
using NNtrain.Runtime.Execution;

namespace NNtrain;

/// <summary>
/// Managed timeline for the most recently completed BF16 gradient exchange.
/// Bucket-ready timestamps describe managed publication; on the non-peer host
/// path, a completed host work item is the first point that proves its physical
/// D2H/H2D exchange finished.  Consequently <see
/// cref="HostWorkCompletedBeforeComplete"/> is a conservative, directly
/// observable measure of communication/backward overlap.
/// </summary>
internal sealed record CudaGradientOverlapTelemetry(
    long StepId,
    int BucketCount,
    int ScheduledHostWorkCount,
    int CompletedHostWorkCount,
    int FailedHostWorkCount,
    int HostWorkCompletedBeforeComplete,
    double? FirstBucketReadyMilliseconds,
    double? LastBucketReadyMilliseconds,
    double? FirstHostWorkStartedMilliseconds,
    double? FirstHostWorkCompletedMilliseconds,
    double? LastHostWorkCompletedMilliseconds,
    double CompleteEnteredMilliseconds,
    double CompleteFinishedMilliseconds,
    double CompleteHostWaitMilliseconds,
    bool UsedAsyncHostPipeline,
    bool UsedExternalCapturedReadyEvents,
    int[] BucketPublicationOrder);

internal enum CudaCapturedBackwardRecordingMode
{
    ReservationPrewarm = 1,
    StreamCapture = 2,
}

/// <summary>
/// Packs completed FP32 leaf accumulators into BF16 transport buckets and
/// overlaps two-GPU exchange with the remaining backward graph on non-blocking
/// CUDA streams. The reduced/unpacked gradient remains FP32 authoritative;
/// this plan is therefore also the mix8_32 transport path.
/// </summary>
internal sealed class CudaBFloat16GradientAllReducePlan
    : ICudaGradientReductionPlan, IDisposable
{
    private const int TargetBucketElements = 4 * 1024 * 1024;
    private const int NonPeerTargetBucketElements = 16 * 1024 * 1024;
    private const int DefaultHostPipelineChunkElements = 4 * 1024 * 1024;
    private readonly Parameter[] _parameters;
    private readonly int[] _devices;
    private readonly Bucket[] _buckets;
    private readonly DeviceBuffers[] _deviceBuffers;
    private readonly Dictionary<Tensor, SegmentLocation> _locations;
    private readonly int[][] _remaining;
    private readonly long[][][] _notificationSteps;
    private readonly int[] _readyDeviceCounts;
    private readonly long[] _deviceBeginSteps;
    private readonly int[][] _capturedBucketOrder;
    private readonly int[] _capturedBucketOrderCounts;
    private readonly int[] _capturedRecordingModes;
    private readonly CudaCapturedGradientPublicationState
        _capturedPublicationState;
    private readonly bool _overlapExchange;
    private readonly bool _useHostPipeline;
    private readonly bool _asyncHostPipeline;
    private readonly bool _usesBFloat16GradientStorage;
    private readonly bool _usesExternalCapturedReadyEvents;
    private readonly int _hostPipelineChunkElements;
    private readonly Func<long, int, int, Exception?>?
        _hostReductionFaultInjector;
    private readonly nint[] _hostPipelines;
    private readonly HostReductionWorker[] _hostWorkers;
    private readonly object _hostCompletionSync = new();
    private readonly ManualResetEventSlim _hostCompletion = new(true);
    private NativeCudaBuffer<double>? _primarySquaredSum;
    private readonly long _reducerGeneration =
        CudaGradientReductionStampSource.CreateReducerGeneration();
    private long _stepSequence;
    private long _activeStepId;
    private long _completedSteps;
    private long _lastCompletedTransportBytes;
    private long _managedLocalPackSubmissionCount;
    private long _capturedReplayReadyEventRecordCount;
    private long _capturedReplayReadyEventRecordTicks;
    private long _stepStartedTicks;
    private long _firstBucketReadyTicks;
    private long _lastBucketReadyTicks;
    private long _firstHostWorkStartedTicks;
    private long _firstHostWorkCompletedTicks;
    private long _lastHostWorkCompletedTicks;
    private long _completeEnteredTicks;
    private long _completeHostWaitTicks;
    private int _completedHostWorkCount;
    private int _failedHostWorkCount;
    private int _hostWorkCompletedBeforeComplete;
    private int _stepUsedExternalCapturedReadyEvents;
    private readonly int[] _stepBucketPublicationOrder;
    private int _stepBucketPublicationCount;
    private CudaGradientOverlapTelemetry? _lastOverlapTelemetry;
    private int _hostOutstanding;
    private int _hostScheduledForStep;
    private Exception? _hostFailure;
    private int _disposed;

    internal CudaBFloat16GradientAllReducePlan(
        IReadOnlyList<Parameter> parameters,
        IReadOnlyList<int> devices,
        CudaDispatchPolicy? dispatchPolicy = null,
        bool useBFloat16GradientStorage = false,
        Func<long, int, int, Exception?>? hostReductionFaultInjector = null,
        Func<int, int, bool>? peerAccessProbe = null)
    {
        if (devices.Count != 2)
        {
            throw new ArgumentException(
                "Asynchronous BF16 gradient buckets currently require two GPUs.",
                nameof(devices));
        }
        _parameters = parameters.ToArray();
        _devices = devices.ToArray();
        _usesBFloat16GradientStorage = useBFloat16GradientStorage;
        _hostReductionFaultInjector = hostReductionFaultInjector;
        if (_usesBFloat16GradientStorage
            && _parameters.Any(parameter =>
                parameter.T.DType != TensorDType.BFloat16))
        {
            throw new ArgumentException(
                "Direct BF16 gradient arenas require BFloat16 parameters.",
                nameof(parameters));
        }
        CudaDispatchPolicy dispatch =
            (dispatchPolicy ?? CudaDispatchPolicy.Current).Validate();
        _usesExternalCapturedReadyEvents =
            !dispatch.DisableExternalGradientReadyEvents
            && CudaNativeGateway.AbiVersion.Minor
                >= CudaAbiVersion.ExternalGradientReadyEventMinor;
        Func<int, int, bool> canAccessPeer =
            peerAccessProbe ?? NativeCudaRuntime.CanAccessPeer;
        _overlapExchange = canAccessPeer(
                _devices[0], _devices[1])
            && canAccessPeer(_devices[1], _devices[0]);
        _useHostPipeline = !_overlapExchange
            && !dispatch.DisableGradientHostPipeline;
        _asyncHostPipeline = _useHostPipeline
            && !dispatch.DisableAsyncGradientHostPipeline;
        _hostPipelineChunkElements =
            ResolveHostPipelineChunkElements(dispatch);
        _buckets = BuildBuckets(
            _parameters,
            ResolveTargetBucketElements(_overlapExchange, dispatch));
        _locations = new Dictionary<Tensor, SegmentLocation>(
            ReferenceEqualityComparer.Instance);
        for (int bucket = 0; bucket < _buckets.Length; bucket++)
        {
            for (int segment = 0;
                segment < _buckets[bucket].Segments.Length;
                segment++)
            {
                _locations.Add(
                    _buckets[bucket].Segments[segment].Tensor,
                    new SegmentLocation(bucket, segment));
            }
        }
        _remaining = Enumerable.Range(0, _buckets.Length)
            .Select(bucket => new int[_devices.Length])
            .ToArray();
        _readyDeviceCounts = new int[_buckets.Length];
        _notificationSteps = Enumerable.Range(0, _buckets.Length)
            .Select(bucket => Enumerable.Range(0, _devices.Length)
                .Select(_ => new long[_buckets[bucket].Segments.Length])
                .ToArray())
            .ToArray();
        _deviceBeginSteps = new long[_devices.Length];
        _capturedBucketOrder = Enumerable.Range(0, _devices.Length)
            .Select(_ => new int[_buckets.Length])
            .ToArray();
        _capturedBucketOrderCounts = new int[_devices.Length];
        _capturedRecordingModes = new int[_devices.Length];
        _stepBucketPublicationOrder = new int[_buckets.Length];
        _capturedPublicationState =
            new CudaCapturedGradientPublicationState(_devices.Length);
        _deviceBuffers = new DeviceBuffers[_devices.Length];
        _hostPipelines = new nint[_devices.Length];
        _hostWorkers = new HostReductionWorker[_devices.Length];
        try
        {
            for (int device = 0; device < _devices.Length; device++)
                _deviceBuffers[device] = CreateDeviceBuffers(device);
            _primarySquaredSum = _deviceBuffers[0].Accelerator
                .Allocate1D<double>(1);
            if (_useHostPipeline)
            {
                for (int destination = 0;
                    destination < _devices.Length;
                    destination++)
                {
                    int source = 1 - destination;
                    _hostPipelines[destination] =
                        CudaGradientBuckets.CreateHostPipeline(
                            _devices[source],
                            _devices[destination],
                            _hostPipelineChunkElements);
                    if (_asyncHostPipeline)
                    {
                        int capturedDestination = destination;
                        _hostWorkers[destination] = new HostReductionWorker(
                            $"NNtrain BF16 all-reduce GPU " +
                            $"{_devices[destination]}",
                            work => ExecuteHostReduction(
                                work,
                                capturedDestination));
                    }
                }
            }
            BindGradientArenas();
            foreach (Parameter parameter in _parameters)
            {
                parameter.T.RegisterCudaGradientReducer(
                    _reducerGeneration,
                    _devices,
                    ownsGradientZeroing: true);
            }
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    internal bool Matches(
        IReadOnlyList<Parameter> parameters,
        IReadOnlyList<int> devices)
        => parameters.Count == _parameters.Length
            && devices.SequenceEqual(_devices)
            && parameters.Select((parameter, index) =>
                ReferenceEquals(parameter, _parameters[index])).All(value => value);

    /// <summary>
    /// True only when the selected transport cannot enqueue communication as
    /// buckets become ready and therefore starts exchange in Complete().
    /// </summary>
    internal bool DefersExchangeUntilBackward
        => !_overlapExchange && !_asyncHostPipeline;

    internal bool UsesHostPipeline => _useHostPipeline;

    internal bool UsesAsyncHostPipeline => _asyncHostPipeline;

    internal bool UsesExternalCapturedReadyEvents
        => _usesExternalCapturedReadyEvents;

    internal int BucketCount => _buckets.Length;

    internal int[] BucketElementCounts
        => _buckets.Select(bucket => bucket.TotalElements).ToArray();

    internal static int ActiveHostWorkerCount
        => HostReductionWorker.ActiveCount;

    internal int TotalGradientElements
        => _buckets.Sum(bucket => bucket.TotalElements);

    internal bool OwnsCommunicationStream(int deviceIndex)
    {
        int slot = Array.IndexOf(_devices, deviceIndex);
        if (slot < 0)
            throw new ArgumentOutOfRangeException(nameof(deviceIndex));
        return _deviceBuffers[slot].OwnsCommunicationStream;
    }

    /// <summary>
    /// Counts payload bytes crossing a device boundary. With two devices each
    /// destination receives the other device's BF16 bucket exactly once.
    /// </summary>
    internal long TransportBytesPerStep => checked(
        (long)_devices.Length * TotalGradientElements * sizeof(ushort));

    internal long LastCompletedTransportBytes
        => Volatile.Read(ref _lastCompletedTransportBytes);

    internal long CompletedSteps => Volatile.Read(ref _completedSteps);

    /// <summary>
    /// Managed submissions of the local FP32-to-BF16 pack operation. CUDA
    /// Graph replay executes the captured native node without increasing this
    /// counter, so it also guards the state-only publication contract.
    /// </summary>
    internal long ManagedLocalPackSubmissionCount
        => Interlocked.Read(ref _managedLocalPackSubmissionCount);

    /// <summary>
    /// Ready events re-recorded after a CUDA Graph launch. Re-recording only
    /// appends an event to the compute stream; it never re-packs gradients.
    /// </summary>
    internal long CapturedReplayReadyEventRecordCount
        => Interlocked.Read(ref _capturedReplayReadyEventRecordCount);

    /// <summary>
    /// Managed submission time for post-replay ready-event records. This is
    /// enqueue overhead, not device execution time.
    /// </summary>
    internal double CapturedReplayReadyEventRecordMilliseconds
        => Interlocked.Read(ref _capturedReplayReadyEventRecordTicks)
            * 1000d / System.Diagnostics.Stopwatch.Frequency;

    internal CudaGradientOverlapTelemetry? LastOverlapTelemetry
        => Volatile.Read(ref _lastOverlapTelemetry);

    internal long BeginStep()
    {
        ObjectDisposedException.ThrowIf(
            Volatile.Read(ref _disposed) != 0, this);
        if (Volatile.Read(ref _activeStepId) != 0)
        {
            throw new InvalidOperationException(
                "The previous BF16 gradient reduction step is still active.");
        }
        long stepId = Interlocked.Increment(ref _stepSequence);
        if (stepId == 0)
            stepId = Interlocked.Increment(ref _stepSequence);
        if (_asyncHostPipeline)
        {
            lock (_hostCompletionSync)
            {
                if (_hostOutstanding != 0)
                {
                    throw new InvalidOperationException(
                        "The previous BF16 host reduction is still running.");
                }
                _hostScheduledForStep = 0;
                _hostFailure = null;
                _hostCompletion.Set();
            }
        }
        Volatile.Write(
            ref _stepStartedTicks,
            System.Diagnostics.Stopwatch.GetTimestamp());
        Volatile.Write(ref _firstBucketReadyTicks, 0);
        Volatile.Write(ref _lastBucketReadyTicks, 0);
        Volatile.Write(ref _firstHostWorkStartedTicks, 0);
        Volatile.Write(ref _firstHostWorkCompletedTicks, 0);
        Volatile.Write(ref _lastHostWorkCompletedTicks, 0);
        Volatile.Write(ref _completeEnteredTicks, 0);
        Volatile.Write(ref _completeHostWaitTicks, 0);
        Volatile.Write(ref _completedHostWorkCount, 0);
        Volatile.Write(ref _failedHostWorkCount, 0);
        Volatile.Write(ref _hostWorkCompletedBeforeComplete, 0);
        Volatile.Write(ref _stepUsedExternalCapturedReadyEvents, 0);
        Volatile.Write(ref _stepBucketPublicationCount, 0);
        Array.Fill(_stepBucketPublicationOrder, -1);
        var stamp = new CudaGradientReductionStamp(
            _reducerGeneration, stepId);
        try
        {
            foreach (Parameter parameter in _parameters)
                parameter.T.BeginCudaGradientReduction(stamp, _devices);
            _capturedPublicationState.BeginStep(stepId);
        }
        catch
        {
            _capturedPublicationState.EndStep(stepId);
            foreach (Parameter parameter in _parameters)
                parameter.T.AbortCudaGradientReduction(stamp);
            throw;
        }
        for (int bucket = 0; bucket < _buckets.Length; bucket++)
        {
            Volatile.Write(ref _readyDeviceCounts[bucket], 0);
            for (int device = 0; device < _devices.Length; device++)
            {
                Volatile.Write(
                    ref _remaining[bucket][device],
                    _buckets[bucket].Segments.Length);
            }
        }
        Volatile.Write(ref _activeStepId, stepId);
        return stepId;
    }

    internal void BeginDeviceStep(long stepId, int deviceIndex)
    {
        if (Volatile.Read(ref _activeStepId) != stepId)
            throw new InvalidOperationException("The gradient step is not active.");
        int deviceSlot = Array.IndexOf(_devices, deviceIndex);
        if (deviceSlot < 0)
            throw new ArgumentOutOfRangeException(nameof(deviceIndex));
        if (Interlocked.Exchange(
                ref _deviceBeginSteps[deviceSlot], stepId) == stepId)
        {
            throw new InvalidOperationException(
                $"CUDA device {deviceIndex} was begun twice for step {stepId}.");
        }

        // CUDA's per-thread default stream is scoped to this worker. Clearing
        // on the coordinator thread can race the worker's first backward
        // accumulation and retain gradients from the preceding step. Queue
        // every reset on the same device/thread that will run the shard.
        if (deviceSlot == 0)
            _primarySquaredSum!.MemSetToZero();
        for (int bucketIndex = 0;
            bucketIndex < _deviceBuffers[deviceSlot].Buckets.Length;
            bucketIndex++)
        {
            BucketBuffers bucket =
                _deviceBuffers[deviceSlot].Buckets[bucketIndex];
            if (_usesBFloat16GradientStorage)
            {
                _ = bucket.LocalArena.ClearIfDirty();
                _ = bucket.GradientArena.ClearIfDirty();
            }
            else
            {
                _ = bucket.GradientArena.ClearIfDirty();
            }
            foreach (Segment segment in _buckets[bucketIndex].Segments)
            {
                segment.Tensor.CompleteReducerOwnedCudaGradientZero(
                    _reducerGeneration,
                    deviceIndex);
            }
        }
        _capturedPublicationState.MarkDeviceBegun(
            stepId,
            deviceSlot,
            deviceIndex);
    }

    internal void NotifyGradientReady(
        Tensor tensor,
        int deviceIndex,
        long stepId)
    {
        if (Volatile.Read(ref _activeStepId) != stepId)
        {
            throw new InvalidOperationException(
                $"BF16 gradient step {stepId} is not active.");
        }
        if (!_locations.TryGetValue(tensor, out SegmentLocation location))
            return;
        int deviceSlot = Array.IndexOf(_devices, deviceIndex);
        if (deviceSlot < 0)
            throw new ArgumentOutOfRangeException(nameof(deviceIndex));
        if (Volatile.Read(ref _deviceBeginSteps[deviceSlot]) != stepId)
        {
            throw new InvalidOperationException(
                $"CUDA device {deviceIndex} was not begun for step {stepId}.");
        }
        bool captureRecording = _capturedPublicationState
            .IsCaptureRecording(stepId, deviceSlot);
        if (captureRecording)
        {
            _capturedPublicationState.EnterCaptureNotificationPath(
                stepId,
                deviceSlot,
                deviceIndex);
        }
        else
        {
            _capturedPublicationState.EnterNotificationPath(
                stepId,
                deviceSlot,
                deviceIndex);
        }

        ref long notification = ref _notificationSteps[location.Bucket]
            [deviceSlot][location.Segment];
        while (true)
        {
            if (Volatile.Read(ref _activeStepId) != stepId)
            {
                throw new InvalidOperationException(
                    $"BF16 gradient step {stepId} stopped during " +
                    "notification.");
            }
            long previous = Volatile.Read(ref notification);
            if (previous == stepId)
            {
                throw new InvalidOperationException(
                    $"BF16 gradient '{tensor.Name}' was notified twice on " +
                    $"CUDA device {deviceIndex} for step {stepId}.");
            }
            if (Interlocked.CompareExchange(
                    ref notification, stepId, previous) == previous)
            {
                break;
            }
        }
        if (Volatile.Read(ref _activeStepId) != stepId)
        {
            throw new InvalidOperationException(
                $"BF16 gradient step {stepId} stopped during notification.");
        }

        DeviceBuffers deviceBuffers = _deviceBuffers[deviceSlot];
        NativeCudaDevice accelerator = deviceBuffers.Accelerator;
        BucketBuffers bucketBuffers = deviceBuffers.Buckets[location.Bucket];
        bool ownsExpectedArena = _usesBFloat16GradientStorage
            ? ReferenceEquals(
                tensor.GetCudaBFloat16GradientArena(deviceIndex),
                bucketBuffers.LocalArena)
            : ReferenceEquals(
                tensor.GetCudaGradientArena(deviceIndex),
                bucketBuffers.GradientArena);
        if (!ownsExpectedArena)
        {
            throw new InvalidOperationException(
                $"Gradient bucket {location.Bucket}, segment " +
                $"{location.Segment} lost its CUDA arena binding on device " +
                $"{deviceIndex} during step {stepId}.");
        }

        if (Interlocked.Decrement(
            ref _remaining[location.Bucket][deviceSlot]) != 0)
        {
            return;
        }
        if (_usesBFloat16GradientStorage)
        {
            bucketBuffers.LocalArena.MarkDirty();
        }
        else
        {
            CudaGradientBuckets.Pack(
                deviceIndex,
                accelerator,
                bucketBuffers.GradientArena.Buffer,
                bucketBuffers.Local,
                0,
                _buckets[location.Bucket].TotalElements);
            Interlocked.Increment(ref _managedLocalPackSubmissionCount);
        }
        bool streamCaptureRecording = captureRecording
            && Volatile.Read(ref _capturedRecordingModes[deviceSlot])
                == (int)CudaCapturedBackwardRecordingMode.StreamCapture;
        if (streamCaptureRecording && _usesExternalCapturedReadyEvents)
        {
            CudaGradientBuckets.RecordReadyExternal(
                deviceIndex,
                accelerator,
                bucketBuffers.ReadyEvent);
        }
        else
        {
            CudaGradientBuckets.RecordReady(
                deviceIndex,
                accelerator,
                bucketBuffers.ReadyEvent);
        }
        if (captureRecording)
        {
            RecordCapturedBucketOrder(deviceSlot, location.Bucket);
            return;
        }
        if (Interlocked.Increment(
                ref _readyDeviceCounts[location.Bucket]) == _devices.Length)
        {
            RecordBucketReady(location.Bucket);
            if (_asyncHostPipeline)
            {
                QueueHostBucketReduction(stepId, location.Bucket);
            }
            else if (_overlapExchange)
            {
                try
                {
                    EnqueueBucketReduction(location.Bucket);
                }
                finally
                {
                    accelerator.Bind();
                }
            }
        }
    }

    void ICudaGradientReductionPlan.NotifyGradientReady(
        Tensor tensor,
        int deviceIndex,
        long stepId)
        => NotifyGradientReady(tensor, deviceIndex, stepId);

    /// <summary>
    /// Marks leaf notifications on one device as graph-recording work. Pack
    /// and ready-event nodes are captured, while cross-device exchange is
    /// suppressed until a later replay generation is state-published.
    /// </summary>
    internal IDisposable BeginCapturedBackwardRecording(
        long stepId,
        int deviceIndex,
        CudaCapturedBackwardRecordingMode recordingMode =
            CudaCapturedBackwardRecordingMode.ReservationPrewarm)
    {
        if (Volatile.Read(ref _activeStepId) != stepId)
            throw new InvalidOperationException("The gradient step is not active.");
        int deviceSlot = Array.IndexOf(_devices, deviceIndex);
        if (deviceSlot < 0)
            throw new ArgumentOutOfRangeException(nameof(deviceIndex));
        if (recordingMode is not
            (CudaCapturedBackwardRecordingMode.ReservationPrewarm
                or CudaCapturedBackwardRecordingMode.StreamCapture))
        {
            throw new ArgumentOutOfRangeException(nameof(recordingMode));
        }
        if (Volatile.Read(ref _deviceBeginSteps[deviceSlot]) != stepId)
        {
            throw new InvalidOperationException(
                $"CUDA device {deviceIndex} was not begun for step {stepId}.");
        }
        Array.Fill(_capturedBucketOrder[deviceSlot], -1);
        Volatile.Write(ref _capturedBucketOrderCounts[deviceSlot], 0);
        _capturedPublicationState.BeginCaptureRecording(
            stepId,
            deviceSlot,
            deviceIndex);
        Volatile.Write(
            ref _capturedRecordingModes[deviceSlot],
            (int)recordingMode);
        return new CudaCapturedBackwardRecordingScope(() =>
        {
            try
            {
                _capturedPublicationState.EndCaptureRecording(
                    stepId,
                    deviceSlot,
                    deviceIndex);
            }
            finally
            {
                Volatile.Write(ref _capturedRecordingModes[deviceSlot], 0);
            }
        });
    }

    /// <summary>
    /// Drops only the managed generation used to record a graph. No graph was
    /// launched and no communication work was submitted, so communication
    /// stream synchronization would be both unnecessary and unsafe here.
    /// </summary>
    internal void DiscardCapturedBackwardRecordingStep(long stepId)
    {
        if (Volatile.Read(ref _activeStepId) != stepId)
            throw new InvalidOperationException("The gradient step is not active.");
        _capturedPublicationState.ValidateCaptureDiscard(stepId);
        var stamp = new CudaGradientReductionStamp(
            _reducerGeneration,
            stepId);
        Volatile.Write(ref _activeStepId, 0);
        _capturedPublicationState.EndStep(stepId);
        foreach (Parameter parameter in _parameters)
            parameter.T.AbortCudaGradientReduction(stamp);
    }

    /// <summary>
    /// Publishes the managed readiness of every arena-backed gradient produced
    /// by one captured backward replay. Pack kernels and compute-stream event
    /// records already belong to the replayed graph; this method must not
    /// enqueue either operation a second time.
    /// </summary>
    internal void PublishCapturedDeviceGradients(
        long stepId,
        int deviceIndex)
        => PublishCapturedDeviceGradientsCore(
            stepId,
            deviceIndex,
            recordReadyEventsAfterReplay: false);

    /// <summary>
    /// Publishes gradients produced by a captured backward replay and records
    /// a fresh ready event after the graph launch on the same compute stream.
    /// Cross-device host pipelines consume this event instead of depending on
    /// an event node whose record belongs to the captured graph. No gradient
    /// pack, quantization, or math kernel is submitted by this method.
    /// </summary>
    internal void PublishCapturedDeviceGradientsAfterReplay(
        long stepId,
        int deviceIndex)
        => PublishCapturedDeviceGradientsCore(
            stepId,
            deviceIndex,
            recordReadyEventsAfterReplay: true);

    /// <summary>
    /// Publishes one graph replay using the strongest event contract exposed by
    /// the loaded native ABI. ABI 1.19+ records external event nodes at each
    /// bucket boundary, while an older binary retains the safe post-replay
    /// compute-stream record used before external events were available.
    /// </summary>
    internal void PublishCapturedDeviceGradientsForReplay(
        long stepId,
        int deviceIndex,
        int[]? capturedBucketOrder = null)
    {
        if (_usesExternalCapturedReadyEvents)
            Volatile.Write(ref _stepUsedExternalCapturedReadyEvents, 1);
        PublishCapturedDeviceGradientsCore(
            stepId,
            deviceIndex,
            recordReadyEventsAfterReplay:
                !_usesExternalCapturedReadyEvents,
            capturedBucketOrder: capturedBucketOrder);
    }

    /// <summary>
    /// Freezes the event-node order into the compiled graph that owns it.
    /// The reducer is shared by the bounded shape cache, so retaining only the
    /// most recently captured order here would let a later shape overwrite the
    /// publication order of an older graph.
    /// </summary>
    internal int[]? SnapshotCapturedBucketOrderForGraph(int deviceIndex)
    {
        if (!_usesExternalCapturedReadyEvents)
            return null;
        int deviceSlot = Array.IndexOf(_devices, deviceIndex);
        if (deviceSlot < 0)
            throw new ArgumentOutOfRangeException(nameof(deviceIndex));
        ValidateCapturedBucketOrder(deviceSlot);
        return (int[])_capturedBucketOrder[deviceSlot].Clone();
    }

    private void PublishCapturedDeviceGradientsCore(
        long stepId,
        int deviceIndex,
        bool recordReadyEventsAfterReplay,
        int[]? capturedBucketOrder = null)
    {
        if (Volatile.Read(ref _activeStepId) != stepId)
        {
            throw new InvalidOperationException(
                $"BF16 gradient step {stepId} is not active.");
        }
        int deviceSlot = Array.IndexOf(_devices, deviceIndex);
        if (deviceSlot < 0)
            throw new ArgumentOutOfRangeException(nameof(deviceIndex));
        if (Volatile.Read(ref _deviceBeginSteps[deviceSlot]) != stepId)
        {
            throw new InvalidOperationException(
                $"CUDA device {deviceIndex} was not begun for step {stepId}.");
        }

        _capturedPublicationState.BeginCapturedPublication(
            stepId,
            deviceSlot,
            deviceIndex);
        try
        {
            DeviceBuffers deviceBuffers = _deviceBuffers[deviceSlot];
            NativeCudaDevice accelerator = deviceBuffers.Accelerator;
            int[]? publicationOrder = null;
            if (_usesExternalCapturedReadyEvents)
            {
                if (capturedBucketOrder is null)
                {
                    ValidateCapturedBucketOrder(deviceSlot);
                    publicationOrder = _capturedBucketOrder[deviceSlot];
                }
                else
                {
                    ValidateCapturedBucketOrder(
                        deviceSlot,
                        capturedBucketOrder);
                    publicationOrder = capturedBucketOrder;
                }
            }
            for (int bucketIndex = 0;
                bucketIndex < _buckets.Length;
                bucketIndex++)
            {
                BucketBuffers bucketBuffers =
                    deviceBuffers.Buckets[bucketIndex];
                foreach (Segment segment in _buckets[bucketIndex].Segments)
                {
                    bool ownsExpectedArena = _usesBFloat16GradientStorage
                        ? ReferenceEquals(
                            segment.Tensor.GetCudaBFloat16GradientArena(
                                deviceIndex),
                            bucketBuffers.LocalArena)
                        : ReferenceEquals(
                            segment.Tensor.GetCudaGradientArena(deviceIndex),
                            bucketBuffers.GradientArena);
                    if (!ownsExpectedArena)
                    {
                        throw new InvalidOperationException(
                            $"Captured gradient bucket {bucketIndex} lost " +
                            $"its CUDA arena binding on device " +
                            $"{deviceIndex} during step {stepId}.");
                    }
                }
                int expected = _buckets[bucketIndex].Segments.Length;
                if (Volatile.Read(
                        ref _remaining[bucketIndex][deviceSlot]) != expected)
                {
                    throw new InvalidOperationException(
                        $"Captured gradient bucket {bucketIndex} on CUDA " +
                        $"device {deviceIndex} was already partially " +
                        $"published for step {stepId}.");
                }
            }

            if (recordReadyEventsAfterReplay)
            {
                long started = System.Diagnostics.Stopwatch.GetTimestamp();
                long recorded = 0;
                try
                {
                    for (int bucketIndex = 0;
                        bucketIndex < _buckets.Length;
                        bucketIndex++)
                    {
                        CudaGradientBuckets.RecordReady(
                            deviceIndex,
                            accelerator,
                            deviceBuffers.Buckets[bucketIndex].ReadyEvent);
                        recorded++;
                    }
                }
                finally
                {
                    Interlocked.Add(
                        ref _capturedReplayReadyEventRecordCount,
                        recorded);
                    Interlocked.Add(
                        ref _capturedReplayReadyEventRecordTicks,
                        System.Diagnostics.Stopwatch.GetTimestamp() - started);
                }
            }

            for (int orderIndex = 0;
                orderIndex < _buckets.Length;
                orderIndex++)
            {
                int bucketIndex = _usesExternalCapturedReadyEvents
                    ? publicationOrder![orderIndex]
                    : orderIndex;
                Bucket bucket = _buckets[bucketIndex];
                int expected = bucket.Segments.Length;
                if (Interlocked.CompareExchange(
                        ref _remaining[bucketIndex][deviceSlot],
                        0,
                        expected) != expected)
                {
                    throw new InvalidOperationException(
                        $"Captured gradient bucket {bucketIndex} changed " +
                        $"while CUDA device {deviceIndex} was publishing " +
                        $"step {stepId}.");
                }
                if (Interlocked.Increment(
                        ref _readyDeviceCounts[bucketIndex])
                        == _devices.Length)
                {
                    RecordBucketReady(bucketIndex);
                    if (_asyncHostPipeline)
                    {
                        QueueHostBucketReduction(stepId, bucketIndex);
                    }
                    else if (_overlapExchange)
                    {
                        try
                        {
                            EnqueueBucketReduction(bucketIndex);
                        }
                        finally
                        {
                            accelerator.Bind();
                        }
                    }
                }
            }
            _capturedPublicationState.CompleteCapturedPublication(
                stepId,
                deviceSlot,
                deviceIndex);
        }
        catch
        {
            _capturedPublicationState.FailCapturedPublication(
                stepId,
                deviceSlot);
            throw;
        }
    }

    internal void Complete(long stepId)
    {
        if (Volatile.Read(ref _activeStepId) != stepId)
            throw new InvalidOperationException("The gradient step is not active.");
        Interlocked.CompareExchange(
            ref _completeEnteredTicks,
            System.Diagnostics.Stopwatch.GetTimestamp(),
            comparand: 0);
        var stamp = new CudaGradientReductionStamp(
            _reducerGeneration, stepId);
        bool publish = false;
        try
        {
            _capturedPublicationState.ValidateComplete(stepId);
            for (int bucket = 0; bucket < _buckets.Length; bucket++)
            {
                for (int device = 0; device < _devices.Length; device++)
                {
                    if (Volatile.Read(ref _deviceBeginSteps[device]) != stepId)
                    {
                        throw new InvalidOperationException(
                            $"CUDA device {_devices[device]} was not begun " +
                            $"for gradient step {stepId}.");
                    }
                    if (Volatile.Read(ref _remaining[bucket][device]) != 0)
                    {
                        throw new InvalidOperationException(
                            $"Gradient bucket {bucket} was not completed on " +
                            $"CUDA device {_devices[device]}.");
                    }
                }
            }
            if (_useHostPipeline)
            {
                if (_asyncHostPipeline)
                {
                    WaitForHostReduction(stepId, throwOnFailure: true);
                }
                else
                {
                    for (int bucket = 0; bucket < _buckets.Length; ++bucket)
                    {
                        int capturedBucket = bucket;
                        Parallel.For(0, _devices.Length, destination =>
                            EnqueueBucketReduction(
                                capturedBucket,
                                destination));
                    }
                }
            }
            else
            {
                if (!_overlapExchange)
                {
                    for (int bucket = 0; bucket < _buckets.Length; bucket++)
                        EnqueueBucketReduction(bucket);
                }
                Parallel.For(0, _devices.Length, device =>
                {
                    DeviceBuffers buffers = _deviceBuffers[device];
                    CudaGradientBuckets.Synchronize(
                        buffers.Accelerator,
                        buffers.DeviceIndex,
                        buffers.CommunicationStream);
                });
            }
            if (_usesBFloat16GradientStorage)
            {
                PublishReducedBFloat16Buckets();
                RecomputeBFloat16SquaredSum();
            }
            var squaredSum = new double[1];
            _primarySquaredSum!.CopyToCPU(squaredSum);
            foreach (Parameter parameter in _parameters)
            {
                if (_usesBFloat16GradientStorage)
                {
                    parameter.T.MarkCudaBFloat16GradientsSynchronized(
                        _devices,
                        stamp);
                }
                else
                {
                    parameter.T.MarkCudaGradientsSynchronized(
                        _devices,
                        stamp);
                }
            }
            TensorCudaKernels.PublishGradientSquaredSum(
                _parameters, _devices, squaredSum[0]);
            Volatile.Write(
                ref _lastCompletedTransportBytes,
                TransportBytesPerStep);
            Interlocked.Increment(ref _completedSteps);
            publish = true;
            PublishOverlapTelemetry(stepId);
        }
        catch
        {
            // A validation error may occur after earlier buckets have already
            // entered the asynchronous host transport. Drain those jobs before
            // releasing this generation so the next step cannot observe stale
            // work or reuse an arena while a transfer is still in flight.
            if (_asyncHostPipeline)
                WaitForHostReduction(stepId, throwOnFailure: false);
            throw;
        }
        finally
        {
            Volatile.Write(ref _activeStepId, 0);
            _capturedPublicationState.EndStep(stepId);
            if (!publish)
            {
                foreach (Parameter parameter in _parameters)
                    parameter.T.AbortCudaGradientReduction(stamp);
            }
        }
    }

    internal void Abort(long stepId)
    {
        if (Volatile.Read(ref _activeStepId) != stepId)
            return;
        var stamp = new CudaGradientReductionStamp(
            _reducerGeneration, stepId);
        List<Exception>? failures = null;
        try
        {
            if (_asyncHostPipeline)
            {
                try
                {
                    WaitForHostReduction(stepId, throwOnFailure: false);
                }
                catch (Exception exception)
                {
                    (failures ??= []).Add(exception);
                }
                Exception? hostFailure = Volatile.Read(ref _hostFailure);
                if (hostFailure is not null)
                    (failures ??= []).Add(hostFailure);
            }
            foreach (DeviceBuffers buffers in _deviceBuffers)
            {
                if (buffers is null)
                    continue;
                try
                {
                    CudaGradientBuckets.Synchronize(
                        buffers.Accelerator,
                        buffers.DeviceIndex,
                        buffers.CommunicationStream);
                }
                catch (Exception exception)
                {
                    (failures ??= []).Add(exception);
                }
            }
        }
        finally
        {
            Volatile.Write(ref _activeStepId, 0);
            _capturedPublicationState.EndStep(stepId);
            foreach (Parameter parameter in _parameters)
                parameter.T.AbortCudaGradientReduction(stamp);
        }
        if (failures is not null)
        {
            throw new AggregateException(
                "BF16 gradient reduction abort failed.", failures);
        }
    }

    private void EnqueueBucketReduction(int bucketIndex)
    {
        for (int destination = 0; destination < _devices.Length; destination++)
            EnqueueBucketReduction(bucketIndex, destination);
    }

    private void QueueHostBucketReduction(long stepId, int bucketIndex)
    {
        if (Volatile.Read(ref _activeStepId) != stepId)
        {
            throw new InvalidOperationException(
                $"BF16 gradient step {stepId} stopped before host exchange.");
        }
        for (int destination = 0; destination < _devices.Length; destination++)
        {
            lock (_hostCompletionSync)
            {
                if (_hostOutstanding == 0)
                    _hostCompletion.Reset();
                checked
                {
                    _hostOutstanding++;
                    _hostScheduledForStep++;
                }
            }
            try
            {
                _hostWorkers[destination].Enqueue(
                    new HostReductionWork(
                        stepId,
                        bucketIndex,
                        DeviceTransferGuard.CaptureCurrentContext()));
            }
            catch
            {
                CompleteHostReductionWork();
                throw;
            }
        }
    }

    private void ExecuteHostReduction(
        HostReductionWork work,
        int destination)
    {
        bool succeeded = false;
        try
        {
            using IDisposable? transferScope = work.TransferContext is null
                ? null
                : DeviceTransferGuard.EnterSharedContext(
                    work.TransferContext);
            if (Volatile.Read(ref _activeStepId) != work.StepId)
            {
                throw new InvalidOperationException(
                    $"Stale BF16 host reduction work for step " +
                    $"{work.StepId} on CUDA {_devices[destination]}.");
            }
            RecordHostWorkStarted();
            if (Volatile.Read(ref _hostFailure) is null)
            {
                Exception? injected = _hostReductionFaultInjector?.Invoke(
                    work.StepId,
                    work.BucketIndex,
                    destination);
                if (injected is not null)
                    throw injected;
                EnqueueBucketReduction(work.BucketIndex, destination);
                succeeded = true;
            }
        }
        catch (Exception exception)
        {
            Interlocked.CompareExchange(
                ref _hostFailure,
                new InvalidOperationException(
                    $"BF16 host reduction bucket {work.BucketIndex}, " +
                    $"destination CUDA {_devices[destination]} failed.",
                    exception),
                comparand: null);
        }
        finally
        {
            RecordHostWorkCompleted(succeeded);
            CompleteHostReductionWork();
        }
    }

    private void CompleteHostReductionWork()
    {
        lock (_hostCompletionSync)
        {
            _hostOutstanding--;
            if (_hostOutstanding < 0)
            {
                _hostOutstanding = 0;
                _hostFailure ??= new InvalidOperationException(
                    "BF16 host reduction work completed more than once.");
            }
            if (_hostOutstanding == 0)
                _hostCompletion.Set();
        }
    }

    private void WaitForHostReduction(long stepId, bool throwOnFailure)
    {
        int expected = checked(_buckets.Length * _devices.Length);
        int scheduled;
        lock (_hostCompletionSync)
            scheduled = _hostScheduledForStep;
        if (throwOnFailure && scheduled != expected)
        {
            throw new InvalidOperationException(
                $"BF16 gradient step {stepId} scheduled {scheduled} host " +
                $"reductions; expected {expected}.");
        }
        long started = System.Diagnostics.Stopwatch.GetTimestamp();
        _hostCompletion.Wait();
        Interlocked.Add(
            ref _completeHostWaitTicks,
            System.Diagnostics.Stopwatch.GetTimestamp() - started);
        Exception? failure = Volatile.Read(ref _hostFailure);
        if (throwOnFailure && failure is not null)
        {
            throw new InvalidOperationException(
                $"BF16 gradient host reduction failed for step {stepId}.",
                failure);
        }
    }

    private void RecordBucketReady(int bucketIndex)
    {
        long now = System.Diagnostics.Stopwatch.GetTimestamp();
        Interlocked.CompareExchange(
            ref _firstBucketReadyTicks,
            now,
            comparand: 0);
        Volatile.Write(ref _lastBucketReadyTicks, now);
        int orderIndex = Interlocked.Increment(
            ref _stepBucketPublicationCount) - 1;
        if ((uint)orderIndex < (uint)_stepBucketPublicationOrder.Length)
            _stepBucketPublicationOrder[orderIndex] = bucketIndex;
    }

    private void RecordCapturedBucketOrder(int deviceSlot, int bucketIndex)
    {
        int orderIndex = Interlocked.Increment(
            ref _capturedBucketOrderCounts[deviceSlot]) - 1;
        if ((uint)orderIndex >= (uint)_buckets.Length)
        {
            throw new InvalidOperationException(
                $"CUDA device {_devices[deviceSlot]} captured more BF16 " +
                $"bucket boundaries than the {_buckets.Length} planned " +
                "buckets.");
        }
        _capturedBucketOrder[deviceSlot][orderIndex] = bucketIndex;
    }

    private void ValidateCapturedBucketOrder(
        int deviceSlot,
        int[]? capturedBucketOrder = null)
    {
        int capturedCount = capturedBucketOrder is null
            ? Volatile.Read(ref _capturedBucketOrderCounts[deviceSlot])
            : capturedBucketOrder.Length;
        if (capturedCount != _buckets.Length)
        {
            throw new InvalidOperationException(
                $"Captured CUDA device {_devices[deviceSlot]} published " +
                $"{capturedCount} BF16 bucket boundaries; expected " +
                $"{_buckets.Length}.");
        }

        int[] order = capturedBucketOrder
            ?? _capturedBucketOrder[deviceSlot];
        var seen = new bool[_buckets.Length];
        for (int orderIndex = 0; orderIndex < order.Length; orderIndex++)
        {
            int bucketIndex = order[orderIndex];
            if ((uint)bucketIndex >= (uint)_buckets.Length
                || seen[bucketIndex])
            {
                throw new InvalidOperationException(
                    $"Captured CUDA device {_devices[deviceSlot]} has an " +
                    $"invalid BF16 bucket order entry {bucketIndex} at " +
                    $"position {orderIndex}.");
            }
            seen[bucketIndex] = true;
        }
    }

    private void RecordHostWorkStarted()
    {
        long now = System.Diagnostics.Stopwatch.GetTimestamp();
        Interlocked.CompareExchange(
            ref _firstHostWorkStartedTicks,
            now,
            comparand: 0);
    }

    private void RecordHostWorkCompleted(bool succeeded)
    {
        long now = System.Diagnostics.Stopwatch.GetTimestamp();
        Interlocked.CompareExchange(
            ref _firstHostWorkCompletedTicks,
            now,
            comparand: 0);
        Volatile.Write(ref _lastHostWorkCompletedTicks, now);
        if (succeeded)
            Interlocked.Increment(ref _completedHostWorkCount);
        else
            Interlocked.Increment(ref _failedHostWorkCount);
        if (succeeded && Volatile.Read(ref _completeEnteredTicks) == 0)
            Interlocked.Increment(ref _hostWorkCompletedBeforeComplete);
    }

    private void PublishOverlapTelemetry(long stepId)
    {
        long finished = System.Diagnostics.Stopwatch.GetTimestamp();
        long started = Volatile.Read(ref _stepStartedTicks);
        long completeEntered = Volatile.Read(ref _completeEnteredTicks);
        int scheduled;
        lock (_hostCompletionSync)
            scheduled = _hostScheduledForStep;
        double? Offset(long timestamp)
            => timestamp == 0
                ? null
                : System.Diagnostics.Stopwatch.GetElapsedTime(
                    started,
                    timestamp).TotalMilliseconds;
        var telemetry = new CudaGradientOverlapTelemetry(
            stepId,
            _buckets.Length,
            scheduled,
            Volatile.Read(ref _completedHostWorkCount),
            Volatile.Read(ref _failedHostWorkCount),
            Volatile.Read(ref _hostWorkCompletedBeforeComplete),
            Offset(Volatile.Read(ref _firstBucketReadyTicks)),
            Offset(Volatile.Read(ref _lastBucketReadyTicks)),
            Offset(Volatile.Read(ref _firstHostWorkStartedTicks)),
            Offset(Volatile.Read(ref _firstHostWorkCompletedTicks)),
            Offset(Volatile.Read(ref _lastHostWorkCompletedTicks)),
            Offset(completeEntered) ?? 0d,
            Offset(finished) ?? 0d,
            Interlocked.Read(ref _completeHostWaitTicks)
                * 1000d / System.Diagnostics.Stopwatch.Frequency,
            _asyncHostPipeline,
            Volatile.Read(ref _stepUsedExternalCapturedReadyEvents) != 0,
            _stepBucketPublicationOrder
                .Take(Math.Min(
                    Volatile.Read(ref _stepBucketPublicationCount),
                    _stepBucketPublicationOrder.Length))
                .ToArray());
        Volatile.Write(ref _lastOverlapTelemetry, telemetry);
    }

    private void EnqueueBucketReduction(int bucketIndex, int destination)
    {
        Bucket bucket = _buckets[bucketIndex];
        int source = 1 - destination;
        DeviceBuffers destinationBuffers = _deviceBuffers[destination];
        BucketBuffers destinationBucket =
            destinationBuffers.Buckets[bucketIndex];
        BucketBuffers sourceBucket = _deviceBuffers[source].Buckets[bucketIndex];
        if (_useHostPipeline)
        {
            CudaGradientBuckets.HostPipelineExchange(
                destinationBuffers.Accelerator,
                _devices[source],
                _hostPipelineChunkElements,
                _hostPipelines[destination],
                destinationBucket.Local,
                sourceBucket.Local,
                destinationBucket.GradientArena.Buffer,
                bucket.TotalElements,
                destination == 0
                    ? _primarySquaredSum!.NativePtr
                    : 0,
                destinationBucket.ReadyEvent,
                sourceBucket.ReadyEvent);
        }
        else
        {
            CudaGradientBuckets.Exchange(
                destinationBuffers.Accelerator,
                destinationBuffers.DeviceIndex,
                _deviceBuffers[source].DeviceIndex,
                destinationBucket.Local,
                sourceBucket.Local,
                destinationBucket.Remote!,
                destinationBucket.GradientArena.Buffer,
                bucket.TotalElements,
                destination == 0
                    ? _primarySquaredSum!.NativePtr
                    : 0,
                destinationBuffers.CommunicationStream,
                destinationBucket.ReadyEvent,
                sourceBucket.ReadyEvent);
        }
        destinationBucket.GradientArena.MarkDirty();
    }

    private void PublishReducedBFloat16Buckets()
    {
        Parallel.For(0, _devices.Length, deviceSlot =>
        {
            DeviceBuffers buffers = _deviceBuffers[deviceSlot];
            NativeCudaDevice accelerator = buffers.Accelerator;
            accelerator.Synchronize(
                $"BF16 gradient reduction device {buffers.DeviceIndex}");
            for (int bucketIndex = 0;
                bucketIndex < _buckets.Length;
                bucketIndex++)
            {
                BucketBuffers bucket = buffers.Buckets[bucketIndex];
                CudaTensorNative.EncodeBFloat16(
                    buffers.DeviceIndex,
                    bucket.GradientArena.NativePtr,
                    bucket.Local.NativePtr,
                    _buckets[bucketIndex].TotalElements);
                bucket.LocalArena.MarkDirty();
            }
            accelerator.Synchronize(
                $"BF16 gradient publication device {buffers.DeviceIndex}");
        });
    }

    private void RecomputeBFloat16SquaredSum()
    {
        DeviceBuffers primary = _deviceBuffers[0];
        _primarySquaredSum!.MemSetToZero();
        for (int bucketIndex = 0;
            bucketIndex < _buckets.Length;
            bucketIndex++)
        {
            BucketBuffers bucket = primary.Buckets[bucketIndex];
            int length = _buckets[bucketIndex].TotalElements;
            CudaTensorNative.DecodeBFloat16(
                primary.DeviceIndex,
                bucket.Local.NativePtr,
                bucket.GradientArena.NativePtr,
                length);
            CudaTensorNative.SquaredSum(
                primary.DeviceIndex,
                bucket.GradientArena.NativePtr,
                length,
                _primarySquaredSum.NativePtr);
        }
    }

    private DeviceBuffers CreateDeviceBuffers(int deviceSlot)
    {
        int deviceIndex = _devices[deviceSlot];
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        bool ownsCommunicationStream;
        nint stream;
        if (NativeCudaRuntime.TryResolveCommunicationStream(
                deviceIndex,
                out nint borrowedStream))
        {
            if (borrowedStream == 0)
            {
                throw new InvalidOperationException(
                    "The active execution lane exposed a null " +
                    "communication stream.");
            }
            stream = borrowedStream;
            ownsCommunicationStream = false;
        }
        else
        {
            stream = CudaGradientBuckets.CreateCommunicationStream(
                accelerator, deviceIndex);
            ownsCommunicationStream = true;
        }
        var buckets = new BucketBuffers[_buckets.Length];
        try
        {
            for (int bucket = 0; bucket < _buckets.Length; bucket++)
            {
                int length = _buckets[bucket].TotalElements;
                buckets[bucket] = CreateBucketBuffers(
                    accelerator, deviceIndex, length);
            }
            return new DeviceBuffers(
                deviceIndex,
                accelerator,
                stream,
                ownsCommunicationStream,
                buckets);
        }
        catch
        {
            foreach (BucketBuffers? bucket in buckets)
                bucket?.Dispose(accelerator, deviceIndex);
            if (ownsCommunicationStream)
            {
                CudaGradientBuckets.DestroyCommunicationStream(
                    accelerator, deviceIndex, stream);
            }
            throw;
        }
    }

    private BucketBuffers CreateBucketBuffers(
        NativeCudaDevice accelerator,
        int deviceIndex,
        int length)
    {
        NativeCudaArena<ushort>? localArena = null;
        NativeCudaBuffer<ushort>? remote = null;
        NativeCudaArena<float>? gradientArena = null;
        nint readyEvent = 0;
        try
        {
            localArena = new NativeCudaArena<ushort>(accelerator, length);
            if (!_useHostPipeline)
                remote = accelerator.Allocate1D<ushort>(length);
            gradientArena = new NativeCudaArena<float>(accelerator, length);
            readyEvent = CudaGradientBuckets.CreateReadyEvent(
                accelerator, deviceIndex);
            return new BucketBuffers(
                localArena, remote, gradientArena, readyEvent);
        }
        catch
        {
            if (readyEvent != 0)
            {
                CudaGradientBuckets.DestroyEvent(
                    accelerator, deviceIndex, readyEvent);
            }
            gradientArena?.Dispose();
            remote?.Dispose();
            localArena?.Dispose();
            throw;
        }
    }

    private void BindGradientArenas()
    {
        for (int device = 0; device < _devices.Length; device++)
        {
            int deviceIndex = _devices[device];
            for (int bucket = 0; bucket < _buckets.Length; bucket++)
            {
                NativeCudaArena<float> arena =
                    _deviceBuffers[device].Buckets[bucket].GradientArena;
                foreach (Segment segment in _buckets[bucket].Segments)
                {
                    if (_usesBFloat16GradientStorage)
                    {
                        segment.Tensor.BindCudaBFloat16GradientArena(
                            deviceIndex,
                            _deviceBuffers[device].Buckets[bucket]
                                .LocalArena.Slice(
                                    segment.Offset,
                                    segment.Length));
                    }
                    else
                    {
                        segment.Tensor.BindCudaGradientArena(
                            deviceIndex,
                            arena.Slice(segment.Offset, segment.Length));
                    }
                }
            }
        }
    }

    private void UnbindGradientArenas()
    {
        for (int device = 0; device < _devices.Length; device++)
        {
            DeviceBuffers? buffers = _deviceBuffers[device];
            if (buffers is null)
                continue;
            int deviceIndex = _devices[device];
            for (int bucket = 0; bucket < _buckets.Length; bucket++)
            {
                NativeCudaArena<float> arena =
                    buffers.Buckets[bucket].GradientArena;
                foreach (Segment segment in _buckets[bucket].Segments)
                {
                    if (_usesBFloat16GradientStorage)
                    {
                        segment.Tensor.UnbindCudaBFloat16GradientArena(
                            deviceIndex,
                            buffers.Buckets[bucket].LocalArena);
                    }
                    else
                    {
                        segment.Tensor.UnbindCudaGradientArena(
                            deviceIndex,
                            arena);
                    }
                }
            }
        }
    }

    private static int ResolveTargetBucketElements(
        bool peerAccess,
        CudaDispatchPolicy dispatch)
        => dispatch.GradientBucketElements
            ?? (peerAccess
                ? TargetBucketElements
                : NonPeerTargetBucketElements);

    private static int ResolveHostPipelineChunkElements(
        CudaDispatchPolicy dispatch)
        => dispatch.GradientHostChunkElements
            ?? DefaultHostPipelineChunkElements;

    private static Bucket[] BuildBuckets(
        IReadOnlyList<Parameter> parameters,
        int targetBucketElements)
    {
        var buckets = new List<Bucket>();
        var segments = new List<Segment>();
        int offset = 0;
        void FinishBucket()
        {
            if (segments.Count == 0)
                return;
            buckets.Add(new Bucket(segments.ToArray(), offset));
            segments.Clear();
            offset = 0;
        }

        foreach (Parameter parameter in parameters)
        {
            int length = parameter.T.Numel;
            if (segments.Count > 0
                && offset + (long)length > targetBucketElements)
            {
                FinishBucket();
            }
            segments.Add(new Segment(parameter.T, offset, length));
            offset = checked(offset + length);
            if (offset >= targetBucketElements)
                FinishBucket();
        }
        FinishBucket();
        return buckets.ToArray();
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        var releases = new List<Action>();
        long activeStep = Volatile.Read(ref _activeStepId);
        if (activeStep != 0)
            releases.Add(() => Abort(activeStep));
        foreach (Parameter parameter in _parameters)
        {
            releases.Add(() => parameter.T.UnregisterCudaGradientReducer(
                _reducerGeneration));
        }
        releases.Add(UnbindGradientArenas);
        foreach (HostReductionWorker? worker in _hostWorkers)
        {
            if (worker is not null)
                releases.Add(worker.Dispose);
        }
        for (int destination = 0;
            destination < _hostPipelines.Length;
            destination++)
        {
            int capturedDestination = destination;
            DeviceBuffers? buffers = _deviceBuffers[destination];
            NativeCudaDevice accelerator = buffers?.Accelerator
                ?? ForgetMemoryV2Cuda.GetAccelerator(_devices[destination]);
            releases.Add(() => CudaGradientBuckets.DestroyHostPipeline(
                accelerator,
                _hostPipelines[capturedDestination]));
        }
        foreach (DeviceBuffers? device in _deviceBuffers)
        {
            if (device is not null)
                releases.Add(device.Dispose);
        }
        NativeCudaBuffer<double>? primarySquaredSum = _primarySquaredSum;
        if (primarySquaredSum is not null)
        {
            releases.Add(() =>
            {
                primarySquaredSum.Dispose();
                _primarySquaredSum = null;
            });
        }
        releases.Add(_hostCompletion.Dispose);
        CudaResourceCleanup.RunAll(
            "BF16 gradient reduction cleanup failed.",
            releases);
    }

    private sealed record Bucket(Segment[] Segments, int TotalElements);
    private sealed record Segment(Tensor Tensor, int Offset, int Length);
    private readonly record struct SegmentLocation(int Bucket, int Segment);
    private readonly record struct HostReductionWork(
        long StepId,
        int BucketIndex,
        DeviceTransferGuard.SharedContext? TransferContext);

    private sealed class HostReductionWorker : IDisposable
    {
        private static int _activeCount;
        private readonly ConcurrentQueue<HostReductionWork> _queue = new();
        private readonly AutoResetEvent _available = new(false);
        private readonly Action<HostReductionWork> _execute;
        private readonly Thread _thread;
        private int _disposed;

        internal static int ActiveCount => Volatile.Read(ref _activeCount);

        internal HostReductionWorker(
            string name,
            Action<HostReductionWork> execute)
        {
            _execute = execute;
            _thread = new Thread(Run)
            {
                IsBackground = true,
                Name = name,
            };
            _thread.Start();
        }

        internal void Enqueue(HostReductionWork work)
        {
            ObjectDisposedException.ThrowIf(
                Volatile.Read(ref _disposed) != 0, this);
            _queue.Enqueue(work);
            _available.Set();
        }

        private void Run()
        {
            Interlocked.Increment(ref _activeCount);
            try
            {
                while (true)
                {
                    _available.WaitOne();
                    while (_queue.TryDequeue(out HostReductionWork work))
                        _execute(work);
                    if (Volatile.Read(ref _disposed) != 0 && _queue.IsEmpty)
                        return;
                }
            }
            finally
            {
                Interlocked.Decrement(ref _activeCount);
            }
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            _available.Set();
            _thread.Join();
            _available.Dispose();
        }
    }

    private sealed class BucketBuffers(
        NativeCudaArena<ushort> localArena,
        NativeCudaBuffer<ushort>? remote,
        NativeCudaArena<float> gradientArena,
        nint readyEvent)
    {
        internal NativeCudaArena<ushort> LocalArena { get; } = localArena;
        internal NativeCudaBuffer<ushort> Local => LocalArena.Buffer;
        internal NativeCudaBuffer<ushort>? Remote { get; } = remote;
        internal NativeCudaArena<float> GradientArena { get; } = gradientArena;
        internal nint ReadyEvent { get; } = readyEvent;

        internal void Dispose(
            NativeCudaDevice accelerator,
            int deviceIndex)
        {
            CudaGradientBuckets.DestroyEvent(
                accelerator, deviceIndex, ReadyEvent);
            LocalArena.Dispose();
            Remote?.Dispose();
            GradientArena.Dispose();
        }
    }

    private sealed class DeviceBuffers(
        int deviceIndex,
        NativeCudaDevice accelerator,
        nint communicationStream,
        bool ownsCommunicationStream,
        BucketBuffers[] buckets) : IDisposable
    {
        internal int DeviceIndex { get; } = deviceIndex;
        internal NativeCudaDevice Accelerator { get; } = accelerator;
        internal nint CommunicationStream { get; } = communicationStream;
        internal bool OwnsCommunicationStream { get; } =
            ownsCommunicationStream;
        internal BucketBuffers[] Buckets { get; } = buckets;

        public void Dispose()
        {
            CudaGradientBuckets.Synchronize(
                Accelerator, DeviceIndex, CommunicationStream);
            foreach (BucketBuffers bucket in Buckets)
                bucket.Dispose(Accelerator, DeviceIndex);
            if (OwnsCommunicationStream)
            {
                CudaGradientBuckets.DestroyCommunicationStream(
                    Accelerator, DeviceIndex, CommunicationStream);
            }
        }
    }
}

internal interface ICudaGradientReductionPlan
{
    void NotifyGradientReady(Tensor tensor, int deviceIndex, long stepId);
}

internal static class CudaGradientReductionContext
{
    private static readonly AsyncLocal<Entry?> Current = new();

    internal static bool HasActivePlan => Current.Value is not null;

    internal static IDisposable Push(
        ICudaGradientReductionPlan plan,
        int deviceIndex,
        long stepId)
    {
        Entry? previous = Current.Value;
        Current.Value = new Entry(plan, deviceIndex, stepId);
        return new Scope(previous);
    }

    internal static void NotifyLeaf(Tensor tensor)
    {
        Entry? current = Current.Value;
        if (current is not null)
        {
            current.Plan.NotifyGradientReady(
                tensor, current.DeviceIndex, current.StepId);
            return;
        }

        if (CudaBfp8GradientPublicationScope.TryPublish(tensor))
            return;

        // Pure BFP8 keeps Float32 only as the in-flight backward
        // accumulator. With no data-parallel reducer installed, leaf
        // completion is the publication boundary.
        PrecisionPolicy? policy = TensorExecutionContext.ActivePrecisionPolicy;
        bool pureBfp8 = policy?.Gradient == NumericFormat.Bfp8
            || (policy is null
                && tensor.Bfp8Quantization?.Granularity
                    == Bfp8ScaleGranularity.Tensor);
        if (Tensor.ExecutionDevice == TensorDevice.Cuda
            && tensor.DType == TensorDType.Bfp8
            && pureBfp8
            && tensor.HasGradientBuffer)
        {
            tensor.PublishCudaBfp8Gradient(Tensor.CudaDeviceIndex);
        }
    }

    private sealed record Entry(
        ICudaGradientReductionPlan Plan,
        int DeviceIndex,
        long StepId);

    private sealed class Scope(Entry? previous) : IDisposable
    {
        private int _disposed;

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            Current.Value = previous;
        }
    }
}
