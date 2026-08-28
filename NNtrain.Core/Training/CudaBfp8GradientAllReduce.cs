namespace NNtrain;

/// <summary>
/// Scale-aware, GPU-only two-device reducer for pure BFP8 gradients. Local
/// backward accumulation remains Float32 until a leaf is complete. Each leaf
/// is then tensor-wide quantized, exchanged on communication streams, summed
/// in Float32 on the primary, requantized once, and broadcast as BFP8.
/// </summary>
internal sealed class CudaBfp8GradientAllReducePlan
    : ICudaGradientReductionPlan, IDisposable
{
    private readonly Parameter[] _parameters;
    private readonly int[] _devices;
    private readonly Entry[] _entries;
    private readonly Dictionary<Tensor, int> _locations;
    private readonly long[][] _notificationSteps;
    private readonly int[] _readyDeviceCounts;
    private readonly long[] _deviceBeginSteps;
    private readonly CudaCapturedGradientPublicationState
        _capturedPublicationState;
    private readonly NativeCudaDevice[] _accelerators;
    private readonly nint[] _communicationStreams;
    private readonly bool[] _ownsCommunicationStreams;
    private readonly NativeCudaBuffer<int>[] _finiteStatus;
    private readonly NativeCudaBuffer<int> _remoteStatusStaging;
    private readonly NativeCudaBuffer<double> _primarySquaredSum;
    private readonly long _reducerGeneration =
        CudaGradientReductionStampSource.CreateReducerGeneration();
    private long _stepSequence;
    private long _activeStepId;
    private long _completedSteps;
    private long _lastCompletedTransportBytes;
    private long _managedLocalQuantizationSubmissionCount;
    private long _capturedReplayReadyEventRecordCount;
    private long _capturedReplayReadyEventRecordTicks;
    private int _disposed;

    internal CudaBfp8GradientAllReducePlan(
        IReadOnlyList<Parameter> parameters,
        IReadOnlyList<int> devices)
    {
        ArgumentNullException.ThrowIfNull(parameters);
        ArgumentNullException.ThrowIfNull(devices);
        if (devices.Count != 2)
        {
            throw new ArgumentException(
                "Pure BFP8 asynchronous reduction requires exactly two GPUs.",
                nameof(devices));
        }
        _parameters = parameters.ToArray();
        _devices = devices.ToArray();
        if (_parameters.Any(parameter =>
                parameter.T.DType != TensorDType.Bfp8
                || parameter.T.Bfp8Quantization?.Granularity
                    != Bfp8ScaleGranularity.Tensor))
        {
            throw new ArgumentException(
                "Every pure BFP8 parameter must use one tensor-wide scale.",
                nameof(parameters));
        }

        _locations = new Dictionary<Tensor, int>(
            ReferenceEqualityComparer.Instance);
        for (int index = 0; index < _parameters.Length; index++)
            _locations.Add(_parameters[index].T, index);
        _notificationSteps = Enumerable.Range(0, _parameters.Length)
            .Select(_ => new long[_devices.Length])
            .ToArray();
        _readyDeviceCounts = new int[_parameters.Length];
        _deviceBeginSteps = new long[_devices.Length];
        _capturedPublicationState =
            new CudaCapturedGradientPublicationState(_devices.Length);
        _accelerators = new NativeCudaDevice[_devices.Length];
        _communicationStreams = new nint[_devices.Length];
        _ownsCommunicationStreams = new bool[_devices.Length];
        _finiteStatus = new NativeCudaBuffer<int>[_devices.Length];
        _entries = new Entry[_parameters.Length];

        var owned = new Stack<IDisposable>();
        try
        {
            for (int device = 0; device < _devices.Length; device++)
            {
                int deviceIndex = _devices[device];
                NativeCudaDevice accelerator =
                    ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
                _accelerators[device] = accelerator;
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
                    _communicationStreams[device] = borrowedStream;
                    _ownsCommunicationStreams[device] = false;
                }
                else
                {
                    _communicationStreams[device] =
                        CudaGradientBuckets.CreateCommunicationStream(
                            accelerator, deviceIndex);
                    _ownsCommunicationStreams[device] = true;
                    var streamLease = new CommunicationStreamLease(
                        accelerator,
                        deviceIndex,
                        _communicationStreams[device]);
                    owned.Push(streamLease);
                }
                _finiteStatus[device] = accelerator.Allocate1D<int>(1);
                owned.Push(_finiteStatus[device]);
            }

            _remoteStatusStaging =
                _accelerators[0].Allocate1D<int>(1);
            owned.Push(_remoteStatusStaging);
            _primarySquaredSum =
                _accelerators[0].Allocate1D<double>(1);
            owned.Push(_primarySquaredSum);

            for (int index = 0; index < _entries.Length; index++)
            {
                Tensor tensor = _parameters[index].T;
                CudaBfp8BufferView primary =
                    tensor.PrepareCudaBfp8GradientReplica(_devices[0]);
                CudaBfp8BufferView secondary =
                    tensor.PrepareCudaBfp8GradientReplica(_devices[1]);
                NativeCudaBuffer<sbyte> remotePayload =
                    _accelerators[0].Allocate1D<sbyte>(tensor.Numel);
                owned.Push(remotePayload);
                NativeCudaBuffer<float> remoteScale =
                    _accelerators[0].Allocate1D<float>(1);
                owned.Push(remoteScale);
                nint primaryReady = CudaGradientBuckets.CreateReadyEvent(
                    _accelerators[0], _devices[0]);
                var primaryEventLease = new EventLease(
                    _accelerators[0], _devices[0], primaryReady);
                owned.Push(primaryEventLease);
                nint secondaryReady = CudaGradientBuckets.CreateReadyEvent(
                    _accelerators[1], _devices[1]);
                var secondaryEventLease = new EventLease(
                    _accelerators[1], _devices[1], secondaryReady);
                owned.Push(secondaryEventLease);
                nint reducedReady = CudaGradientBuckets.CreateReadyEvent(
                    _accelerators[0], _devices[0]);
                var reducedEventLease = new EventLease(
                    _accelerators[0], _devices[0], reducedReady);
                owned.Push(reducedEventLease);

                _entries[index] = new Entry(
                    tensor,
                    [primary, secondary],
                    remotePayload,
                    remoteScale,
                    [primaryEventLease, secondaryEventLease],
                    reducedEventLease);
            }
            foreach (Parameter parameter in _parameters)
            {
                parameter.T.RegisterCudaGradientReducer(
                    _reducerGeneration,
                    _devices,
                    ownsGradientZeroing: true);
            }
            owned.Clear();
        }
        catch
        {
            while (owned.TryPop(out IDisposable? resource))
            {
                try
                {
                    resource.Dispose();
                }
                catch
                {
                    // Preserve the allocation failure. Every later lease is
                    // still attempted so one cleanup error cannot strand VRAM.
                }
            }
            throw;
        }
    }

    internal bool Matches(
        IReadOnlyList<Parameter> parameters,
        IReadOnlyList<int> devices)
        => devices.SequenceEqual(_devices)
            && parameters.Count == _parameters.Length
            && parameters.Select((parameter, index) =>
                ReferenceEquals(parameter, _parameters[index])).All(value => value);

    internal bool DefersExchangeUntilBackward => false;

    internal long TransportBytesPerStep => checked(
        2L * _entries.Sum(entry =>
            (long)entry.Tensor.Numel * sizeof(sbyte) + sizeof(float)));

    internal long LastCompletedTransportBytes
        => Volatile.Read(ref _lastCompletedTransportBytes);

    internal long CompletedSteps => Volatile.Read(ref _completedSteps);

    /// <summary>
    /// Managed submissions of the local FP32-to-BFP8 quantization operation.
    /// Captured replay executes the native graph node directly, so state-only
    /// publication must leave this counter unchanged.
    /// </summary>
    internal long ManagedLocalQuantizationSubmissionCount
        => Interlocked.Read(ref _managedLocalQuantizationSubmissionCount);

    internal long CapturedReplayReadyEventRecordCount
        => Interlocked.Read(ref _capturedReplayReadyEventRecordCount);

    internal double CapturedReplayReadyEventRecordMilliseconds
        => Interlocked.Read(ref _capturedReplayReadyEventRecordTicks)
            * 1000d / System.Diagnostics.Stopwatch.Frequency;

    internal bool OwnsCommunicationStream(int deviceIndex)
        => _ownsCommunicationStreams[GetDeviceSlot(deviceIndex)];

    internal long BeginStep()
    {
        ObjectDisposedException.ThrowIf(
            Volatile.Read(ref _disposed) != 0, this);
        if (Volatile.Read(ref _activeStepId) != 0)
        {
            throw new InvalidOperationException(
                "The previous BFP8 gradient reduction step is still active.");
        }
        long stepId = Interlocked.Increment(ref _stepSequence);
        if (stepId == 0)
            stepId = Interlocked.Increment(ref _stepSequence);
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
        Array.Clear(_readyDeviceCounts);
        foreach (long[] notifications in _notificationSteps)
            Array.Clear(notifications);
        Volatile.Write(ref _activeStepId, stepId);
        return stepId;
    }

    internal void BeginDeviceStep(long stepId, int deviceIndex)
    {
        RequireActiveStep(stepId);
        int deviceSlot = GetDeviceSlot(deviceIndex);
        if (Interlocked.Exchange(
                ref _deviceBeginSteps[deviceSlot], stepId) == stepId)
        {
            throw new InvalidOperationException(
                $"CUDA device {deviceIndex} was begun twice for BFP8 step " +
                $"{stepId}.");
        }

        _finiteStatus[deviceSlot].MemSetToZero();
        if (deviceSlot == 0)
            _primarySquaredSum.MemSetToZero();
        foreach (Entry entry in _entries)
        {
            NativeCudaBuffer<float> gradient =
                entry.Tensor.PrepareReducerOwnedCudaGradientBuffer(
                    _reducerGeneration,
                    deviceIndex);
            gradient.MemSetToZero();
            entry.Tensor.CompleteReducerOwnedCudaGradientZero(
                _reducerGeneration,
                deviceIndex);
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
        RequireActiveStep(stepId);
        if (!_locations.TryGetValue(tensor, out int parameterIndex))
            return;
        int deviceSlot = GetDeviceSlot(deviceIndex);
        if (Volatile.Read(ref _deviceBeginSteps[deviceSlot]) != stepId)
        {
            throw new InvalidOperationException(
                $"CUDA device {deviceIndex} was not begun for BFP8 step " +
                $"{stepId}.");
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
        if (Interlocked.CompareExchange(
                ref _notificationSteps[parameterIndex][deviceSlot],
                stepId,
                comparand: 0) != 0)
        {
            throw new InvalidOperationException(
                $"BFP8 gradient '{tensor.Name}' was notified twice on " +
                $"CUDA device {deviceIndex} for step {stepId}.");
        }

        Entry entry = _entries[parameterIndex];
        NativeCudaDevice accelerator = _accelerators[deviceSlot];
        NativeCudaBuffer<float> gradient =
            tensor.EnsureCudaGradientBuffer(deviceIndex);
        CudaBfp8GradientNative.Quantize(
            deviceIndex,
            gradient,
            entry.Replicas[deviceSlot],
            _finiteStatus[deviceSlot],
            accelerator.DefaultStream);
        Interlocked.Increment(
            ref _managedLocalQuantizationSubmissionCount);
        CudaGradientBuckets.RecordReady(
            deviceIndex,
            accelerator,
            entry.ReadyEvents[deviceSlot].Handle);

        if (captureRecording)
            return;

        if (Interlocked.Increment(
                ref _readyDeviceCounts[parameterIndex]) == _devices.Length)
        {
            try
            {
                EnqueueReduction(entry);
            }
            finally
            {
                accelerator.Bind();
            }
        }
    }

    void ICudaGradientReductionPlan.NotifyGradientReady(
        Tensor tensor,
        int deviceIndex,
        long stepId)
        => NotifyGradientReady(tensor, deviceIndex, stepId);

    internal IDisposable BeginCapturedBackwardRecording(
        long stepId,
        int deviceIndex)
    {
        RequireActiveStep(stepId);
        int deviceSlot = GetDeviceSlot(deviceIndex);
        if (Volatile.Read(ref _deviceBeginSteps[deviceSlot]) != stepId)
        {
            throw new InvalidOperationException(
                $"CUDA device {deviceIndex} was not begun for BFP8 step " +
                $"{stepId}.");
        }
        _capturedPublicationState.BeginCaptureRecording(
            stepId,
            deviceSlot,
            deviceIndex);
        return new CudaCapturedBackwardRecordingScope(() =>
            _capturedPublicationState.EndCaptureRecording(
                stepId,
                deviceSlot,
                deviceIndex));
    }

    internal void DiscardCapturedBackwardRecordingStep(long stepId)
    {
        RequireActiveStep(stepId);
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
    /// Publishes only managed readiness after a captured backward replay.
    /// Quantization and ready-event records were captured with the backward
    /// graph and therefore must not be submitted again here.
    /// </summary>
    internal void PublishCapturedDeviceGradients(
        long stepId,
        int deviceIndex)
        => PublishCapturedDeviceGradientsCore(
            stepId,
            deviceIndex,
            recordReadyEventsAfterReplay: false);

    internal void PublishCapturedDeviceGradientsAfterReplay(
        long stepId,
        int deviceIndex)
        => PublishCapturedDeviceGradientsCore(
            stepId,
            deviceIndex,
            recordReadyEventsAfterReplay: true);

    private void PublishCapturedDeviceGradientsCore(
        long stepId,
        int deviceIndex,
        bool recordReadyEventsAfterReplay)
    {
        RequireActiveStep(stepId);
        int deviceSlot = GetDeviceSlot(deviceIndex);
        if (Volatile.Read(ref _deviceBeginSteps[deviceSlot]) != stepId)
        {
            throw new InvalidOperationException(
                $"CUDA device {deviceIndex} was not begun for BFP8 step " +
                $"{stepId}.");
        }

        _capturedPublicationState.BeginCapturedPublication(
            stepId,
            deviceSlot,
            deviceIndex);
        try
        {
            for (int parameterIndex = 0;
                parameterIndex < _entries.Length;
                parameterIndex++)
            {
                if (Volatile.Read(
                        ref _notificationSteps[parameterIndex][deviceSlot])
                    != 0)
                {
                    throw new InvalidOperationException(
                        $"Captured BFP8 gradient '{_entries[parameterIndex]
                            .Tensor.Name}' was already partially published " +
                    $"on CUDA device {deviceIndex} for step {stepId}.");
                }
            }

            NativeCudaDevice accelerator = _accelerators[deviceSlot];
            if (recordReadyEventsAfterReplay)
            {
                long started = System.Diagnostics.Stopwatch.GetTimestamp();
                long recorded = 0;
                try
                {
                    for (int parameterIndex = 0;
                        parameterIndex < _entries.Length;
                        parameterIndex++)
                    {
                        CudaGradientBuckets.RecordReady(
                            deviceIndex,
                            accelerator,
                            _entries[parameterIndex]
                                .ReadyEvents[deviceSlot].Handle);
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
            for (int parameterIndex = 0;
                parameterIndex < _entries.Length;
                parameterIndex++)
            {
                if (Interlocked.CompareExchange(
                        ref _notificationSteps[parameterIndex][deviceSlot],
                        stepId,
                        comparand: 0) != 0)
                {
                    throw new InvalidOperationException(
                        $"Captured BFP8 gradient '{_entries[parameterIndex]
                            .Tensor.Name}' changed while CUDA device " +
                        $"{deviceIndex} was publishing step {stepId}.");
                }
                if (Interlocked.Increment(
                        ref _readyDeviceCounts[parameterIndex])
                    == _devices.Length)
                {
                    try
                    {
                        EnqueueReduction(_entries[parameterIndex]);
                    }
                    finally
                    {
                        accelerator.Bind();
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
        RequireActiveStep(stepId);
        var stamp = new CudaGradientReductionStamp(
            _reducerGeneration, stepId);
        bool publish = false;
        try
        {
            _capturedPublicationState.ValidateComplete(stepId);
            for (int parameter = 0; parameter < _entries.Length; parameter++)
            {
                if (Volatile.Read(ref _readyDeviceCounts[parameter])
                    != _devices.Length)
                {
                    throw new InvalidOperationException(
                        $"BFP8 gradient '{_entries[parameter].Tensor.Name}' " +
                        $"was incomplete for step {stepId}.");
                }
            }
            for (int device = 0; device < _devices.Length; device++)
            {
                if (Volatile.Read(ref _deviceBeginSteps[device]) != stepId)
                {
                    throw new InvalidOperationException(
                        $"CUDA device {_devices[device]} was not begun for " +
                        $"BFP8 step {stepId}.");
                }
            }

            // Primary completion fences every reduce; secondary completion
            // fences every event-dependent broadcast. No workspace is reused
            // before both fences pass.
            CudaGradientBuckets.Synchronize(
                _accelerators[0], _devices[0], _communicationStreams[0]);
            CudaGradientBuckets.Synchronize(
                _accelerators[1], _devices[1], _communicationStreams[1]);

            var finite = new int[1];
            for (int device = 0; device < _devices.Length; device++)
            {
                _finiteStatus[device].CopyToCPU(finite);
                if (finite[0] != 0)
                {
                    throw new InvalidOperationException(
                        $"Non-finite CUDA gradient detected before BFP8 " +
                        $"publication on device {_devices[device]} at step " +
                        $"{stepId}.");
                }
            }

            foreach (Parameter parameter in _parameters)
            {
                parameter.T.MarkCudaBfp8GradientsSynchronized(
                    _devices, stamp);
            }
            var squaredSum = new double[1];
            _primarySquaredSum.CopyToCPU(squaredSum);
            TensorCudaKernels.PublishGradientSquaredSum(
                _parameters, _devices, squaredSum[0]);
            Volatile.Write(
                ref _lastCompletedTransportBytes,
                TransportBytesPerStep);
            Interlocked.Increment(ref _completedSteps);
            publish = true;
        }
        finally
        {
            Volatile.Write(ref _activeStepId, 0);
            _capturedPublicationState.EndStep(stepId);
            if (!publish)
            {
                // A failed generation is never allowed to masquerade as a
                // completed BFP8 replica on the following optimizer step.
                foreach (int device in _devices)
                    _accelerators[Array.IndexOf(_devices, device)].Bind();
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
        for (int device = 0; device < _devices.Length; device++)
        {
            try
            {
                CudaGradientBuckets.Synchronize(
                    _accelerators[device],
                    _devices[device],
                    _communicationStreams[device]);
            }
            catch (Exception exception)
            {
                (failures ??= []).Add(exception);
            }
        }
        Volatile.Write(ref _activeStepId, 0);
        _capturedPublicationState.EndStep(stepId);
        foreach (Parameter parameter in _parameters)
            parameter.T.AbortCudaGradientReduction(stamp);
        if (failures is not null)
        {
            throw new AggregateException(
                "BFP8 gradient reduction abort failed.", failures);
        }
    }

    private void EnqueueReduction(Entry entry)
    {
        NativeCudaBuffer<float> primaryFloat =
            entry.Tensor.EnsureCudaGradientBuffer(_devices[0]);
        NativeCudaBuffer<float> secondaryFloat =
            entry.Tensor.EnsureCudaGradientBuffer(_devices[1]);
        CudaBfp8GradientNative.Reduce(
            _devices[0],
            _devices[1],
            entry.Replicas[0],
            entry.Replicas[1],
            entry.RemotePayload,
            entry.RemoteScale,
            primaryFloat,
            entry.Replicas[0],
            reductionScale: 1f,
            _finiteStatus[0],
            _finiteStatus[1],
            _remoteStatusStaging,
            _primarySquaredSum,
            _communicationStreams[0],
            entry.ReadyEvents[0].Handle,
            entry.ReadyEvents[1].Handle,
            entry.ReducedEvent.Handle);
        CudaBfp8GradientNative.Broadcast(
            _devices[1],
            _devices[0],
            entry.Replicas[0],
            entry.Replicas[1],
            secondaryFloat,
            _finiteStatus[1],
            _communicationStreams[1],
            entry.ReducedEvent.Handle);
    }

    private int GetDeviceSlot(int deviceIndex)
    {
        int slot = Array.IndexOf(_devices, deviceIndex);
        return slot >= 0
            ? slot
            : throw new ArgumentOutOfRangeException(nameof(deviceIndex));
    }

    private void RequireActiveStep(long stepId)
    {
        ObjectDisposedException.ThrowIf(
            Volatile.Read(ref _disposed) != 0, this);
        if (Volatile.Read(ref _activeStepId) != stepId || stepId == 0)
        {
            throw new InvalidOperationException(
                $"BFP8 gradient step {stepId} is not active.");
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        long activeStep = Volatile.Read(ref _activeStepId);
        if (activeStep != 0)
            Abort(activeStep);
        foreach (Parameter parameter in _parameters)
            parameter.T.UnregisterCudaGradientReducer(_reducerGeneration);
        List<Exception>? failures = null;
        void TryDispose(IDisposable? resource)
        {
            if (resource is null)
                return;
            try
            {
                resource.Dispose();
            }
            catch (Exception exception)
            {
                (failures ??= []).Add(exception);
            }
        }

        for (int device = 0; device < _devices.Length; device++)
        {
            try
            {
                CudaGradientBuckets.Synchronize(
                    _accelerators[device],
                    _devices[device],
                    _communicationStreams[device]);
            }
            catch (Exception exception)
            {
                (failures ??= []).Add(exception);
            }
        }
        foreach (Entry? entry in _entries)
            TryDispose(entry);
        TryDispose(_remoteStatusStaging);
        TryDispose(_primarySquaredSum);
        foreach (NativeCudaBuffer<int>? status in _finiteStatus)
            TryDispose(status);
        for (int device = 0; device < _communicationStreams.Length; device++)
        {
            if (!_ownsCommunicationStreams[device])
                continue;
            try
            {
                CudaGradientBuckets.DestroyCommunicationStream(
                    _accelerators[device],
                    _devices[device],
                    _communicationStreams[device]);
            }
            catch (Exception exception)
            {
                (failures ??= []).Add(exception);
            }
        }
        Volatile.Write(ref _activeStepId, 0);
        if (failures is not null)
        {
            throw new AggregateException(
                "One or more BFP8 reducer resources failed to dispose.",
                failures);
        }
    }

    private sealed class Entry(
        Tensor tensor,
        CudaBfp8BufferView[] replicas,
        NativeCudaBuffer<sbyte> remotePayload,
        NativeCudaBuffer<float> remoteScale,
        EventLease[] readyEvents,
        EventLease reducedEvent) : IDisposable
    {
        internal Tensor Tensor { get; } = tensor;
        internal CudaBfp8BufferView[] Replicas { get; } = replicas;
        internal NativeCudaBuffer<sbyte> RemotePayload { get; } = remotePayload;
        internal NativeCudaBuffer<float> RemoteScale { get; } = remoteScale;
        internal EventLease[] ReadyEvents { get; } = readyEvents;
        internal EventLease ReducedEvent { get; } = reducedEvent;

        public void Dispose()
        {
            List<Exception>? failures = null;
            foreach (IDisposable resource in ReadyEvents.Cast<IDisposable>()
                .Append(ReducedEvent)
                .Append(RemotePayload)
                .Append(RemoteScale))
            {
                try
                {
                    resource.Dispose();
                }
                catch (Exception exception)
                {
                    (failures ??= []).Add(exception);
                }
            }
            if (failures is not null)
                throw new AggregateException("BFP8 reducer entry cleanup failed.", failures);
        }
    }

    private sealed class EventLease(
        NativeCudaDevice accelerator,
        int deviceIndex,
        nint handle) : IDisposable
    {
        private int _disposed;
        internal nint Handle { get; } = handle;

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) == 0)
            {
                CudaGradientBuckets.DestroyEvent(
                    accelerator, deviceIndex, Handle);
            }
        }
    }

    private sealed class CommunicationStreamLease(
        NativeCudaDevice accelerator,
        int deviceIndex,
        nint stream) : IDisposable
    {
        private int _disposed;
        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) == 0)
            {
                CudaGradientBuckets.DestroyCommunicationStream(
                    accelerator, deviceIndex, stream);
            }
        }
    }
}
