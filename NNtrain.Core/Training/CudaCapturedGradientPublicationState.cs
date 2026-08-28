namespace NNtrain;

/// <summary>
/// Step-local arbitration between leaf-by-leaf autograd notification and a
/// CUDA Graph replay that publishes one whole device after captured backward.
/// A step selects exactly one path; captured publication is exactly once per
/// begun device and becomes permanently failed until Abort/Complete ends it.
/// </summary>
internal sealed class CudaCapturedGradientPublicationState
{
    private const int PathNone = 0;
    private const int PathNotifications = 1;
    private const int PathCaptured = 2;
    private const int PathCaptureRecording = 3;
    private const int DeviceNone = 0;
    private const int DevicePublishing = 1;
    private const int DevicePublished = 2;
    private const int DeviceFailed = 3;

    private readonly long[] _deviceBeginSteps;
    private readonly int[] _capturedDeviceStates;
    private readonly int[] _captureRecordingStates;
    private long _activeStepId;
    private int _path;

    internal CudaCapturedGradientPublicationState(int deviceCount)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(deviceCount);
        _deviceBeginSteps = new long[deviceCount];
        _capturedDeviceStates = new int[deviceCount];
        _captureRecordingStates = new int[deviceCount];
    }

    internal void BeginStep(long stepId)
    {
        if (stepId == 0)
            throw new ArgumentOutOfRangeException(nameof(stepId));
        if (Volatile.Read(ref _activeStepId) != 0)
        {
            throw new InvalidOperationException(
                "The previous captured-gradient publication step is still active.");
        }
        Array.Clear(_deviceBeginSteps);
        Array.Clear(_capturedDeviceStates);
        Array.Clear(_captureRecordingStates);
        Volatile.Write(ref _path, PathNone);
        Volatile.Write(ref _activeStepId, stepId);
    }

    internal void MarkDeviceBegun(
        long stepId,
        int deviceSlot,
        int deviceIndex)
    {
        RequireActive(stepId);
        ValidateDeviceSlot(deviceSlot);
        if (Interlocked.CompareExchange(
                ref _deviceBeginSteps[deviceSlot],
                stepId,
                comparand: 0) != 0)
        {
            throw new InvalidOperationException(
                $"CUDA device {deviceIndex} was begun twice for captured " +
                $"gradient step {stepId}.");
        }
    }

    internal void EnterNotificationPath(
        long stepId,
        int deviceSlot,
        int deviceIndex)
    {
        RequireBegun(stepId, deviceSlot, deviceIndex);
        ClaimPath(
            PathNotifications,
            "leaf notification",
            "captured device publication");
    }

    internal void BeginCaptureRecording(
        long stepId,
        int deviceSlot,
        int deviceIndex)
    {
        RequireBegun(stepId, deviceSlot, deviceIndex);
        ClaimPath(
            PathCaptureRecording,
            "captured backward recording",
            "normal or replay publication");
        if (Interlocked.CompareExchange(
                ref _captureRecordingStates[deviceSlot],
                DevicePublishing,
                DeviceNone) != DeviceNone)
        {
            throw new InvalidOperationException(
                $"CUDA device {deviceIndex} began captured backward " +
                $"recording twice for step {stepId}.");
        }
    }

    internal bool IsCaptureRecording(long stepId, int deviceSlot)
        => Volatile.Read(ref _activeStepId) == stepId
            && (uint)deviceSlot < (uint)_captureRecordingStates.Length
            && Volatile.Read(ref _path) == PathCaptureRecording
            && Volatile.Read(ref _captureRecordingStates[deviceSlot])
                == DevicePublishing;

    internal void EnterCaptureNotificationPath(
        long stepId,
        int deviceSlot,
        int deviceIndex)
    {
        RequireBegun(stepId, deviceSlot, deviceIndex);
        if (!IsCaptureRecording(stepId, deviceSlot))
        {
            throw new InvalidOperationException(
                $"CUDA device {deviceIndex} is not recording captured " +
                $"backward gradients for step {stepId}.");
        }
    }

    internal void EndCaptureRecording(
        long stepId,
        int deviceSlot,
        int deviceIndex)
    {
        RequireActive(stepId);
        ValidateDeviceSlot(deviceSlot);
        if (Interlocked.CompareExchange(
                ref _captureRecordingStates[deviceSlot],
                DevicePublished,
                DevicePublishing) != DevicePublishing)
        {
            throw new InvalidOperationException(
                $"CUDA device {deviceIndex} has no captured backward " +
                $"recording to end for step {stepId}.");
        }
    }

    internal void ValidateCaptureDiscard(long stepId)
    {
        RequireActive(stepId);
        if (Volatile.Read(ref _path) != PathCaptureRecording)
        {
            throw new InvalidOperationException(
                $"Gradient step {stepId} is not a captured backward " +
                "recording generation.");
        }
        bool recorded = false;
        for (int slot = 0; slot < _captureRecordingStates.Length; slot++)
        {
            int state = Volatile.Read(ref _captureRecordingStates[slot]);
            if (state == DevicePublishing)
            {
                throw new InvalidOperationException(
                    $"Captured backward recording is still active on " +
                    $"device slot {slot} for step {stepId}.");
            }
            recorded |= state == DevicePublished;
        }
        if (!recorded)
        {
            throw new InvalidOperationException(
                $"Captured backward recording step {stepId} recorded no " +
                "device graph.");
        }
    }

    internal void BeginCapturedPublication(
        long stepId,
        int deviceSlot,
        int deviceIndex)
    {
        RequireBegun(stepId, deviceSlot, deviceIndex);
        ClaimPath(
            PathCaptured,
            "captured device publication",
            "leaf notification");
        for (int slot = 0; slot < _capturedDeviceStates.Length; slot++)
        {
            if (Volatile.Read(ref _capturedDeviceStates[slot])
                == DeviceFailed)
            {
                throw new InvalidOperationException(
                    $"Captured gradient step {stepId} already failed and " +
                    "must be aborted before reuse.");
            }
        }
        int previous = Interlocked.CompareExchange(
            ref _capturedDeviceStates[deviceSlot],
            DevicePublishing,
            DeviceNone);
        if (previous != DeviceNone)
        {
            string status = previous switch
            {
                DevicePublishing => "is already being published",
                DevicePublished => "was published twice",
                DeviceFailed => "previously failed",
                _ => "has an invalid publication state",
            };
            throw new InvalidOperationException(
                $"CUDA device {deviceIndex} {status} for captured gradient " +
                $"step {stepId}.");
        }
    }

    internal void CompleteCapturedPublication(
        long stepId,
        int deviceSlot,
        int deviceIndex)
    {
        RequireActive(stepId);
        ValidateDeviceSlot(deviceSlot);
        if (Interlocked.CompareExchange(
                ref _capturedDeviceStates[deviceSlot],
                DevicePublished,
                DevicePublishing) != DevicePublishing)
        {
            throw new InvalidOperationException(
                $"CUDA device {deviceIndex} has no captured gradient " +
                $"publication to complete for step {stepId}.");
        }
    }

    internal void FailCapturedPublication(long stepId, int deviceSlot)
    {
        if (Volatile.Read(ref _activeStepId) != stepId
            || (uint)deviceSlot >= (uint)_capturedDeviceStates.Length)
        {
            return;
        }
        _ = Interlocked.CompareExchange(
            ref _capturedDeviceStates[deviceSlot],
            DeviceFailed,
            DevicePublishing);
    }

    internal void ValidateComplete(long stepId)
    {
        RequireActive(stepId);
        int path = Volatile.Read(ref _path);
        if (path == PathCaptureRecording)
        {
            throw new InvalidOperationException(
                $"Captured backward recording step {stepId} must be " +
                "discarded, not completed as a reduction.");
        }
        if (path != PathCaptured)
            return;
        for (int slot = 0; slot < _capturedDeviceStates.Length; slot++)
        {
            int state = Volatile.Read(ref _capturedDeviceStates[slot]);
            if (state == DeviceFailed)
            {
                throw new InvalidOperationException(
                    $"Captured gradient publication failed on device slot " +
                    $"{slot} for step {stepId}; abort is required.");
            }
            if (state != DevicePublished)
            {
                throw new InvalidOperationException(
                    $"Captured gradients were not published on device slot " +
                    $"{slot} for step {stepId}.");
            }
        }
    }

    internal void EndStep(long stepId)
    {
        if (Interlocked.CompareExchange(
                ref _activeStepId,
                0,
                stepId) == stepId)
        {
            Volatile.Write(ref _path, PathNone);
        }
    }

    private void ClaimPath(
        int requested,
        string requestedName,
        string conflictingName)
    {
        while (true)
        {
            int current = Volatile.Read(ref _path);
            if (current == requested)
                return;
            if (current != PathNone)
            {
                throw new InvalidOperationException(
                    $"Cannot mix {requestedName} with {conflictingName} in " +
                    "one CUDA gradient reduction step.");
            }
            if (Interlocked.CompareExchange(
                    ref _path,
                    requested,
                    PathNone) == PathNone)
            {
                return;
            }
        }
    }

    private void RequireBegun(
        long stepId,
        int deviceSlot,
        int deviceIndex)
    {
        RequireActive(stepId);
        ValidateDeviceSlot(deviceSlot);
        if (Volatile.Read(ref _deviceBeginSteps[deviceSlot]) != stepId)
        {
            throw new InvalidOperationException(
                $"CUDA device {deviceIndex} was not begun for captured " +
                $"gradient step {stepId}.");
        }
    }

    private void RequireActive(long stepId)
    {
        long active = Volatile.Read(ref _activeStepId);
        if (stepId == 0 || active != stepId)
        {
            throw new InvalidOperationException(
                $"Captured gradient step {stepId} is not active; active " +
                $"step is {active}.");
        }
    }

    private void ValidateDeviceSlot(int deviceSlot)
    {
        if ((uint)deviceSlot >= (uint)_deviceBeginSteps.Length)
            throw new ArgumentOutOfRangeException(nameof(deviceSlot));
    }
}

internal sealed class CudaCapturedBackwardRecordingScope(
    Action complete) : IDisposable
{
    private Action? _complete = complete
        ?? throw new ArgumentNullException(nameof(complete));

    public void Dispose()
        => Interlocked.Exchange(ref _complete, null)?.Invoke();
}
