using NNtrain.Cuda.Interop;
using NNtrain.Cuda.Memory;

namespace NNtrain.Cuda.Execution;

/// <summary>
/// A CUDA event recorded after the last use of a resource. The fence can be
/// transferred to <see cref="CudaMemoryLease"/> or waited by a native-resource
/// owner before destroying stream-visible state.
/// </summary>
public sealed class CudaEventCompletionFence : ICudaCompletionFence
{
    private const int Success = 0;
    private const int NotReady = 600;

    private readonly int _deviceIndex;
    private nint _event;

    private CudaEventCompletionFence(int deviceIndex, nint cudaEvent)
    {
        _deviceIndex = deviceIndex;
        _event = cudaEvent;
    }

    public bool IsCompleted
    {
        get
        {
            nint cudaEvent = GetRequiredEvent();
            int status = CudaNativeGateway.EventQuery(
                _deviceIndex,
                cudaEvent);
            if (status == Success)
                return true;
            if (status == NotReady)
                return false;
            Throw(status, "cudaEventQuery(resource fence)");
            return false;
        }
    }

    /// <summary>Records a completion event on <paramref name="stream"/>.</summary>
    public static CudaEventCompletionFence Record(
        int deviceIndex,
        nint stream)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(deviceIndex);
        int createStatus = CudaNativeGateway.EventCreate(
            deviceIndex,
            out nint cudaEvent);
        if (createStatus != Success)
            Throw(createStatus, "cudaEventCreate(resource fence)");

        try
        {
            int recordStatus = CudaNativeGateway.EventRecord(
                deviceIndex,
                cudaEvent,
                stream);
            if (recordStatus != Success)
                Throw(recordStatus, "cudaEventRecord(resource fence)");
            return new CudaEventCompletionFence(deviceIndex, cudaEvent);
        }
        catch (Exception recordFailure)
        {
            int destroyStatus = CudaNativeGateway.EventDestroy(
                deviceIndex,
                cudaEvent);
            if (destroyStatus == Success)
                throw;
            throw new AggregateException(
                "CUDA resource-fence creation and rollback both failed.",
                recordFailure,
                CreateFailure(
                    destroyStatus,
                    "cudaEventDestroy(resource fence rollback)"));
        }
    }

    public void Wait()
    {
        int status = CudaNativeGateway.EventSynchronize(
            _deviceIndex,
            GetRequiredEvent());
        if (status != Success)
            Throw(status, "cudaEventSynchronize(resource fence)");
    }

    public void Dispose()
    {
        nint cudaEvent = Interlocked.Exchange(ref _event, nint.Zero);
        if (cudaEvent == nint.Zero)
            return;
        int status = CudaNativeGateway.EventDestroy(
            _deviceIndex,
            cudaEvent);
        if (status != Success)
            Throw(status, "cudaEventDestroy(resource fence)");
    }

    private nint GetRequiredEvent()
    {
        nint cudaEvent = Volatile.Read(ref _event);
        ObjectDisposedException.ThrowIf(
            cudaEvent == nint.Zero,
            this);
        return cudaEvent;
    }

    private static void Throw(int status, string operation)
        => throw CreateFailure(status, operation);

    private static InvalidOperationException CreateFailure(
        int status,
        string operation)
        => new(
            $"{operation} failed with CUDA error {status}: " +
            CudaNativeGateway.ErrorString(status));
}
