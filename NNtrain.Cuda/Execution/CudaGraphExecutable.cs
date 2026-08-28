using System.Runtime.InteropServices;
using NNtrain.Cuda.Interop;
using NNtrain.Runtime.Execution;

namespace NNtrain.Cuda.Execution;

/// <summary>
/// Lane-owned instantiated CUDA Graph. SafeHandle cleanup never throws;
/// explicit callers may inspect or surface a destroy failure with
/// <see cref="DisposeChecked"/>.
/// </summary>
public sealed class CudaGraphExecutable : SafeHandle
{
    private readonly CudaExecutionLane _lane;
    private Exception? _releaseFailure;

    private CudaGraphExecutable(CudaExecutionLane lane, nint executable)
        : base(nint.Zero, ownsHandle: true)
    {
        _lane = lane;
        SetHandle(executable);
    }

    public override bool IsInvalid => handle == nint.Zero;

    public int DeviceIndex => _lane.DeviceIndex;

    public Exception? ReleaseFailure => Volatile.Read(ref _releaseFailure);

    /// <summary>
    /// Captures synchronous submissions made by <paramref name="record"/> on
    /// the lane's compute stream, instantiates them, and transfers the
    /// executable to the lane/session ownership tree.
    /// </summary>
    public static CudaGraphExecutable Capture(
        CudaExecutionLane lane,
        Action record)
    {
        ArgumentNullException.ThrowIfNull(lane);
        ArgumentNullException.ThrowIfNull(record);
        if (!lane.CudaCapabilities.Supports(CudaKernelFeature.CudaGraphs))
        {
            throw new NotSupportedException(
                $"CUDA Graphs are not available on device {lane.DeviceIndex}.");
        }

        lane.ActivateComputeStream();
        CudaGraphStatus.Check(
            CudaNativeGateway.GraphBeginCapture(
                lane.DeviceIndex,
                lane.ComputeStreamHandle),
            "cudaStreamBeginCapture",
            lane.DeviceIndex);

        var failures = new List<Exception>();
        try
        {
            record();
        }
        catch (Exception exception)
        {
            failures.Add(exception);
        }

        nint graph = nint.Zero;
        try
        {
            int endStatus = CudaNativeGateway.GraphEndCapture(
                lane.DeviceIndex,
                lane.ComputeStreamHandle,
                out graph);
            if (endStatus != 0)
            {
                failures.Add(CudaGraphStatus.CreateFailure(
                    endStatus,
                    "cudaStreamEndCapture",
                    lane.DeviceIndex));
            }
        }
        catch (Exception exception)
        {
            failures.Add(exception);
        }

        if (failures.Count != 0)
        {
            TryDestroyGraph(lane.DeviceIndex, graph, failures);
            ThrowCaptureFailures(failures);
        }

        int instantiateStatus = CudaNativeGateway.GraphInstantiate(
            lane.DeviceIndex,
            graph,
            out nint executable);
        if (instantiateStatus != 0 || executable == nint.Zero)
        {
            failures.Add(CudaGraphStatus.CreateFailure(
                instantiateStatus,
                "cudaGraphInstantiate",
                lane.DeviceIndex));
            TryDestroyGraph(lane.DeviceIndex, graph, failures);
            ThrowCaptureFailures(failures);
        }

        var result = new CudaGraphExecutable(lane, executable);
        int destroyStatus = CudaNativeGateway.GraphDestroy(
            lane.DeviceIndex,
            graph);
        if (destroyStatus != 0)
        {
            failures.Add(CudaGraphStatus.CreateFailure(
                destroyStatus,
                "cudaGraphDestroy after instantiate",
                lane.DeviceIndex));
            result.Dispose();
            if (result.ReleaseFailure is Exception releaseFailure)
                failures.Add(releaseFailure);
            ThrowCaptureFailures(failures);
        }

        return ExecutionLaneResources.Attach(lane, result);
    }

    public void Launch()
    {
        ObjectDisposedException.ThrowIf(IsClosed, this);
        bool addedReference = false;
        try
        {
            DangerousAddRef(ref addedReference);
            _lane.ActivateComputeStream();
            CudaGraphStatus.Check(
                CudaNativeGateway.GraphLaunch(
                    DeviceIndex,
                    DangerousGetHandle(),
                    _lane.ComputeStreamHandle),
                "cudaGraphLaunch",
                DeviceIndex);
        }
        finally
        {
            if (addedReference)
                DangerousRelease();
        }
    }

    public void DisposeChecked()
    {
        Dispose();
        if (ReleaseFailure is Exception failure)
        {
            throw new InvalidOperationException(
                $"CUDA Graph executable cleanup failed on device " +
                $"{DeviceIndex}.",
                failure);
        }
    }

    protected override bool ReleaseHandle()
    {
        try
        {
            int status = CudaNativeGateway.GraphExecutableDestroy(
                DeviceIndex,
                handle);
            if (status == 0)
                return true;
            Interlocked.CompareExchange(
                ref _releaseFailure,
                CudaGraphStatus.CreateFailure(
                    status,
                    "cudaGraphExecDestroy",
                    DeviceIndex),
                comparand: null);
            return false;
        }
        catch (Exception exception)
        {
            Interlocked.CompareExchange(
                ref _releaseFailure,
                exception,
                comparand: null);
            return false;
        }
    }

    private static void TryDestroyGraph(
        int deviceIndex,
        nint graph,
        List<Exception> failures)
    {
        if (graph == nint.Zero)
            return;
        try
        {
            int status = CudaNativeGateway.GraphDestroy(deviceIndex, graph);
            if (status != 0)
            {
                failures.Add(CudaGraphStatus.CreateFailure(
                    status,
                    "cudaGraphDestroy",
                    deviceIndex));
            }
        }
        catch (Exception exception)
        {
            failures.Add(exception);
        }
    }

    private static void ThrowCaptureFailures(List<Exception> failures)
    {
        if (failures.Count == 1)
        {
            System.Runtime.ExceptionServices.ExceptionDispatchInfo
                .Capture(failures[0])
                .Throw();
        }
        throw new AggregateException(
            "CUDA Graph capture or cleanup failed.",
            failures);
    }
}

internal static class CudaGraphStatus
{
    internal static void Check(int status, string operation, int deviceIndex)
    {
        if (status != 0)
            throw CreateFailure(status, operation, deviceIndex);
    }

    internal static Exception CreateFailure(
        int status,
        string operation,
        int deviceIndex)
        => new InvalidOperationException(
            $"{operation} failed on CUDA device {deviceIndex} with error " +
            $"{status}: {CudaNativeGateway.ErrorString(status)}");
}
