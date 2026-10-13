using System.Diagnostics;
using NNtrain.Cuda.Interop;

namespace NNtrain;

public sealed partial class NekoMuon
{
    private static bool UsesPureBfp8OptimizerState(Tensor tensor)
    {
        NNtrain.Runtime.Execution.PrecisionPolicy? policy =
            TensorExecutionContext.ActivePrecisionPolicy;
        if (policy is not null
            && policy.OptimizerState
                != NNtrain.Runtime.Execution.NumericFormat.Bfp8)
        {
            return false;
        }
        return tensor.DType == TensorDType.Bfp8
            && tensor.Bfp8Quantization
                == Bfp8QuantizationDescriptor.TensorWide;
    }

    private NativeCudaBuffer<int> GetOrCreateBfp8FiniteStatus(int deviceIndex)
    {
        if (_cudaBfp8FiniteStatus.TryGetValue(
                deviceIndex,
                out NativeCudaBuffer<int>? status))
        {
            return status;
        }
        status = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex)
            .Allocate1D<int>(1);
        _cudaBfp8FiniteStatus.Add(deviceIndex, status);
        return status;
    }

    private CudaOptimizerFiniteStatusReadback
        GetOrCreateBfp8FiniteReadback(int deviceIndex)
    {
        if (_cudaBfp8FiniteReadbacks.TryGetValue(
                deviceIndex,
                out CudaOptimizerFiniteStatusReadback? readback))
        {
            return readback;
        }
        readback = new CudaOptimizerFiniteStatusReadback(deviceIndex);
        _cudaBfp8FiniteReadbacks.Add(deviceIndex, readback);
        return readback;
    }

    internal (CudaBfp8BufferView Fast, CudaBfp8BufferView Slow)
        GetCudaBfp8Moments(int parameterIndex, int deviceIndex)
    {
        CudaOptimizerKernels.NekoMuonBfp8ResidentState state =
            _cudaBfp8States[parameterIndex]
            ?? throw new InvalidOperationException(
                "The NekoMuon parameter has no resident BFP8 state.");
        return (state.GetFast(deviceIndex), state.GetSlow(deviceIndex));
    }

    internal (NativeCudaBuffer<ushort> Fast, NativeCudaBuffer<ushort> Slow)
        GetCudaBFloat16Moments(int parameterIndex, int deviceIndex)
    {
        CudaOptimizerKernels.NekoMuonBFloat16ResidentState state =
            _cudaBFloat16States[parameterIndex]
            ?? throw new InvalidOperationException(
                "The NekoMuon parameter has no resident BF16 state.");
        return (state.GetFast(deviceIndex), state.GetSlow(deviceIndex));
    }

    internal void DisposeCudaResources()
    {
        _arcState.Dispose();
        List<Exception>? failures = null;
        for (int index = 0; index < _cudaStates.Length; index++)
        {
            if (_cudaStates[index] is IDisposable floatState)
                TryDisposeCudaResource(floatState, ref failures);
            if (_cudaBFloat16States[index] is IDisposable bfloat16State)
                TryDisposeCudaResource(bfloat16State, ref failures);
            if (_cudaBfp8States[index] is IDisposable bfp8State)
                TryDisposeCudaResource(bfp8State, ref failures);
            _cudaStates[index] = null;
            _cudaBFloat16States[index] = null;
            _cudaBfp8States[index] = null;
        }
        foreach (CudaOptimizerKernels.NekoMuonStatsBatch batch
            in _cudaStatsBatches.Values)
        {
            TryDisposeCudaResource(batch, ref failures);
        }
        _cudaStatsBatches.Clear();
        foreach (CudaOptimizerKernels.NekoMuonConfidenceBatch batch
            in _cudaConfidenceBatches.Values)
        {
            TryDisposeCudaResource(batch, ref failures);
        }
        _cudaConfidenceBatches.Clear();
        foreach (CudaOptimizerKernels.NekoMuonBFloat16StatsBatch batch
            in _cudaBFloat16StatsBatches.Values)
        {
            TryDisposeCudaResource(batch, ref failures);
        }
        _cudaBFloat16StatsBatches.Clear();
        foreach (CudaOptimizerKernels.NekoMuonBfp8StatsBatch batch
            in _cudaBfp8StatsBatches.Values)
        {
            TryDisposeCudaResource(batch, ref failures);
        }
        _cudaBfp8StatsBatches.Clear();
        foreach (CudaOptimizerKernels.NekoMuonDeviceScratch scratch
            in _cudaScratch.Values)
        {
            TryDisposeCudaResource(scratch, ref failures);
        }
        _cudaScratch.Clear();
        foreach (CudaOptimizerFiniteStatusReadback readback
            in _cudaBfp8FiniteReadbacks.Values)
        {
            TryDisposeCudaResource(readback, ref failures);
        }
        _cudaBfp8FiniteReadbacks.Clear();
        foreach (NativeCudaBuffer<int> status
            in _cudaBfp8FiniteStatus.Values)
        {
            TryDisposeCudaResource(status, ref failures);
        }
        _cudaBfp8FiniteStatus.Clear();
        if (_cudaMix8Diagnostics is IDisposable mix8Diagnostics)
            TryDisposeCudaResource(mix8Diagnostics, ref failures);
        _cudaMix8Diagnostics = null;
        _cudaMix8DiagnosticsDevice = -1;
        _cudaStateAuthorityDevice = null;
        if (failures is not null)
        {
            throw new AggregateException(
                "NekoMuon CUDA resource cleanup failed.", failures);
        }
    }

    private void MakeCpuStateAuthoritative()
    {
        if (_cudaStateAuthorityDevice is null)
            return;

        _ = CaptureStateForStreaming();
        DisposeCudaResources();
    }

    private static void TryDisposeCudaResource(
        IDisposable resource,
        ref List<Exception>? failures)
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

}
