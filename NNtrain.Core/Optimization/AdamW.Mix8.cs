using NNtrain.Runtime.Execution;

namespace NNtrain;

public partial class AdamW
{
    private static bool UsesMix8Parameter(Tensor tensor)
        => tensor.DType == TensorDType.Bfp8
            && tensor.Bfp8Quantization?.Granularity
                == Bfp8ScaleGranularity.Block;

    private static void ValidateMix8OptimizerContract()
    {
        PrecisionPolicy? policy = TensorExecutionContext.ActivePrecisionPolicy;
        if (policy is null)
            return;
        if (policy.Mode == PrecisionMode.Mix8_32
            && policy.Gradient == NumericFormat.Float32
            && policy.OptimizerState == NumericFormat.Float32
            && policy.MasterWeight == NumericFormat.Float32)
        {
            return;
        }

        throw new InvalidOperationException(
            "Block-scaled BFP8 parameter storage requires the mix8_32 " +
            "optimizer contract (FP32 gradient, optimizer state, and " +
            "master weight). The active precision policy is " +
            $"'{policy}'.");
    }

    private void StepCudaMix8(int[] devices, AdamWOptions options)
    {
        if (devices.Length == 0)
        {
            throw new InvalidOperationException(
                "mix8_32 AdamW requires at least one CUDA device.");
        }

        // A precision transition may leave tensor-wide BFP8 state resident.
        // Preserve it once, then make FP32 moments/master the sole optimizer
        // authority for the mixed contract.
        bool transitioned = _parameterRuntime.Any(runtime =>
            runtime.CudaBfp8State is not null
            || runtime.CudaState is null
            || runtime.CudaBFloat16State is not null
            || runtime.CudaMixedState is not null);
        if (transitioned)
        {
            foreach (CudaOptimizerKernels.AdamWMultiTensorPlan plan
                in _cudaMultiTensorPlans.Values)
            {
                plan.Dispose();
            }
            _cudaMultiTensorPlans.Clear();
            DisposeCudaBfp8Plans();
        }

        int primaryDevice = devices[0];
        CudaMix8QuantizationDiagnostics diagnosticsOwner =
            _cudaMix8Diagnostics ??= new CudaMix8QuantizationDiagnostics();
        _cudaMix8DiagnosticsDevice = -1;
        var diagnostics = diagnosticsOwner.Reset(primaryDevice);
        foreach (AdamWParameterRuntime runtime in _parameterRuntime)
        {
            if (runtime.CudaBfp8State is not null)
            {
                runtime.CudaBfp8State.SynchronizeHost(primaryDevice);
                runtime.CudaBfp8State.Dispose();
                runtime.CudaBfp8State = null;
            }
            if (runtime.CudaBFloat16State is not null)
            {
                runtime.CudaBFloat16State.SynchronizeHost(primaryDevice);
                runtime.CudaBFloat16State.Dispose();
                runtime.CudaBFloat16State = null;
            }
            if (runtime.CudaMixedState is not null)
            {
                runtime.CudaMixedState.SynchronizeHost(primaryDevice);
                runtime.CudaMixedState.Dispose();
                runtime.CudaMixedState = null;
            }
            if (runtime.FirstMomentBFloat16 is not null)
            {
                runtime.FirstMoment = DecodeBFloat16(
                    runtime.FirstMomentBFloat16);
                runtime.FirstMomentBFloat16 = null;
            }
            if (runtime.SecondMomentBFloat16 is not null)
            {
                runtime.SecondMoment = DecodeBFloat16(
                    runtime.SecondMomentBFloat16);
                runtime.SecondMomentBFloat16 = null;
            }
            runtime.CudaState ??=
                new CudaOptimizerKernels.AdamWResidentState(
                    runtime.FirstMoment,
                    runtime.SecondMoment);
            foreach (int deviceIndex in devices)
                runtime.CudaState.GetOrCreate(deviceIndex);
        }

        CudaOptimizerKernels.AdamWMultiTensorItem[] items =
            _parameterRuntime.Select(runtime => new CudaOptimizerKernels
                .AdamWMultiTensorItem(
                    runtime.Parameter.T,
                    runtime.CudaState,
                    BFloat16State: null,
                    ApplyWeightDecay: runtime.ApplyWeightDecay,
                    PureBFloat16: false))
                .ToArray();
        var plans = new CudaOptimizerKernels.AdamWMultiTensorPlan[
            devices.Length];
        var statuses = new NativeCudaBuffer<int>[devices.Length];
        for (int deviceSlot = 0; deviceSlot < devices.Length; deviceSlot++)
        {
            int deviceIndex = devices[deviceSlot];
            bool enableMix8Diagnostics = deviceIndex == primaryDevice;
            if (!_cudaMultiTensorPlans.TryGetValue(
                    deviceIndex,
                    out CudaOptimizerKernels.AdamWMultiTensorPlan? plan)
                    || !plan.Matches(items, enableMix8Diagnostics))
            {
                if (plan is not null)
                {
                    plan.Dispose();
                    _cudaMultiTensorPlans.Remove(deviceIndex);
                }
                plan = new CudaOptimizerKernels.AdamWMultiTensorPlan(
                    deviceIndex,
                    items,
                    enableMix8Diagnostics);
                _cudaMultiTensorPlans.Add(deviceIndex, plan);
                _cudaMultiTensorPlanBuildCount++;
            }
            plans[deviceSlot] = plan;
            statuses[deviceSlot] = GetOrCreateBfp8FiniteStatus(deviceIndex);
            statuses[deviceSlot].MemSetToZero();
        }

        Parallel.For(0, devices.Length, deviceSlot =>
        {
            int deviceIndex = devices[deviceSlot];
            var deviceDiagnostics = deviceIndex == primaryDevice
                ? diagnostics
                : null;
            if (deviceDiagnostics is null)
            {
                plans[deviceSlot].Execute(
                    options.Beta1,
                    options.Beta2,
                    options.LearningRate,
                    options.WeightDecay,
                    _stepUpdateScale,
                    _stepScaledEpsilon);
            }
            else
            {
                plans[deviceSlot].ExecuteMix8Diagnostic(
                    options.Beta1,
                    options.Beta2,
                    options.LearningRate,
                    options.WeightDecay,
                    _stepUpdateScale,
                    _stepScaledEpsilon,
                    deviceDiagnostics);
            }
            foreach (AdamWParameterRuntime runtime in _parameterRuntime)
            {
                CudaOptimizerKernels.AccumulateAdamWMix8FiniteStatus(
                    runtime.Parameter.T,
                    deviceIndex,
                    runtime.CudaState!,
                    statuses[deviceSlot]);
                CudaOptimizerKernels.PublishMix8Master(
                    runtime.Parameter.T,
                    deviceIndex,
                    statuses[deviceSlot],
                    deviceDiagnostics);
            }
        });

        CudaOptimizerFiniteStatusReadback[] readbacks = devices
            .Select(GetOrCreateBfp8FiniteReadback)
            .ToArray();
        CudaOptimizerStepBatch.CompleteAfterSynchronization(
            devices,
            "mix8_32 AdamW update",
            queueReadback: () =>
            {
                for (int deviceSlot = 0;
                    deviceSlot < devices.Length;
                    deviceSlot++)
                {
                    readbacks[deviceSlot].Begin(statuses[deviceSlot]);
                }
            },
            finalize: () =>
            {
                ThrowIfMix8PublicationNonFinite(
                    readbacks, devices, "AdamW", _step);
                foreach (AdamWParameterRuntime runtime in _parameterRuntime)
                {
                    runtime.Parameter.T
                        .MarkCudaBfp8DataReplicasSynchronized(devices);
                }
            });
        _cudaMix8DiagnosticsDevice = primaryDevice;
    }

    private static void ThrowIfMix8PublicationNonFinite(
        IReadOnlyList<CudaOptimizerFiniteStatusReadback> readbacks,
        IReadOnlyList<int> devices,
        string optimizer,
        int step)
    {
        int nonFiniteDevice = -1;
        for (int deviceSlot = 0; deviceSlot < devices.Count; deviceSlot++)
        {
            int finite = readbacks[deviceSlot].ReadAfterSynchronization();
            if (finite != 0 && nonFiniteDevice < 0)
                nonFiniteDevice = devices[deviceSlot];
        }
        if (nonFiniteDevice >= 0)
        {
            throw new InvalidOperationException(
                $"Non-finite CUDA value detected while publishing " +
                $"mix8_32 {optimizer} parameters on device " +
                $"{nonFiniteDevice} at optimizer step {step}.");
        }
    }

    internal (NativeCudaBuffer<float> First, NativeCudaBuffer<float> Second)
        GetCudaMix8Moments(int parameterIndex, int deviceIndex)
    {
        AdamWParameterRuntime runtime = _parameterRuntime[parameterIndex];
        CudaOptimizerKernels.AdamWResidentState state = runtime.CudaState
            ?? throw new InvalidOperationException(
                "The AdamW parameter has no resident FP32 mixed state.");
        CudaOptimizerKernels.AdamWResidentState.Buffers buffers =
            state.GetOrCreate(deviceIndex);
        return (buffers.First, buffers.Second);
    }
}
