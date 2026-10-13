namespace NNtrain;

public partial class AdamW
{
    private void StepCudaBfp8(
        int[] devices,
        AdamWOptions options,
        bool mixedBlockState = false)
    {
        if (devices.Length == 0)
        {
            throw new InvalidOperationException(
                "Pure BFP8 AdamW requires at least one CUDA device.");
        }

        // A precision transition invalidates plans which captured Float32 or
        // BF16 moment addresses. Preserve their latest values before making
        // the BFP8 payloads the only resident optimizer authority.
        bool transitioned = _parameterRuntime.Any(runtime =>
            runtime.CudaBfp8State is null);
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
        foreach (AdamWParameterRuntime runtime in _parameterRuntime)
        {
            if (runtime.CudaBfp8State is null)
            {
                runtime.CudaState?.SynchronizeHost(primaryDevice);
                runtime.CudaBFloat16State?.SynchronizeHost(primaryDevice);
                runtime.CudaMixedState?.SynchronizeHost(primaryDevice);
                runtime.CudaState?.Dispose();
                runtime.CudaState = null;
                runtime.CudaBFloat16State?.Dispose();
                runtime.CudaBFloat16State = null;
                runtime.CudaMixedState?.Dispose();
                runtime.CudaMixedState = null;

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
                if (runtime.FirstMoment.Length != runtime.Parameter.T.Numel)
                    runtime.FirstMoment = new float[runtime.Parameter.T.Numel];
                if (runtime.SecondMoment.Length != runtime.Parameter.T.Numel)
                    runtime.SecondMoment = new float[runtime.Parameter.T.Numel];

                runtime.CudaBfp8State =
                    new CudaOptimizerKernels.AdamWBfp8ResidentState(
                        runtime.FirstMoment,
                        runtime.SecondMoment,
                        mixedBlockState
                            ? Bfp8QuantizationDescriptor.Mix8_32
                            : Bfp8QuantizationDescriptor.TensorWide);
            }
            foreach (int deviceIndex in devices)
                runtime.CudaBfp8State.GetOrCreate(deviceIndex);
        }

        var statuses = new NativeCudaBuffer<int>[devices.Length];
        for (int deviceSlot = 0; deviceSlot < devices.Length; deviceSlot++)
        {
            int deviceIndex = devices[deviceSlot];
            statuses[deviceSlot] = GetOrCreateBfp8FiniteStatus(deviceIndex);
            statuses[deviceSlot].MemSetToZero();
        }

        CudaOptimizerKernels.AdamWBfp8MultiTensorPlan[]? purePlans = null;
        if (!mixedBlockState)
        {
            CudaOptimizerKernels.AdamWBfp8MultiTensorItem[] items =
                CreateCudaBfp8PlanItems();
            purePlans = PrepareCudaBfp8Plans(devices, items);
        }

        Parallel.For(0, devices.Length, deviceSlot =>
        {
            int deviceIndex = devices[deviceSlot];
            NativeCudaBuffer<int> finiteStatus = statuses[deviceSlot];
            if (!mixedBlockState)
            {
                purePlans![deviceSlot].Execute(
                    options.Beta1,
                    options.Beta2,
                    options.LearningRate,
                    options.WeightDecay,
                    _stepUpdateScale,
                    _stepScaledEpsilon,
                    finiteStatus);
                return;
            }
            foreach (AdamWParameterRuntime runtime in _parameterRuntime)
            {
                runtime.CudaBfp8State!.Execute(
                    runtime.Parameter.T,
                    deviceIndex,
                    finiteStatus,
                    scratch: null,
                    options.Beta1,
                    options.Beta2,
                    options.LearningRate,
                    options.WeightDecay,
                    _stepUpdateScale,
                    _stepScaledEpsilon,
                    runtime.ApplyWeightDecay,
                    mixedBlockState: true);
            }
        });

        CudaOptimizerFiniteStatusReadback[] readbacks = devices
            .Select(GetOrCreateBfp8FiniteReadback)
            .ToArray();
        CudaOptimizerStepBatch.CompleteAfterSynchronization(
            devices,
            mixedBlockState
                ? "block-BFP8-state mix8_32 AdamW update"
                : "pure BFP8 AdamW update",
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
                int nonFiniteDevice = -1;
                for (int deviceSlot = 0;
                    deviceSlot < devices.Length;
                    deviceSlot++)
                {
                    int finite =
                        readbacks[deviceSlot].ReadAfterSynchronization();
                    if (finite != 0 && nonFiniteDevice < 0)
                        nonFiniteDevice = devices[deviceSlot];
                }
                if (nonFiniteDevice >= 0)
                {
                    throw new InvalidOperationException(
                        $"Non-finite CUDA value detected while publishing " +
                        $"pure BFP8 AdamW state on device " +
                        $"{nonFiniteDevice} at optimizer step {_step}.");
                }

                foreach (AdamWParameterRuntime runtime in _parameterRuntime)
                {
                    runtime.Parameter.T
                        .MarkCudaBfp8DataReplicasSynchronized(devices);
                }
            });
    }

    private CudaOptimizerKernels.AdamWBfp8MultiTensorItem[]
        CreateCudaBfp8PlanItems()
        => _parameterRuntime
            .Select(runtime => new CudaOptimizerKernels
                .AdamWBfp8MultiTensorItem(
                    runtime.Parameter.T,
                    runtime.CudaBfp8State
                        ?? throw new InvalidOperationException(
                            "Pure BFP8 AdamW state was not prepared."),
                    runtime.ApplyWeightDecay))
            .ToArray();

    private CudaOptimizerKernels.AdamWBfp8MultiTensorPlan[]
        PrepareCudaBfp8Plans(
            IReadOnlyList<int> devices,
            IReadOnlyList<CudaOptimizerKernels.AdamWBfp8MultiTensorItem>
                items)
    {
        var plans = new CudaOptimizerKernels.AdamWBfp8MultiTensorPlan[
            devices.Count];
        for (int slot = 0; slot < devices.Count; slot++)
        {
            int deviceIndex = devices[slot];
            if (!_cudaBfp8MultiTensorPlans.TryGetValue(
                    deviceIndex,
                    out CudaOptimizerKernels.AdamWBfp8MultiTensorPlan? plan)
                || !plan.Matches(items))
            {
                if (plan is not null)
                {
                    plan.Dispose();
                    _cudaBfp8MultiTensorPlans.Remove(deviceIndex);
                }
                plan = new CudaOptimizerKernels.AdamWBfp8MultiTensorPlan(
                    deviceIndex, items);
                _cudaBfp8MultiTensorPlans.Add(deviceIndex, plan);
                _cudaMultiTensorPlanBuildCount++;
            }
            plans[slot] = plan;
        }
        return plans;
    }

    private void DisposeCudaBfp8Plans()
    {
        foreach (CudaOptimizerKernels.AdamWBfp8MultiTensorPlan plan
            in _cudaBfp8MultiTensorPlans.Values)
        {
            plan.Dispose();
        }
        _cudaBfp8MultiTensorPlans.Clear();
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

    private CudaOptimizerKernels.AdamWBfp8DeviceScratch
        GetOrCreateBfp8Scratch(int deviceIndex, int capacity)
    {
        if (_cudaBfp8Scratch.TryGetValue(
                deviceIndex,
                out CudaOptimizerKernels.AdamWBfp8DeviceScratch? scratch))
        {
            if (scratch.Capacity >= capacity)
                return scratch;
            scratch.Dispose();
            _cudaBfp8Scratch.Remove(deviceIndex);
        }
        scratch = new CudaOptimizerKernels.AdamWBfp8DeviceScratch(
            deviceIndex, capacity);
        _cudaBfp8Scratch.Add(deviceIndex, scratch);
        return scratch;
    }

    private void DisposeBfp8FiniteStatus()
    {
        List<Exception>? failures = null;
        foreach (CudaOptimizerFiniteStatusReadback readback
            in _cudaBfp8FiniteReadbacks.Values)
        {
            try
            {
                readback.Dispose();
            }
            catch (Exception exception)
            {
                (failures ??= []).Add(exception);
            }
        }
        _cudaBfp8FiniteReadbacks.Clear();
        foreach (NativeCudaBuffer<int> status
            in _cudaBfp8FiniteStatus.Values)
        {
            try
            {
                status.Dispose();
            }
            catch (Exception exception)
            {
                (failures ??= []).Add(exception);
            }
        }
        _cudaBfp8FiniteStatus.Clear();
        if (failures is not null)
        {
            throw new AggregateException(
                "AdamW BFP8 finite-status cleanup failed.", failures);
        }
    }

    private void DisposeBfp8Scratch()
    {
        List<Exception>? failures = null;
        foreach (CudaOptimizerKernels.AdamWBfp8DeviceScratch scratch
            in _cudaBfp8Scratch.Values)
        {
            try
            {
                scratch.Dispose();
            }
            catch (Exception exception)
            {
                (failures ??= []).Add(exception);
            }
        }
        _cudaBfp8Scratch.Clear();
        if (failures is not null)
        {
            throw new AggregateException(
                "AdamW BFP8 shared-scratch cleanup failed.", failures);
        }
    }

    internal void DisposeCudaResources()
    {
        _arcState.Dispose();
        List<Exception>? failures = null;
        foreach (CudaOptimizerKernels.AdamWMultiTensorPlan plan
            in _cudaMultiTensorPlans.Values)
        {
            TryDisposeCudaResource(plan, ref failures);
        }
        _cudaMultiTensorPlans.Clear();
        foreach (CudaOptimizerKernels.AdamWBfp8MultiTensorPlan plan
            in _cudaBfp8MultiTensorPlans.Values)
        {
            TryDisposeCudaResource(plan, ref failures);
        }
        _cudaBfp8MultiTensorPlans.Clear();
        foreach (AdamWParameterRuntime runtime in _parameterRuntime)
        {
            if (runtime.CudaState is not null)
                TryDisposeCudaResource(runtime.CudaState, ref failures);
            if (runtime.CudaBFloat16State is not null)
                TryDisposeCudaResource(runtime.CudaBFloat16State, ref failures);
            if (runtime.CudaMixedState is not null)
                TryDisposeCudaResource(runtime.CudaMixedState, ref failures);
            if (runtime.CudaBfp8State is not null)
                TryDisposeCudaResource(runtime.CudaBfp8State, ref failures);
            runtime.CudaState = null;
            runtime.CudaBFloat16State = null;
            runtime.CudaMixedState = null;
            runtime.CudaBfp8State = null;
        }
        try
        {
            DisposeBfp8FiniteStatus();
        }
        catch (Exception exception)
        {
            (failures ??= []).Add(exception);
        }
        try
        {
            DisposeBfp8Scratch();
        }
        catch (Exception exception)
        {
            (failures ??= []).Add(exception);
        }
        if (_cudaMix8Diagnostics is IDisposable mix8Diagnostics)
            TryDisposeCudaResource(mix8Diagnostics, ref failures);
        _cudaMix8Diagnostics = null;
        _cudaMix8DiagnosticsDevice = -1;
        _cudaStateAuthorityDevice = null;
        if (failures is not null)
        {
            throw new AggregateException(
                "AdamW CUDA resource cleanup failed.", failures);
        }
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

    /// <summary>
    /// Transfers resident moments to their existing managed backing arrays,
    /// then drops plans and device buffers before a CPU update. This is a
    /// precision-preserving authority handoff; BF16 and BFP8 state is decoded
    /// only by the same resident-state codecs used for checkpoint streaming.
    /// </summary>
    private void MakeCpuStateAuthoritative()
    {
        if (_cudaStateAuthorityDevice is null)
            return;

        SynchronizeStateForStreaming();
        DisposeCudaResources();
    }

    internal (CudaBfp8BufferView First, CudaBfp8BufferView Second)
        GetCudaBfp8Moments(int parameterIndex, int deviceIndex)
    {
        CudaOptimizerKernels.AdamWBfp8ResidentState state =
            _parameterRuntime[parameterIndex].CudaBfp8State
            ?? throw new InvalidOperationException(
                "The AdamW parameter has no resident BFP8 state.");
        return (
            state.GetFirstMoment(deviceIndex),
            state.GetSecondMoment(deviceIndex));
    }

    internal (NativeCudaBuffer<short> First, NativeCudaBuffer<short> Second)
        GetCudaBFloat16Moments(int parameterIndex, int deviceIndex)
    {
        CudaOptimizerKernels.AdamWBFloat16ResidentState state =
            _parameterRuntime[parameterIndex].CudaBFloat16State
            ?? throw new InvalidOperationException(
                "The AdamW parameter has no resident BF16 state.");
        CudaOptimizerKernels.AdamWBFloat16ResidentState.Buffers buffers =
            state.GetOrCreate(deviceIndex);
        return (buffers.First, buffers.Second);
    }

}
