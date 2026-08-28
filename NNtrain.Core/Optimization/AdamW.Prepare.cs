namespace NNtrain;

public partial class AdamW
{
    private readonly object _cudaPreparationSync = new();

    /// <summary>
    /// Materializes every persistent CUDA optimizer resource before a
    /// transfer-guarded training step begins. Repeated calls validate the
    /// current descriptor/device set and reuse the existing residency.
    /// </summary>
    public void prepare()
    {
        if (Tensor.ExecutionDevice != TensorDevice.Cuda)
            return;

        int[] devices = Tensor.CudaDeviceIndices.ToArray();
        if (devices.Length == 0)
        {
            throw new InvalidOperationException(
                "CUDA AdamW preparation requires at least one device.");
        }

        lock (_cudaPreparationSync)
            PrepareCudaResidency(devices);
    }

    private void PrepareCudaResidency(int[] devices)
    {
        bool pureBFloat16 = UsesPureBFloat16OptimizerState();
        if (TensorExecutionContext.ActivePrecisionPolicy?.OptimizerState
                == NNtrain.Runtime.Execution.NumericFormat.BFloat16
            && !pureBFloat16)
        {
            throw new InvalidOperationException(
                "The pure BFloat16 AdamW contract requires every parameter " +
                "to use physical BFloat16 storage.");
        }
        bool pureBfp8 = _parameterRuntime.All(runtime =>
            UsesPureBfp8OptimizerState(runtime.Parameter.T));
        if (TensorExecutionContext.ActivePrecisionPolicy?.OptimizerState
                == NNtrain.Runtime.Execution.NumericFormat.Bfp8
            && !pureBfp8)
        {
            throw new InvalidOperationException(
                "The pure BFP8 AdamW contract requires every parameter " +
                "to use tensor-wide BFP8 storage.");
        }
        if (pureBfp8)
        {
            PrepareCudaBfp8Residency(devices);
            return;
        }

        bool anyMix8 = _parameterRuntime.Any(runtime =>
            UsesMix8Parameter(runtime.Parameter.T));
        bool mix8 = _parameterRuntime.All(runtime =>
            UsesMix8Parameter(runtime.Parameter.T));
        if (anyMix8 && !mix8)
        {
            throw new InvalidOperationException(
                "The mix8_32 AdamW contract cannot mix block-scaled " +
                "BFP8 parameters with another storage format.");
        }
        if (anyMix8)
            ValidateMix8OptimizerContract();
        if (TensorExecutionContext.ActivePrecisionPolicy?.Mode
                == NNtrain.Runtime.Execution.PrecisionMode.Mix8_32
            && !mix8)
        {
            throw new InvalidOperationException(
                "The mix8_32 AdamW contract requires every parameter " +
                "to use block-scaled BFP8 storage.");
        }
        if (mix8)
        {
            PrepareCudaMix8Residency(devices);
            return;
        }

        PrepareCudaStandardResidency(devices);
    }

    private void PrepareCudaStandardResidency(int[] devices)
    {
        foreach (AdamWParameterRuntime runtime in _parameterRuntime)
        {
            if (runtime.FirstMomentBFloat16 is null
                && runtime.SecondMomentBFloat16 is null)
            {
                runtime.CudaState ??=
                    new CudaOptimizerKernels.AdamWResidentState(
                        runtime.FirstMoment,
                        runtime.SecondMoment);
                foreach (int deviceIndex in devices)
                    runtime.CudaState.GetOrCreate(deviceIndex);
                continue;
            }

            if (runtime.FirstMomentBFloat16 is not null
                && runtime.SecondMomentBFloat16 is not null)
            {
                runtime.CudaBFloat16State ??=
                    new CudaOptimizerKernels.AdamWBFloat16ResidentState(
                        runtime.FirstMomentBFloat16,
                        runtime.SecondMomentBFloat16);
                foreach (int deviceIndex in devices)
                    runtime.CudaBFloat16State.GetOrCreate(deviceIndex);
            }
        }

        CudaOptimizerKernels.AdamWMultiTensorItem[] items =
            CreatePreparedPlanItems();
        // A pure-BF16 plan captures the address of the authoritative BF16
        // gradient. prepare() is also a public pre-gradient hook, and the
        // data-parallel reducer may bind its stable arena only afterwards.
        // Materialize weights/moments here, then build the descriptor plan on
        // the first step once every real gradient address is available.
        if (items.Any(item => item.PureBFloat16
            && devices.Any(deviceIndex =>
                !item.Parameter.TryGetCudaBFloat16GradientBuffer(
                    deviceIndex, out _))))
        {
            return;
        }
        PrepareCudaPlans(devices, items);
    }

    private void PrepareCudaMix8Residency(int[] devices)
    {
        bool transitioned = _parameterRuntime.Any(runtime =>
            runtime.CudaBfp8State is not null
            || runtime.CudaState is null
            || runtime.CudaBFloat16State is not null);
        if (transitioned)
            DisposeCudaPlans();

        int primaryDevice = devices[0];
        foreach (AdamWParameterRuntime runtime in _parameterRuntime)
        {
            Tensor parameter = runtime.Parameter.T;
            foreach (int deviceIndex in devices)
                _ = parameter.EnsureCudaBfp8Buffer(deviceIndex);

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
            {
                _ = parameter.EnsureCudaMasterFloat32Buffer(deviceIndex);
                runtime.CudaState.GetOrCreate(deviceIndex);
                _ = GetOrCreateBfp8FiniteStatus(deviceIndex);
                _ = GetOrCreateBfp8FiniteReadback(deviceIndex);
            }
        }

        PrepareCudaPlans(devices, CreatePreparedPlanItems());
    }

    private void PrepareCudaBfp8Residency(int[] devices)
    {
        bool transitioned = _parameterRuntime.Any(runtime =>
            runtime.CudaBfp8State is null);
        if (transitioned)
            DisposeCudaPlans();

        int primaryDevice = devices[0];
        int maximumLeafLength = _parameterRuntime.Max(
            runtime => runtime.Parameter.T.Numel);
        foreach (AdamWParameterRuntime runtime in _parameterRuntime)
        {
            if (runtime.CudaBfp8State is null)
            {
                runtime.CudaState?.SynchronizeHost(primaryDevice);
                runtime.CudaBFloat16State?.SynchronizeHost(primaryDevice);
                runtime.CudaState?.Dispose();
                runtime.CudaState = null;
                runtime.CudaBFloat16State?.Dispose();
                runtime.CudaBFloat16State = null;

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
                {
                    runtime.FirstMoment = new float[
                        runtime.Parameter.T.Numel];
                }
                if (runtime.SecondMoment.Length != runtime.Parameter.T.Numel)
                {
                    runtime.SecondMoment = new float[
                        runtime.Parameter.T.Numel];
                }

                runtime.CudaBfp8State =
                    new CudaOptimizerKernels.AdamWBfp8ResidentState(
                        runtime.FirstMoment,
                        runtime.SecondMoment);
            }

            foreach (int deviceIndex in devices)
            {
                _ = runtime.Parameter.T.EnsureCudaBfp8Buffer(deviceIndex);
                runtime.CudaBfp8State.GetOrCreate(deviceIndex);
                _ = GetOrCreateBfp8FiniteStatus(deviceIndex);
                _ = GetOrCreateBfp8FiniteReadback(deviceIndex);
                CudaOptimizerKernels.AdamWBfp8DeviceScratch scratch =
                    GetOrCreateBfp8Scratch(
                        deviceIndex,
                        maximumLeafLength);
                _ = scratch.Get(runtime.Parameter.T.Numel);
            }
        }
    }

    private CudaOptimizerKernels.AdamWMultiTensorItem[]
        CreatePreparedPlanItems()
    {
        bool pureBFloat16 = UsesPureBFloat16OptimizerState();
        return _parameterRuntime
            .Where(runtime => runtime.CudaState is not null
                || runtime.CudaBFloat16State is not null)
            .Select(runtime => new CudaOptimizerKernels.AdamWMultiTensorItem(
                runtime.Parameter.T,
                runtime.CudaState,
                runtime.CudaBFloat16State,
                runtime.ApplyWeightDecay,
                pureBFloat16))
            .ToArray();
    }

    private void PrepareCudaPlans(
        IReadOnlyList<int> devices,
        IReadOnlyList<CudaOptimizerKernels.AdamWMultiTensorItem> items)
    {
        if (items.Count == 0)
            return;
        foreach (int deviceIndex in devices)
        {
            if (_cudaMultiTensorPlans.TryGetValue(
                    deviceIndex,
                    out CudaOptimizerKernels.AdamWMultiTensorPlan? plan)
                && plan.Matches(items))
            {
                continue;
            }
            if (plan is not null)
            {
                plan.Dispose();
                _cudaMultiTensorPlans.Remove(deviceIndex);
            }
            plan = new CudaOptimizerKernels.AdamWMultiTensorPlan(
                deviceIndex,
                items);
            _cudaMultiTensorPlans.Add(deviceIndex, plan);
            _cudaMultiTensorPlanBuildCount++;
        }
    }

    private void DisposeCudaPlans()
    {
        foreach (CudaOptimizerKernels.AdamWMultiTensorPlan plan
            in _cudaMultiTensorPlans.Values)
        {
            plan.Dispose();
        }
        _cudaMultiTensorPlans.Clear();
    }
}
