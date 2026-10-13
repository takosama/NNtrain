using NNtrain.Cuda.Execution;
using NNtrain.Cuda.Memory;
using NNtrain.Runtime.Execution;

namespace NNtrain;

public sealed partial class CudaDataParallelEngine
{
    private void PrepareParameterResidency()
    {
        PreflightCudaCapabilities();
        if (_parameterResidencyPrepared)
            return;
        foreach (int deviceIndex in _cudaDeviceIndices)
            _ = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);

        var primary = new TorchDevice(
            TensorDevice.Cuda, _cudaDeviceIndices[0]);
        foreach (Parameter parameter in _model.Parameters())
        {
            // Moving the logical tensor once prevents CUDA result construction
            // from treating a lazily-resident parameter as a CPU autograd
            // parent.  Materialize every replica up front so the hot path never
            // routes parameter data or gradients through the shared host mirror.
            parameter.T.to(primary);
            for (int device = 1; device < _cudaDeviceIndices.Length; device++)
            {
                if (parameter.T.DType == TensorDType.BFloat16)
                {
                    parameter.T.EnsureCudaBFloat16Buffer(
                        _cudaDeviceIndices[device]);
                }
                else if (parameter.T.DType == TensorDType.Bfp8)
                {
                    parameter.T.EnsureCudaBfp8Buffer(
                        _cudaDeviceIndices[device]);
                }
                else
                {
                    parameter.T.EnsureCudaFloat32Buffer(
                        _cudaDeviceIndices[device]);
                }
            }
        }
        ForgetMemoryV2Cuda.GetAccelerator(_cudaDeviceIndices[0]).Bind();
        _parameterResidencyPrepared = true;
    }

    private void PreflightCudaCapabilities()
    {
        bool compiledGraphRequested = IsCompiledGraphPreflightRequested();
        TensorPrecisionMode precisionMode = _model.PrecisionMode;
        if (_preflightPrecisionMode == precisionMode
            && _preflightCompiledGraphRequested == compiledGraphRequested)
        {
            return;
        }

        CudaKernelFeature required = ResolveRequiredCudaFeatures(
            _model.GetType(),
            precisionMode,
            _cudaDeviceIndices.Length,
            compiledGraphRequested);
        ExecutionSession? session = ExecutionSession.Current;
        foreach (int deviceIndex in _cudaDeviceIndices)
        {
            bool hasLaneCapabilities = TryGetSessionCudaCapabilities(
                session,
                deviceIndex,
                out CudaKernelCapabilities? laneCapabilities);
            CudaKernelCapabilities capabilities = hasLaneCapabilities
                ? laneCapabilities!
                : NativeCudaRuntime.GetKernelCapabilities(deviceIndex);
            EnsureCudaCapabilities(
                _model.GetType(),
                precisionMode,
                deviceIndex,
                required,
                capabilities);
        }

        _preflightPrecisionMode = precisionMode;
        _preflightCompiledGraphRequested = compiledGraphRequested;
    }

    internal static void EnsureCudaCapabilities(
        Type modelType,
        TensorPrecisionMode precisionMode,
        int deviceIndex,
        CudaKernelFeature required,
        CudaKernelCapabilities capabilities)
    {
        ArgumentNullException.ThrowIfNull(modelType);
        ArgumentOutOfRangeException.ThrowIfNegative(deviceIndex);
        ArgumentNullException.ThrowIfNull(capabilities);
        CudaKernelFeature missing = required & ~capabilities.Features;
        if (missing == CudaKernelFeature.None)
            return;

        throw new NotSupportedException(
            $"CUDA training capability preflight failed for device " +
            $"{deviceIndex} (SM {capabilities.ComputeCapabilityMajor}." +
            $"{capabilities.ComputeCapabilityMinor}, model " +
            $"{modelType.Name}, precision " +
            $"{TensorPrecisionModeNames.Format(precisionMode)}). " +
            $"Missing required CUDA kernel capabilities: " +
            $"{FormatCudaFeatures(missing)}. CPU fallback is forbidden.");
    }

    private bool IsCompiledGraphPreflightRequested()
    {
        if (!_model.HasCheckpointableTrainingRandom
            || _dispatchPolicy.DisableCudaGraphs
            || _dispatchPolicy.SynchronizeDataParallelPhases)
        {
            return false;
        }

        ExecutionSession? session = ExecutionSession.Current;
        if (session is null)
            return false;
        foreach (int deviceIndex in _cudaDeviceIndices)
        {
            if (!session.TryGetLane(
                    ExecutionDeviceKind.Cuda,
                    deviceIndex,
                    out IExecutionLane? executionLane)
                || executionLane is not CudaExecutionLane lane
                || !ReferenceEquals(
                    lane.Profiler,
                    NullExecutionProfiler.Instance))
            {
                return false;
            }
        }
        return true;
    }

    private static bool TryGetSessionCudaCapabilities(
        ExecutionSession? session,
        int deviceIndex,
        out CudaKernelCapabilities? capabilities)
    {
        capabilities = null;
        if (session is null
            || !session.TryGetLane(
                ExecutionDeviceKind.Cuda,
                deviceIndex,
                out IExecutionLane? executionLane)
            || executionLane is not CudaExecutionLane lane)
        {
            return false;
        }
        capabilities = lane.CudaCapabilities;
        return true;
    }

    internal static CudaKernelFeature ResolveRequiredCudaFeatures(
        Type modelType,
        TensorPrecisionMode precisionMode,
        int deviceCount,
        bool compiledGraphRequested)
    {
        ArgumentNullException.ThrowIfNull(modelType);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(deviceCount);
        if (!typeof(LanguageModel).IsAssignableFrom(modelType))
        {
            throw new ArgumentException(
                $"'{modelType.FullName}' is not a language model type.",
                nameof(modelType));
        }

        CudaKernelFeature required = CudaKernelFeature.None;
        switch (precisionMode)
        {
            case TensorPrecisionMode.Float32:
                break;
            case TensorPrecisionMode.BFloat16:
            case TensorPrecisionMode.Mix16_32:
                required |= CudaKernelFeature.TensorCores
                    | CudaKernelFeature.BFloat16;
                break;
            case TensorPrecisionMode.Bfp8:
            case TensorPrecisionMode.Mix8_32:
                required |= CudaKernelFeature.TensorCores
                    | CudaKernelFeature.BFloat16
                    | CudaKernelFeature.Bfp8Quantization;
                break;
            default:
                throw new ArgumentOutOfRangeException(
                    nameof(precisionMode), precisionMode, null);
        }

        if (typeof(GptRinWikiJp).IsAssignableFrom(modelType))
        {
            required |= CudaKernelFeature.FlashAttention
                | CudaKernelFeature.FusedLayerNorm;
        }
        if (typeof(ForgetMemoryV2Gpt).IsAssignableFrom(modelType)
            || typeof(ForgetScanGpt).IsAssignableFrom(modelType))
        {
            required |= CudaKernelFeature.ForgetMemory;
        }
        if (deviceCount > 1)
            required |= CudaKernelFeature.AsynchronousGradientReduction;
        if (compiledGraphRequested)
            required |= CudaKernelFeature.CudaGraphs;
        return required;
    }

    internal static string FormatCudaFeatures(CudaKernelFeature features)
    {
        if (features == CudaKernelFeature.None)
            return nameof(CudaKernelFeature.None);
        return string.Join(
            ", ",
            Enum.GetValues<CudaKernelFeature>()
                .Where(feature => feature != CudaKernelFeature.None
                    && IsSingleCudaFeature(feature)
                    && features.HasFlag(feature))
                .Select(static feature => feature.ToString()));
    }

    private static bool IsSingleCudaFeature(CudaKernelFeature feature)
    {
        int value = (int)feature;
        return (value & (value - 1)) == 0;
    }

}
