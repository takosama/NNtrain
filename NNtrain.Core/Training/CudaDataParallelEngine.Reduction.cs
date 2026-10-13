using NNtrain.Cuda.Execution;
using NNtrain.Cuda.Memory;
using NNtrain.Runtime.Execution;

namespace NNtrain;

public sealed partial class CudaDataParallelEngine
{
    private TensorCudaKernels.FlatGradientPlan GetFlatGradientPlan(
        IReadOnlyList<Parameter> parameters,
        IReadOnlyList<int> devices)
    {
        if (_flatGradientPlan is null
            || !_flatGradientPlan.Matches(parameters, devices))
        {
            _flatGradientPlan?.Dispose();
            _flatGradientPlan = new TensorCudaKernels.FlatGradientPlan(
                parameters, devices);
            Interlocked.Increment(ref _flatGradientPlanBuildCount);
        }
        return _flatGradientPlan;
    }

    private CudaBFloat16GradientAllReducePlan? GetBFloat16GradientPlan(
        IReadOnlyList<Parameter> parameters,
        IReadOnlyList<int> devices)
    {
        if (!UseBFloat16GradientBuckets(devices, _model.PrecisionMode))
        {
            _bfloat16GradientPlan?.Dispose();
            _bfloat16GradientPlan = null;
            return null;
        }
        _flatGradientPlan?.Dispose();
        _flatGradientPlan = null;
        bool useBlockBfp8Transport =
            _model.PrecisionMode == TensorPrecisionMode.Mix8_32;
        if (_bfloat16GradientPlan is null
            || !_bfloat16GradientPlan.Matches(parameters, devices)
            || _bfloat16GradientPlan.UsesBlockBfp8Transport
                != useBlockBfp8Transport)
        {
            _bfloat16GradientPlan?.Dispose();
            _bfloat16GradientPlan =
                new CudaBFloat16GradientAllReducePlan(
                    parameters,
                    devices,
                    _dispatchPolicy,
                    useBFloat16GradientStorage:
                        ResolvePrecisionPolicy(_model.PrecisionMode).Gradient
                            == NumericFormat.BFloat16,
                    useBlockBfp8Transport: useBlockBfp8Transport);
            Interlocked.Increment(ref _bfloat16GradientPlanBuildCount);
        }
        return _bfloat16GradientPlan;
    }

    private CudaBfp8GradientAllReducePlan? GetBfp8GradientPlan(
        IReadOnlyList<Parameter> parameters,
        IReadOnlyList<int> devices)
    {
        if (devices.Count != 2
            || ResolvePrecisionPolicy(_model.PrecisionMode).Gradient
                != NumericFormat.Bfp8)
        {
            _bfp8GradientPlan?.Dispose();
            _bfp8GradientPlan = null;
            return null;
        }
        _flatGradientPlan?.Dispose();
        _flatGradientPlan = null;
        if (_bfp8GradientPlan is null
            || !_bfp8GradientPlan.Matches(parameters, devices))
        {
            _bfp8GradientPlan?.Dispose();
            _bfp8GradientPlan =
                new CudaBfp8GradientAllReducePlan(parameters, devices);
        }
        return _bfp8GradientPlan;
    }

    private static PrecisionPolicy ResolvePrecisionPolicy(
        TensorPrecisionMode precisionMode)
        => precisionMode switch
        {
            TensorPrecisionMode.Float32 => PrecisionPolicy.Float32,
            TensorPrecisionMode.BFloat16 => PrecisionPolicy.BFloat16,
            TensorPrecisionMode.Mix16_32 => PrecisionPolicy.Mix16_32,
            TensorPrecisionMode.Bfp8 => PrecisionPolicy.Bfp8,
            TensorPrecisionMode.Mix8_32 => PrecisionPolicy.Mix8_32,
            _ => throw new ArgumentOutOfRangeException(
                nameof(precisionMode), precisionMode, null),
        };

    private bool UseBFloat16GradientBuckets(
        IReadOnlyList<int> devices,
        TensorPrecisionMode precisionMode)
        => devices.Count == 2
            && (precisionMode == TensorPrecisionMode.Mix8_32
                || (ResolvePrecisionPolicy(precisionMode).Gradient
                        == NumericFormat.BFloat16
                    && !_dispatchPolicy.DisableBFloat16GradientBuckets));

}
