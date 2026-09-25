using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

internal static class ArcOptimizerFastPath
{
    // Diagnostic same-binary A/B switches; enabled by default.
    internal static readonly bool SkipUnneededClip = Environment.GetEnvironmentVariable(
        "NNTRAIN_ARC_SKIP_UNNEEDED_CLIP") != "0";
    internal static readonly bool ShareNesterovHat = Environment.GetEnvironmentVariable(
        "NNTRAIN_ARC_SHARE_NESTEROV_HAT") != "0";
    internal static readonly bool ReuseMuonNorm = Environment.GetEnvironmentVariable(
        "NNTRAIN_ARC_REUSE_MUON_NORM") != "0";
    internal static readonly bool NesterovSingleReduction = Environment.GetEnvironmentVariable(
        "NNTRAIN_ARC_NESTEROV_SINGLE_REDUCTION") != "0";
}

/// <summary>Host-staged OpenCL arithmetic. State/checkpoint ownership remains with the optimizer.</summary>
internal static class ArcTrainingMath
{
    internal static float Sum(float[] input, bool squared = false)
    {
        if (input.Length == 0) return 0;
        float[] current = input;
        do
        {
            var partials = new float[(current.Length + 255) / 256];
            Tensor.ArcLane.Run("reduce_sum", (long)partials.Length * 256, 256,
                In(current), InOut(partials), current.Length, squared ? 1 : 0, new LocalMemory(1024));
            current = partials;
            squared = false;
        } while (current.Length > 1);
        return current[0];
    }

    internal static void Combine(float[] a, float[] b, float[] output, float sa, float sb)
        => Tensor.ArcLane.Run("axpby", output.Length, 0,
            In(a), In(b), InOut(output), output.Length, sa, sb);

    internal static void TransposeScale(float[] input, float[] output, int rows, int columns, bool transpose, float scale)
        => Tensor.ArcLane.Run("transpose_scale", input.Length, 0,
            In(input), InOut(output), rows, columns, transpose ? 1 : 0, scale);

    internal static void NewtonSchulz(float[] x, float[] next, float[] gram, float[] square,
        int rows, int columns, bool bf16, float a, float b, float c)
    {
        // These are the same three matrix products as the CUDA mixed-precision path.
        // The portable implementation emulates BF16 operands but accumulates in FP32.
        Gemm(x, x, gram, rows, rows, columns, false, true, bf16);
        Gemm(gram, gram, square, rows, rows, rows, false, true, bf16);
        var coefficient = new float[gram.Length];
        Tensor.ArcLane.Run("ns_coefficient", coefficient.Length, 0,
            In(gram), In(square), InOut(coefficient), rows, a, b, c);
        Gemm(coefficient, x, next, rows, columns, rows, false, false, bf16);
    }

    private static void Gemm(float[] a, float[] b, float[] c, int m, int n, int k, bool ta, bool tb, bool bf16)
        => Tensor.ArcLane.Run("gemm_precision", c.Length, 0,
            In(a), In(b), InOut(c), m, n, k, ta ? 1 : 0, tb ? 1 : 0, bf16 ? 1 : 0);

    internal static float Clip(IEnumerable<Parameter> parameters, float maxNorm)
    {
        if (Tensor.ArcResident) return ClipResident(parameters, maxNorm);
        float[][] gradients = parameters.Where(p => p.T.HasGradientBuffer).Select(p => p.T.GradientBuffer).ToArray();
        bool roundBFloat16 = TensorExecutionContext.ActivePrecisionPolicy?.Mode
            == NNtrain.Runtime.Execution.PrecisionMode.Mix8_16;
        if (roundBFloat16)
            foreach (float[] gradient in gradients)
                for (int i = 0; i < gradient.Length; i++)
                    gradient[i] = TensorStorageCodec.RoundToBFloat16(gradient[i]);
        float[] partials = gradients.Select(g => Sum(g, squared: true)).ToArray();
        float norm = MathF.Sqrt(Sum(partials));
        if (norm > maxNorm)
            foreach (float[] gradient in gradients)
                Tensor.ArcAccumulate(gradient, gradient, maxNorm / (norm + 1e-6f), add: false);
        if (roundBFloat16 && norm > maxNorm)
            foreach (float[] gradient in gradients)
                for (int i = 0; i < gradient.Length; i++)
                    gradient[i] = TensorStorageCodec.RoundToBFloat16(gradient[i]);
        return norm;
    }

    private static float ClipResident(IEnumerable<Parameter> parameters, float maxNorm)
    {
        var lane = Tensor.ArcLane;
        Tensor[] tensors = parameters.Where(p => p.T.HasGradientBuffer).Select(p => p.T).ToArray();
        if (tensors.Length == 0) return 0;
        bool packedBFloat16 = TensorExecutionContext.ActivePrecisionPolicy?.Mode
            == NNtrain.Runtime.Execution.PrecisionMode.Mix8_16;
        using var sums = lane.Allocate(tensors.Length);
        using var result = lane.Allocate(3);
        for (int i = 0; i < tensors.Length; i++)
        {
            Tensor tensor = tensors[i];
            ReduceToSlot(packedBFloat16 ? tensor.ArcBFloat16Gradient() : tensor.ArcGradient(),
                tensor.Numel, sums, i, squared: true, packedBFloat16: packedBFloat16);
        }
        using var total = lane.Allocate(1);
        ReduceToSlot(sums, tensors.Length, total, 0, squared: false);
        lane.Run("resident_clip_total", 1, 0, total, result, lane.NumericStatus, 1, maxNorm);
        void Apply()
        {
            foreach (Tensor tensor in tensors)
                lane.Run(packedBFloat16 ? "resident_clip_apply_bf16" : "resident_clip_apply",
                    tensor.Numel, 0,
                    packedBFloat16 ? tensor.ArcBFloat16Gradient() : tensor.ArcGradient(),
                    result, tensor.Numel);
        }
        var host = new float[3];
        if (!ArcOptimizerFastPath.SkipUnneededClip) Apply();
        lane.Read(result, host);
        if (host[2] != 0 || !float.IsFinite(host[0]))
            throw new ArithmeticException("Arc gradient or BFP8 publication is non-finite; refusing the optimizer update.");
        if (ArcOptimizerFastPath.SkipUnneededClip && host[1] < 1f) Apply();
        return host[0];
    }

    private static void ReduceToSlot(ArcBuffer input, int n, ArcBuffer output, int slot,
        bool squared, bool packedBFloat16 = false)
    {
        var lane = Tensor.ArcLane;
        var buffers = new List<ArcBuffer>();
        try
        {
            do
            {
                int groups = (n + 255) / 256;
                var partials = lane.Allocate(groups); buffers.Add(partials);
                lane.Run(packedBFloat16 ? "reduce_sum_bf16" : "reduce_sum",
                    groups * 256L, 256, input, partials, n, squared ? 1 : 0,
                    new LocalMemory(1024));
                input = partials; n = groups; squared = false;
                packedBFloat16 = false;
            } while (n > 1);
            lane.CopyBytes(input, output, 0, slot * 4, 4);
        }
        finally { foreach (var buffer in buffers) buffer.Dispose(); }
    }

    internal static bool GradientsFinite(IEnumerable<Parameter> parameters)
    {
        var lane = Tensor.ArcLane;
        bool packedBFloat16 = TensorExecutionContext.ActivePrecisionPolicy?.Mode
            == NNtrain.Runtime.Execution.PrecisionMode.Mix8_16;
        using var status = lane.Allocate(1);
        lane.Run("resident_zero", 1, 0, status, 1);
        foreach (Parameter parameter in parameters)
            if (parameter.T.HasGradientBuffer)
                lane.Run(packedBFloat16 ? "resident_finite_bf16" : "resident_finite",
                    parameter.T.Numel, 0,
                    packedBFloat16 ? parameter.T.ArcBFloat16Gradient() : parameter.T.ArcGradient(),
                    status, parameter.T.Numel);
        var value = new int[1]; lane.ReadRaw(status, value);
        return value[0] == 0;
    }
}
