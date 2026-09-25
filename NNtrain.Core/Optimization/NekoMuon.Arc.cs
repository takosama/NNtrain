using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

public sealed partial class NekoMuon
{
    private readonly ArcOptimizerStateCache _arcState = new();
    /// <summary>One parameter's moments, normalization, NS and update share device buffers.</summary>
    private void UpdateArcParameter(int index, NekoMuonOptions options, float fastCorrection, float slowCorrection)
    {
        Parameter parameter = _parameters[index];
        NekoMuonParameterState state = _state.ParameterStates[index];
        var lane = Tensor.ArcLane;
        int n = parameter.T.Numel;
        bool resident = Tensor.ArcResident;
        bool bf16State = TensorExecutionContext.ActivePrecisionPolicy?.Mode
            == NNtrain.Runtime.Execution.PrecisionMode.Mix8_16;
        if (bf16State && !resident)
            throw new NotSupportedException("Arc mix8_16 NekoMuon requires resident tensors and physical BF16 optimizer state.");
        float[] gradient = resident ? [] : parameter.T.GradientBuffer;
        using var g = resident
            ? (bf16State ? parameter.T.ArcBFloat16Gradient() : parameter.T.ArcGradient()).Borrow()
            : lane.Upload(gradient.Length == 0 ? new float[n] : gradient);
        using var fast = resident
            ? (bf16State ? _arcState.GetBFloat16(_arcPackedFastMoments[index]) : _arcState.Get(state.FastMoment)).Borrow()
            : lane.Upload(state.FastMoment);
        using var slow = resident
            ? (bf16State ? _arcState.GetBFloat16(_arcPackedSlowMoments[index]) : _arcState.Get(state.SlowMoment)).Borrow()
            : lane.Upload(state.SlowMoment);
        using var fh = lane.Allocate(n);
        // Nesterov emits the same direction into both hats. They can share
        // one scratch buffer; after normalization it can hold the transpose.
        using var sh = ArcOptimizerFastPath.ShareNesterovHat && options.Nesterov
            ? fh.Borrow() : lane.Allocate(n);
        lane.Run(bf16State ? "moments_bf16_packed" : "moments", n, 0,
            g, fast, slow, fh, sh, n, options.BetaFast, options.BetaSlow,
            fastCorrection, slowCorrection, options.Nesterov ? 1 : 0);
        bool singleReduction = bf16State && options.Nesterov
            && ArcOptimizerFastPath.NesterovSingleReduction;
        float fastSumSquares;
        float confidenceRaw;
        if (singleReduction)
        {
            fastSumSquares = ArcMuonMath.SumSquares(lane, fh, n);
            confidenceRaw = ArcMuonMath.ConfidenceForIdenticalDirections(
                fastSumSquares, options.Epsilon);
        }
        else
            confidenceRaw = ArcMuonMath.Confidence(lane, fh, sh, n, options.Epsilon,
                out fastSumSquares);
        float confidence = Math.Clamp(options.Rho * state.Confidence + (1f - options.Rho) * confidenceRaw, 0, 1);
        bool runNs = _state.Step % options.NewtonSchulzInterval == 0;
        float depth = ForceFullNewtonSchulz && runNs ? options.MaxNewtonSchulzSteps
            : ResolveNewtonSchulzDepth(options, confidence, runNs);
        int whole = Math.Min(options.MaxNewtonSchulzSteps, (int)MathF.Floor(depth));
        float fraction = depth - whole;
        GetMatrixShape(parameter, out int originalRows, out int originalColumns);
        int rows = Math.Min(originalRows, originalColumns), columns = Math.Max(originalRows, originalColumns);
        bool transpose = originalRows > originalColumns;
        float inverseNorm = 1f / (MathF.Sqrt(singleReduction || (bf16State && ArcOptimizerFastPath.ReuseMuonNorm)
            ? fastSumSquares : ArcMuonMath.SumSquares(lane, fh, n)) + options.Epsilon);
        using var first = lane.Allocate(n);
        using var second = lane.Allocate(n);
        using var gram = lane.Allocate(rows * rows);
        using var square = lane.Allocate(rows * rows);
        using var coefficient = lane.Allocate(rows * rows);
        lane.Run("transpose_scale", n, 0, fh, first, originalRows, originalColumns, transpose ? 1 : 0, inverseNorm);
        ArcBuffer current = first, next = second;
        int bf16 = parameter.T.DType is TensorDType.BFloat16 or TensorDType.Bfp8 ? 3 : 0;
        void Iterate()
        {
            ArcMuonMath.Gemm(lane, current, current, gram, rows, rows, columns, tb: true, bf16: bf16);
            ArcMuonMath.Gemm(lane, gram, gram, square, rows, rows, rows, tb: true, bf16: bf16);
            lane.Run("ns_coefficient", rows * rows, 0, gram, square, coefficient, rows, NewtonSchulzA, NewtonSchulzB, NewtonSchulzC);
            ArcMuonMath.Gemm(lane, coefficient, current, next, rows, columns, rows, bf16: bf16);
        }
        for (int iteration = 0; iteration < whole; iteration++) { Iterate(); (current, next) = (next, current); }
        if (fraction > 0) { Iterate(); lane.Run("axpby", n, 0, current, next, current, n, 1f - fraction, fraction); }
        ArcBuffer update = current;
        if (transpose)
        {
            lane.Run("transpose_scale", n, 0, current, sh, rows, columns, 1, 1f);
            update = sh;
        }
        float[] master = resident ? [] : parameter.DataBuffer;
        using var weights = resident
            ? (bf16State ? parameter.T.ArcBFloat16Master() : parameter.T.ArcMaster()).Borrow()
            : lane.Upload(master);
        bool decay = parameter.WeightDecay == WeightDecayPolicy.Apply || (options.Decay1D && parameter.T.Rank == 1);
        float scale = MathF.Sqrt(MathF.Max(1, (float)originalRows / originalColumns));
        if (resident && bf16State)
            lane.Run("muon_weight_update_bf16_packed", n, 0, weights, update, n,
                decay ? 1f - options.LearningRate * options.WeightDecay : 1f, -options.LearningRate * scale);
        else
            lane.Run("axpby", n, 0, weights, update, weights, n,
                decay ? 1f - options.LearningRate * options.WeightDecay : 1f, -options.LearningRate * scale);
        // CPU state is the portable checkpoint authority. No intermediate directions or Gram matrices cross PCIe.
        if (resident) parameter.T.CompleteArcUpdate();
        else
        {
            lane.Read(fast, state.FastMoment); lane.Read(slow, state.SlowMoment); lane.Read(weights, master);
            parameter.CompleteUpdate();
        }
        _state.ParameterStates[index] = state with { Confidence = confidence };
    }
}
