using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

partial class Tensor
{
    private Tensor ArcResidentLinear(Tensor weight, Tensor bias, bool relu, TensorDType? dtype)
    {
        int ni = _shape[^1], no = weight._shape[0], rows = Numel / ni;
        if (weight.Rank != 2 || weight._shape[1] != ni || bias.Rank != 1 || bias.Numel != no)
            throw new ArgumentException("Arc linear dimensions do not match.");
        var lane = ArcLane;
        int[] shape = (int[])_shape.Clone(); shape[^1] = no;
        Tensor? fusedInference = TryArcFusedInferenceGemv(weight, bias, shape, relu, dtype);
        if (fusedInference is not null) return fusedInference;
        using var biases = bias.ArcUploadValues();
        int operandPrecision = DType != TensorDType.Float32 && weight.DType != TensorDType.Float32 ? 3 : 0;
        Tensor? result = TryArcInferenceGemv(weight, bias, biases, shape, relu, dtype)
            ?? TryArcInt8Bfp8Linear(weight, bias, biases, shape, relu, dtype)
            ?? TryArcFusedBfp8Linear(weight, bias, biases, shape, relu, dtype);
        if (result is null)
        {
            using var output = lane.Allocate(checked(rows * no));
            if (!lane.Options.PackedMatrixStorage || !TryArcPackedLinear(weight, output, biases, relu))
            {
                using var input = ArcUploadValues(true);
                using var weights = weight.ArcUploadValues(true);
                ArcMuonMath.Gemm(lane, input, weights, output, rows, no, ni, tb: true, bf16: operandPrecision, bias: biases, relu: relu);
            }
            result = ArcDeviceResult(output, shape, [this, weight, bias], dtype);
        }
        result.Node.BackwardAction = () => {
            ArcBuffer dy = result.ArcGradient(), dx = ArcGradient(), dw = weight.ArcGradient(), db = bias.ArcGradient();
            if (relu && operandPrecision == 3 && ArcUsesMixedMatrixOperands && lane.Options.InlineMatrixGradient
                && TryArcPackedReluBackward(weight, result, dy, dx, dw, db)) return;
            using var gate = relu ? result.ArcUploadValues() : null;
            bool mixed = operandPrecision == 3 && ArcUsesMixedMatrixOperands;
            bool inline = mixed && lane.Options.InlineMatrixGradient;
            using var encoded = ArcEncodeMatrixGradient(dy, gate, checked(rows * no), mixed && !inline);
            ArcBuffer? remainingGate = mixed && !inline ? null : gate;
            if (!mixed || !TryArcPackedLinearBackward(weight, encoded, dx, dw, remainingGate, relu))
            {
                using var x = ArcUploadValues(true);
                using var w = weight.ArcUploadValues(true);
                ArcMuonMath.Gemm(lane, encoded, w, dx, rows, ni, no, bf16: mixed ? 3 : 0, accumulate: true, gate: remainingGate, gateOperand: remainingGate is null ? 0 : 1);
                ArcMuonMath.Gemm(lane, encoded, x, dw, no, ni, rows, ta: true, bf16: mixed ? 3 : 0, accumulate: true, gate: remainingGate, gateOperand: remainingGate is null ? 0 : 1);
            }
            ArcBiasGradient(encoded, remainingGate, db, rows, no, inline);
        };
        return result;
    }

    private Tensor ArcResidentEmbedding(Tensor? positions, int[] indices, int[] shape, int sequence)
    {
        var lane = ArcLane;
        int width = _shape[^1], length = checked(indices.Length * width);
        using var table = ArcUploadValues();
        using var pos = positions?.ArcUploadValues();
        using var output = lane.Allocate(length);
        var ids = lane.UploadRaw(indices);
        try
        {
            lane.Run("embedding", length, 0, table, pos ?? table, ids, output, indices.Length, width, sequence, positions is null ? 0 : 1);
            Tensor result = ArcDeviceResult(output, shape, positions is null ? [this] : [this, positions]);
            if (result.Node.IsDetached) { ids.Dispose(); return result; }
            result.Node.RegisterResource(ids);
            result.Node.BackwardAction = () => {
                ArcBuffer gradient = ArcGradient();
                lane.Run("embedding_back", length, 0, result.ArcGradient(), ids, gradient,
                    positions?.ArcGradient() ?? gradient, indices.Length, width, sequence, positions is null ? 0 : 1);
            };
            return result;
        }
        catch { ids.Dispose(); throw; }
    }

    private Tensor ArcResidentDropout(Tensor? residual, uint seed, uint threshold, float scale)
    {
        var lane = ArcLane;
        using var x = ArcUploadValues(); using var r = residual?.ArcUploadValues(); using var output = lane.Allocate(Numel);
        lane.Run("dropout", Numel, 0, x, r ?? x, output, Numel, seed, threshold, scale, residual is null ? 0 : 1);
        Tensor result = ArcDeviceResult(output, _shape, residual is null ? [this] : [this, residual]);
        result.Node.BackwardAction = () => {
            ArcBuffer dy = result.ArcGradient();
            lane.Run("dropout_back", Numel, 0, dy, ArcGradient(), Numel, seed, threshold, scale);
            if (residual is not null) lane.Run("copy_scale", Numel, 0, dy, residual.ArcGradient(), Numel, 1f, 1);
        };
        return result;
    }

    private Tensor ArcResidentNorm(Tensor gamma, Tensor beta, float eps, Tensor? branch, float probability, Random? random)
    {
        int width = _shape[^1], rows = Numel / width;
        if (gamma.Numel != width || beta.Numel != width) throw new ArgumentException("LayerNorm parameter dimensions do not match.");
        if (branch is not null && !_shape.AsSpan().SequenceEqual(branch._shape)) throw ShapeMismatch(this, branch, "Arc residual");
        uint seed = probability == 0 ? 0 : NextDropoutSeed(random ?? Random.Shared);
        uint threshold = (uint)(probability * (uint.MaxValue + 1d)); float scale = 1f / (1f - probability);
        var lane = ArcLane;
        if (lane.Options.BlockResidualNorm && width <= 2048 && rows >= 512 && lane.Options.ParallelReductions)
            return ArcBlockResidualNorm(gamma, beta, eps, branch, seed, threshold, scale, width, rows);
        if (branch is not null && lane.Options.FusedPackedResidualNorm && lane.Options.PackedNormInput
            && lane.Options.OrderedTiledNorm && lane.Options.ParallelReductions && width == 512 && rows >= 512
            && lane.Options.XmxMatrices && lane.Device.SupportsXmx && lane.Device.MinimumSubgroupSize == 16)
            return ArcFusedPackedResidualNorm(gamma, beta, eps, branch, seed, threshold, scale, width, rows);
        // SG16 coalesces each row's loads but keeps the original increasing-
        // channel sum order. Four independent rows per workgroup need neither
        // scratch buffers nor barriers; no parallel reduction changes rounding.
        bool ordered = lane.Options.OrderedTiledNorm
            && (rows >= 128 || lane.Options.InferenceSmallRowNorm && !AutogradContext.IsRecordingEnabled)
            && width is >= 64 and <= 2048
            && lane.Options.XmxMatrices && lane.Device.SupportsXmx && lane.Device.MinimumSubgroupSize == 16;
        long normWork = ordered ? ((rows + 3L) / 4) * 64 : rows;
        int normLocal = ordered ? 64 : 0;
        Tensor[] parents = branch is null ? [this, gamma, beta] : [this, branch, gamma, beta];
        bool directBf16Output = ordered && lane.Options.Mix8_16DirectBf16NormOutput
            && ArcMayPublishBFloat16Activation(parents);
        bool fusedResidualBackward = ordered && branch is not null
            && lane.Options.Mix8_16FusedNormResidualBackward
            && TensorExecutionContext.ActivePrecisionPolicy is
                { Mode: NNtrain.Runtime.Execution.PrecisionMode.Mix8_16,
                  NonWeightSelection: NNtrain.Runtime.Execution.NonWeightSelectionPolicy.FastestAvailable }
            && !ReferenceEquals(this, gamma) && !ReferenceEquals(this, beta)
            && !ReferenceEquals(branch, gamma) && !ReferenceEquals(branch, beta);
        bool parallelNorm = ordered && lane.Options.Mix8_16ParallelNormReduction
            && TensorExecutionContext.ActivePrecisionPolicy is
                { Mode: NNtrain.Runtime.Execution.PrecisionMode.Mix8_16,
                  NonWeightSelection: NNtrain.Runtime.Execution.NonWeightSelectionPolicy.FastestAvailable };
        bool fusedNormParameterGradients = fusedResidualBackward && parallelNorm
            && lane.Options.Mix8_16FusedNormParameterGradients
            && lane.Options.ParallelReductions && rows >= 512 && width == 512;
        ArcBuffer Input()
        {
            if (branch is not null && lane.Options.PackedNormInput)
                return ArcPackedResidualInput(branch, seed, threshold, scale);
            ArcBuffer value = ArcUploadValues();
            if (branch is null) return value;
            using (value)
            using (var other = branch.ArcUploadValues())
            {
                var sum = lane.Allocate(Numel);
                try { lane.Run("dropout", Numel, 0, other, value, sum, Numel, seed, threshold, scale, 1); return sum; }
                catch { sum.Dispose(); throw; }
            }
        }
        using var input = Input(); using var g = gamma.ArcUploadValues(); using var b = beta.ArcUploadValues();
        ArcBuffer output = directBf16Output
            ? lane.AllocateBytes(checked(Numel * sizeof(ushort))) : lane.Allocate(Numel);
        bool outputOwnedByResult = false;
        var stats = lane.Allocate(checked(2 * rows));
        try
        {
            lane.Run(directBf16Output && parallelNorm ? "norm_row_sg16_w64_parallel_bf16"
                    : directBf16Output ? "norm_row_sg16_w64_direct_bf16"
                    : ordered ? "norm_row_sg16_w64_candidate" : "norm", normWork, normLocal,
                input, g, b, output, stats, rows, width, eps);
            Tensor result = directBf16Output
                ? ArcDeviceBFloat16Result(output, _shape, parents)
                : ArcDeviceResult(output, _shape, parents);
            if (directBf16Output) outputOwnedByResult = true;
            if (result.Node.IsDetached) { stats.Dispose(); return result; }
            result.Node.RegisterResource(stats);
            result.Node.BackwardAction = () => {
                using var x = Input(); using var gammaValue = gamma.ArcUploadValues();
                ArcBuffer dy = result.ArcGradient();
                if (fusedResidualBackward)
                {
                    if (fusedNormParameterGradients)
                    {
                        int fourRowGroups = (rows + 3) / 4, groups = (rows + 255) / 256;
                        using var fourRowPartials = lane.Allocate(checked(2 * fourRowGroups * width));
                        using var tilePartials = lane.Allocate(checked(2 * groups * width));
                        lane.Run("norm_dx_row_sg16_w64_parallel_residual_parameter_bf16",
                            normWork, normLocal,
                            x, gammaValue, dy, stats, ArcGradient(), branch!.ArcGradient(),
                            fourRowPartials, rows, width, seed, threshold, scale);
                        lane.Run2D("norm_parameter_4row_partials_256row",
                            ((width + 31L) / 32) * 32, groups * 8L, 32, 8,
                            fourRowPartials, tilePartials, rows, width);
                        lane.Run("gradient_rows_finish", width, 0, tilePartials,
                            beta.ArcGradient(), gamma.ArcGradient(), groups, width, 1);
                    }
                    else
                    {
                        lane.Run(parallelNorm ? "norm_dx_row_sg16_w64_parallel_residual_bf16"
                                : "norm_dx_row_sg16_w64_residual_accumulate_bf16", normWork, normLocal,
                            x, gammaValue, dy, stats, ArcGradient(), branch!.ArcGradient(),
                            rows, width, seed, threshold, scale);
                        ArcNormGradient(x, dy, stats, gamma.ArcGradient(), beta.ArcGradient(), rows, width);
                    }
                }
                else
                {
                    using var dx = lane.Allocate(Numel);
                    lane.Run(ordered ? "norm_dx_row_sg16_w64_candidate" : "norm_dx_set", normWork, normLocal,
                        x, gammaValue, dy, stats, dx, rows, width);
                    ArcNormGradient(x, dy, stats, gamma.ArcGradient(), beta.ArcGradient(), rows, width);
                    if (branch is not null && lane.Options.FusedNormGradient)
                        lane.Run("norm_residual_back_accumulate", Numel, 0, dx, ArcGradient(), branch.ArcGradient(),
                            Numel, seed, threshold, scale);
                    else
                    {
                        lane.Run("copy_scale", Numel, 0, dx, ArcGradient(), Numel, 1f, 1);
                        if (branch is not null)
                            lane.Run("dropout_back", Numel, 0, dx, branch.ArcGradient(), Numel, seed, threshold, scale);
                    }
                }
            };
            return result;
        }
        catch { stats.Dispose(); throw; }
        finally { if (!outputOwnedByResult) output.Dispose(); }
    }

    private Tensor ArcResidentAttention(int batch, int sequence, int width, int heads, bool causal)
    {
        if (CanUseArcFlashAttention(sequence, width, heads))
            return ArcFlashAttention(batch, sequence, width, heads, causal);
        if (ArcLane.Options.BatchedAttention && (long)sequence * sequence * 4 <= 32 * 1024 * 1024)
            return ArcBatchedAttention(batch, sequence, width, heads, causal);
        var lane = ArcLane;
        int groups = checked(batch * heads * sequence), length = checked(batch * sequence * width);
        bool coalesced = lane.Options.CoalescedAttention;
        string kernel = coalesced ? "attention_coalesced" : "attention_streaming";
        ArcBuffer Pack(ArcBuffer source, int components)
        {
            if (!coalesced) return source.Borrow();
            var packed = lane.Allocate(checked(length * components));
            try
            {
                lane.Run2D("attention_pack", ((width / heads + 15L) / 16) * batch * heads * components * 16,
                    ((sequence + 15L) / 16) * 16, 16, 16, source, packed, batch, sequence, width, heads, components);
                return packed;
            }
            catch { packed.Dispose(); throw; }
        }
        using var source = ArcUploadValues(true); using var input = Pack(source, 3); using var output = lane.Allocate(length);
        var stats = lane.Allocate(checked(groups * 2));
        try
        {
            lane.Run(kernel, (long)groups * 64, 64, input, output, stats,
                batch, sequence, width, heads, causal ? 1 : 0, new LocalMemory(checked((sequence + 64) * 4)));
            Tensor result = ArcDeviceResult(output, Rank == 3 ? [batch, sequence, width] : [sequence, width], [this]);
            if (result.Node.IsDetached) { stats.Dispose(); return result; }
            result.Node.RegisterResource(stats);
            result.Node.BackwardAction = () => {
                using var values = ArcUploadValues(true); using var x = Pack(values, 3); using var delta = lane.Allocate(groups);
                using var dy = Pack(result.ArcGradient(), 1);
                ArcBuffer dx = ArcGradient();
                lane.Run(kernel + "_dq", (long)groups * 64, 64, x, dy, stats, dx, delta,
                    batch, sequence, width, heads, causal ? 1 : 0, new LocalMemory(checked((sequence * 2 + 64) * 4)));
                lane.Run(kernel + "_dkv", (long)groups * 64, 64, x, dy, stats, delta, dx,
                    batch, sequence, width, heads, causal ? 1 : 0, new LocalMemory(checked(sequence * 2 * 4)));
            };
            return result;
        }
        catch { stats.Dispose(); throw; }
    }

    private Tensor ArcResidentAdd(Tensor right)
    {
        using var x = ArcUploadValues(); using var r = right.ArcUploadValues(); using var y = ArcLane.Allocate(Numel);
        ArcLane.Run("add", Numel, 0, x, r, y, Numel);
        Tensor result = ArcDeviceResult(y, _shape, [this, right]);
        result.Node.BackwardAction = () => {
            ArcLane.Run("copy_scale", Numel, 0, result.ArcGradient(), ArcGradient(), Numel, 1f, 1);
            ArcLane.Run("copy_scale", Numel, 0, result.ArcGradient(), right.ArcGradient(), Numel, 1f, 1);
        };
        return result;
    }

    private Tensor ArcResidentSlice(int first, int count, int[] shape)
    {
        using var x = ArcUploadValues(); using var y = ArcLane.Allocate(count);
        ArcLane.Run("copy_range", count, 0, x, y, first, 0, count, 0);
        Tensor result = ArcDeviceResult(y, shape, [this]);
        result.Node.BackwardAction = () => ArcLane.Run("range_back", count, 0, result.ArcGradient(), ArcGradient(), first, count);
        return result;
    }

    private Tensor ArcResidentCrossEntropy(int[] labels, int rows, int columns, int ignore, int valid, float smoothing)
    {
        var lane = ArcLane;
        using var x = ArcUploadValues(); using var losses = lane.Allocate(rows);
        var stats = lane.Allocate(checked(rows * 2)); var ids = lane.UploadRaw(labels);
        try
        {
            lane.Run("cross_entropy", rows, 0, x, ids, stats, losses, rows, columns, ignore, valid, smoothing);
            using var sum = lane.Allocate(1);
            lane.Run("resident_sum_slot", 1, 0, losses, sum, rows, 0);
            lane.Run("resident_validate_loss", 1, 0, sum, lane.NumericStatus);
            float[] host = new float[1]; lane.Read(sum, host);
            if (!float.IsFinite(host[0])) throw new ArithmeticException("Arc loss or BFP8 publication is non-finite; refusing the training step.");
            Tensor result = ArcResult(host, [1], [this], TensorDType.Float32);
            if (result.Node.IsDetached) { ids.Dispose(); stats.Dispose(); return result; }
            result.Node.RegisterResource(ids); result.Node.RegisterResource(stats);
            result.Node.BackwardAction = () => {
                using var values = ArcUploadValues();
                lane.Run("cross_entropy_back", Numel, 0, values, ids, stats, ArcGradient(), rows, columns, ignore, valid, smoothing, result.GradientBuffer[0]);
            };
            return result;
        }
        catch { ids.Dispose(); stats.Dispose(); throw; }
    }
}
