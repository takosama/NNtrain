using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

partial class Tensor
{
    private Tensor ArcResidentLossHead(Tensor weight, Tensor bias, int[] labels, int ignore)
    {
        if (TryArcPackedLossHead(weight, bias, labels, ignore, out Tensor packed)) return packed;
        int width = _shape[^1], rows = Numel / width, vocabulary = weight._shape[0];
        if (labels.Length != rows) throw new ArgumentException("Target count does not match loss rows.");
        int valid = 0;
        foreach (int label in labels)
        {
            if (label == ignore) continue;
            if ((uint)label >= (uint)vocabulary) throw new ArgumentOutOfRangeException(nameof(labels));
            valid++;
        }
        var lane = ArcLane;
        // Bound each logits/gradient scratch to 32 MiB while amortizing the
        // vocabulary-weight gradient's read/modify/write across more tokens.
        int capacity = Math.Min(rows, Math.Min(lane.Options.LossChunkRows, Math.Max(1, (32 * 1024 * 1024) / checked(vocabulary * 4))));
        int divisor = Math.Max(1, valid);
        var ids = lane.UploadRaw(labels);
        bool roundLogits = DType != TensorDType.Float32;
        void Tile(ArcBuffer input, ArcBuffer w, ArcBuffer b, ArcBuffer chunk, ArcBuffer logits,
            ArcBuffer stats, ArcBuffer losses, ArcBuffer targets, int first, int count)
        {
            lane.CopyBytes(input, chunk, first * width * 4, 0, count * width * 4);
            lane.CopyBytes(ids, targets, first * 4, 0, count * 4);
            ArcMuonMath.Gemm(lane, chunk, w, logits, count, vocabulary, width, tb: true,
                bf16: DType != TensorDType.Float32 && weight.DType != TensorDType.Float32 ? 3 : 0, bias: b);
            if (roundLogits) lane.Run("round_bf16_values", count * vocabulary, 0, logits, count * vocabulary);
            lane.Run("cross_entropy_rows", count * 128L, 128, logits, targets, stats, losses,
                count, vocabulary, ignore, divisor, new LocalMemory(512));
        }
        try
        {
            float[] total = new float[1];
            using (var input = ArcUploadValues(true))
            using (var w = weight.ArcUploadValues(true))
            using (var b = bias.ArcUploadValues())
            using (var chunk = lane.Allocate(capacity * width))
            using (var logits = lane.Allocate(capacity * vocabulary))
            using (var stats = lane.Allocate(capacity * 2))
            using (var losses = lane.Allocate(capacity))
            using (var targets = lane.Allocate(capacity))
            using (var sum = lane.Allocate(1))
            {
                lane.Run("resident_zero", 1, 0, sum, 1);
                for (int first = 0; first < rows; first += capacity)
                {
                    int count = Math.Min(capacity, rows - first);
                    Tile(input, w, b, chunk, logits, stats, losses, targets, first, count);
                    lane.Run("loss_total", 1, 0, losses, sum, count);
                }
                lane.Run("resident_validate_loss", 1, 0, sum, lane.NumericStatus);
                lane.Read(sum, total);
                if (!float.IsFinite(total[0])) throw new ArithmeticException("Arc loss or BFP8 publication is non-finite; refusing the training step.");
            }
            Tensor result = ArcResult(total, [1], [this, weight, bias], TensorDType.Float32);
            if (result.Node.IsDetached) { ids.Dispose(); return result; }
            result.Node.RegisterResource(ids);
            result.Node.BackwardAction = () => {
                using var input = ArcUploadValues(true); using var w = weight.ArcUploadValues(true); using var b = bias.ArcUploadValues();
                using var chunk = lane.Allocate(capacity * width); using var logits = lane.Allocate(capacity * vocabulary);
                using var stats = lane.Allocate(capacity * 2); using var losses = lane.Allocate(capacity);
                using var targets = lane.Allocate(capacity); using var dLogits = lane.Allocate(capacity * vocabulary);
                using var dChunk = lane.Allocate(capacity * width);
                ArcBuffer dx = ArcGradient(), dw = weight.ArcGradient(), db = bias.ArcGradient();
                float upstream = result.GradientBuffer[0];
                for (int first = 0; first < rows; first += capacity)
                {
                    int count = Math.Min(capacity, rows - first);
                    Tile(input, w, b, chunk, logits, stats, losses, targets, first, count);
                    lane.Run("cross_entropy_gradient", count * vocabulary, 0, logits, targets, stats, dLogits,
                        count, vocabulary, ignore, divisor, upstream);
                    bool mixed = DType != TensorDType.Float32 && weight.DType != TensorDType.Float32 && ArcUsesMixedMatrixOperands;
                    bool inline = mixed && lane.Options.InlineMatrixGradient;
                    if (mixed && !inline) lane.Run("matrix_gradient_bf16", count * vocabulary, 0, dLogits, dLogits, dLogits, count * vocabulary, 0);
                    ArcMuonMath.Gemm(lane, dLogits, w, dChunk, count, width, vocabulary, bf16: mixed ? 3 : 0);
                    lane.Run("copy_range", count * width, 0, dChunk, dx, 0, first * width, count * width, 1);
                    ArcMuonMath.Gemm(lane, dLogits, chunk, dw, vocabulary, width, count, ta: true, bf16: mixed ? 3 : 0, accumulate: true);
                    ArcBiasGradient(dLogits, null, db, count, vocabulary, inline);
                }
            };
            return result;
        }
        catch { ids.Dispose(); throw; }
    }
}
