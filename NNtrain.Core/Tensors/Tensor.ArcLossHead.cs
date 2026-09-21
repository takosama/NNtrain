using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

partial class Tensor
{
    /// <summary>
    /// Recomputes bounded logit tiles in backward instead of retaining N*vocabulary
    /// logits and gradients. BF16 loss-head outputs retain the existing mixed contract.
    /// </summary>
    internal Tensor ArcLinearCrossEntropy(Tensor weight, Tensor bias, int[] labels, int ignore)
    {
        if (ArcResident) return ArcResidentLossHead(weight, bias, labels, ignore);
        int width = _shape[^1], rows = Numel / width, vocabulary = weight._shape[0];
        if (labels.Length != rows) throw new ArgumentException("Target count does not match loss rows.");
        int valid = 0;
        foreach (int label in labels)
        {
            if (label == ignore) continue;
            if ((uint)label >= (uint)vocabulary) throw new ArgumentOutOfRangeException(nameof(labels));
            valid++;
        }
        int divisor = Math.Max(1, valid);
        const int tileRows = 128;
        int capacity = Math.Min(tileRows, rows);
        var lane = ArcLane;
        bool roundLogits = DType != TensorDType.Float32;
        float[] total = new float[1];
        void ForwardTile(ArcBuffer input, ArcBuffer w, ArcBuffer b, ArcBuffer chunk, ArcBuffer logits,
            ArcBuffer stats, ArcBuffer losses, int first, int count)
        {
            lane.Run("copy_range", count * width, 0, input, chunk, first * width, 0, count * width, 0);
            ArcMuonMath.Gemm(lane, chunk, w, logits, count, vocabulary, width, tb: true, bias: b);
            if (roundLogits) lane.Run("round_bf16_values", count * vocabulary, 0, logits, count * vocabulary);
            lane.Run("cross_entropy_rows", count * 128L, 128, logits, In(labels.AsSpan(first, count).ToArray()), stats, losses,
                count, vocabulary, ignore, divisor, new LocalMemory(128 * 4));
        }
        using (var input = ArcUploadValues(true))
        using (var w = weight.ArcUploadValues(true))
        using (var b = bias.ArcUploadValues())
        using (var chunk = lane.Allocate(capacity * width))
        using (var logits = lane.Allocate(capacity * vocabulary))
        using (var stats = lane.Allocate(capacity * 2))
        using (var losses = lane.Allocate(capacity))
        using (var sum = lane.Upload(total))
        {
            for (int first = 0; first < rows; first += capacity)
            {
                int count = Math.Min(capacity, rows - first);
                ForwardTile(input, w, b, chunk, logits, stats, losses, first, count);
                lane.Run("loss_total", 1, 0, losses, sum, count);
            }
            lane.Read(sum, total);
        }
        Tensor result = ArcResult(total, [1], [this, weight, bias], TensorDType.Float32);
        result.Node.BackwardAction = () => {
            using var input = ArcUploadValues(true);
            using var w = weight.ArcUploadValues(true);
            using var b = bias.ArcUploadValues();
            using var chunk = lane.Allocate(capacity * width);
            using var logits = lane.Allocate(capacity * vocabulary);
            using var stats = lane.Allocate(capacity * 2);
            using var losses = lane.Allocate(capacity);
            using var dLogits = lane.Allocate(capacity * vocabulary);
            using var dx = lane.Upload(_grad);
            using var dw = lane.Upload(weight._grad);
            using var db = lane.Upload(bias._grad);
            using var dChunk = lane.Allocate(capacity * width);
            for (int first = 0; first < rows; first += capacity)
            {
                int count = Math.Min(capacity, rows - first);
                ForwardTile(input, w, b, chunk, logits, stats, losses, first, count);
                lane.Run("cross_entropy_gradient", count * vocabulary, 0, logits, In(labels.AsSpan(first, count).ToArray()),
                    stats, dLogits, count, vocabulary, ignore, divisor, result._grad[0]);
                bool mixed = ArcUsesMixedMatrixOperands && DType != TensorDType.Float32 && weight.DType != TensorDType.Float32;
                if (mixed) lane.Run("matrix_gradient_bf16", count * vocabulary, 0, dLogits, dLogits, dLogits, count * vocabulary, 0);
                ArcMuonMath.Gemm(lane, dLogits, w, dChunk, count, width, vocabulary, bf16: mixed ? 3 : 0);
                lane.Run("copy_range", count * width, 0, dChunk, dx, 0, first * width, count * width, 1);
                ArcMuonMath.Gemm(lane, dLogits, chunk, dw, vocabulary, width, count, ta: true, bf16: mixed ? 3 : 0, accumulate: true);
                lane.Run("linear_db_chunk", vocabulary, 0, dLogits, dLogits, db, count, vocabulary, 0, 0, count);
            }
            lane.Read(dx, _grad); lane.Read(dw, weight._grad); lane.Read(db, bias._grad);
        };
        return result;
    }
}
