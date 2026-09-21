using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

partial class Tensor
{
    /// <summary>
    /// Select the typed-storage loss head only if every forward, dX and dW
    /// chunk is supported. A short tail never switches gradient policy midway.
    /// </summary>
    private bool TryArcPackedLossHead(Tensor weight, Tensor bias, int[] labels, int ignore, out Tensor result)
    {
        result = null!;
        if (!ArcResident || !ArcLane.Options.PackedMatrixStorage || !ArcUsesMixedMatrixOperands
            || DType == TensorDType.Float32 || weight.DType == TensorDType.Float32
            || Rank == 0 || weight.Rank != 2 || _shape[^1] <= 0 || weight._shape[1] != _shape[^1]) return false;
        int width = _shape[^1], rows = Numel / width, vocabulary = weight._shape[0];
        if (rows == 0 || vocabulary <= 0) return false;
        var lane = ArcLane;
        int capacity = Math.Min(rows, Math.Min(lane.Options.LossChunkRows,
            Math.Max(1, (32 * 1024 * 1024) / checked(vocabulary * 4))));
        bool Supports(int count) => ArcXmxStorageOperand.CanRun(lane, count, vocabulary, width)
            && ArcXmxStorageOperand.CanRun(lane, count, width, vocabulary)
            && ArcXmxStorageOperand.CanRun(lane, vocabulary, width, count);
        if (!Supports(capacity) || (rows % capacity != 0 && !Supports(rows % capacity))) return false;
        // Backward retains both weight layouts. Bound all simultaneously live
        // panel buffers, not merely each individual GEMM's operands.
        static long ABytes(int m, int k) => ((m + 7L) / 8) * ((k + 15L) / 16) * 256;
        static long BBytes(int n, int k) => ((n + 15L) / 16) * ((k + 15L) / 16) * 512;
        long retainedWeights = BBytes(vocabulary, width) + BBytes(width, vocabulary);
        long transient = Math.Max(ABytes(capacity, width), Math.Max(ABytes(capacity, vocabulary),
            ABytes(vocabulary, capacity) + BBytes(width, capacity)));
        if (retainedWeights + transient > 96L * 1024 * 1024) return false;
        result = ArcPackedLossHead(weight, bias, labels, ignore, capacity);
        return true;
    }

    private Tensor ArcPackedLossHead(Tensor weight, Tensor bias, int[] labels, int ignore, int capacity)
    {
        int width = _shape[^1], rows = Numel / width, vocabulary = weight._shape[0];
        if (labels.Length != rows) throw new ArgumentException("Target count does not match loss rows.");
        if (bias.Rank != 1 || bias.Numel != vocabulary) throw new ArgumentException("Loss-head bias dimensions do not match.");
        int valid = 0;
        foreach (int label in labels)
        {
            if (label == ignore) continue;
            if ((uint)label >= (uint)vocabulary) throw new ArgumentOutOfRangeException(nameof(labels));
            valid++;
        }
        int divisor = Math.Max(1, valid);
        var lane = ArcLane;
        var ids = lane.UploadRaw(labels);
        void Tile(ArcXmxStorageOperand input, ArcBuffer packedWeights, ArcBuffer biases,
            ArcBuffer logits, ArcBuffer stats, ArcBuffer losses, ArcBuffer targets, int first, int count)
        {
            // Only targets are copied; the input slice addresses its original
            // resident payload and scale origin directly during panel packing.
            lane.CopyBytes(ids, targets, first * 4, 0, count * 4);
            using (var packedInput = input.PackA(count, width, false))
                ArcXmxStorageOperand.GemmPanels(lane, packedInput, packedWeights, logits, count, vocabulary, width,
                    tb: true, bias: biases);
            lane.Run("round_bf16_values", count * vocabulary, 0, logits, count * vocabulary);
            lane.Run("cross_entropy_rows", count * 128L, 128, logits, targets, stats, losses,
                count, vocabulary, ignore, divisor, new LocalMemory(512));
        }
        try
        {
            float[] total = new float[1];
            using (var input = ArcMatrixOperand())
            using (var weights = weight.ArcMatrixOperand())
            using (var packedWeights = weights.PackB(vocabulary, width, true))
            using (var biases = bias.ArcUploadValues())
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
                    using var chunk = input.Slice(checked(first * width), checked(count * width));
                    Tile(chunk, packedWeights, biases, logits, stats, losses, targets, first, count);
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
                using var input = ArcMatrixOperand(); using var weights = weight.ArcMatrixOperand();
                // These two immutable weight panels live only during this
                // backward callback and are retired before optimizer commit.
                using var forwardWeights = weights.PackB(vocabulary, width, true);
                using var gradientWeights = weights.PackB(width, vocabulary, false);
                using var biases = bias.ArcUploadValues();
                using var logits = lane.Allocate(capacity * vocabulary); using var stats = lane.Allocate(capacity * 2);
                using var losses = lane.Allocate(capacity); using var targets = lane.Allocate(capacity);
                using var dLogits = lane.Allocate(capacity * vocabulary); using var dChunk = lane.Allocate(capacity * width);
                ArcBuffer dx = ArcGradient(), dw = weight.ArcGradient(), db = bias.ArcGradient();
                float upstream = result.GradientBuffer[0];
                bool inline = lane.Options.InlineMatrixGradient;
                for (int first = 0; first < rows; first += capacity)
                {
                    int count = Math.Min(capacity, rows - first);
                    using var chunk = input.Slice(checked(first * width), checked(count * width));
                    Tile(chunk, forwardWeights, biases, logits, stats, losses, targets, first, count);
                    lane.Run("cross_entropy_gradient", count * vocabulary, 0, logits, targets, stats, dLogits,
                        count, vocabulary, ignore, divisor, upstream);
                    if (!inline) lane.Run("matrix_gradient_bf16", count * vocabulary, 0,
                        dLogits, dLogits, dLogits, count * vocabulary, 0);
                    using var gradient = new ArcXmxStorageOperand(lane, dLogits, null, TensorDType.Float32,
                        checked(count * vocabulary));
                    using (var packedGradient = gradient.PackA(count, vocabulary, false))
                        ArcXmxStorageOperand.GemmPanels(lane, packedGradient, gradientWeights, dChunk, count, width, vocabulary);
                    lane.Run("copy_range", count * width, 0, dChunk, dx, 0, first * width, count * width, 1);
                    using (var packedGradient = gradient.PackA(vocabulary, count, true))
                    using (var packedInput = chunk.PackB(width, count, false))
                        ArcXmxStorageOperand.GemmPanels(lane, packedGradient, packedInput, dw, vocabulary, width, count,
                            ta: true, accumulate: true);
                    ArcBiasGradient(dLogits, null, db, count, vocabulary, inline);
                }
            };
            return result;
        }
        catch { ids.Dispose(); throw; }
    }
}
