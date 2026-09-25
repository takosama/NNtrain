using NNtrain.Runtime.Execution;
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
        long logitsRowBudget = (long)lane.Options.LossLogitsWorkspaceMiB * 1024 * 1024
            / ((long)vocabulary * sizeof(float));
        int capacity = (int)Math.Min(rows, Math.Min((long)lane.Options.LossChunkRows,
            Math.Max(1L, logitsRowBudget)));
        bool Supports(int count) => ArcXmxStorageOperand.CanRun(lane, count, vocabulary, width)
            && ArcXmxStorageOperand.CanRun(lane, count, width, vocabulary)
            && ArcXmxStorageOperand.CanRun(lane, vocabulary, width, count);
        if (!Supports(capacity) || (rows % capacity != 0 && !Supports(rows % capacity))) return false;
        // Backward retains both weight layouts. Bound all simultaneously live
        // panel buffers, not merely each individual GEMM's operands.
        static long ABytes(int m, int k) => ((m + 7L) / 8) * ((k + 15L) / 16) * 256;
        static long BBytes(int n, int k) => ((n + 15L) / 16) * ((k + 15L) / 16) * 512;
        long retainedWeights = BBytes(vocabulary, width) + BBytes(width, vocabulary);
        long backwardPanels = lane.Options.FusedDualGradientPack
            ? ABytes(capacity, vocabulary) + ABytes(vocabulary, capacity) + BBytes(width, capacity)
            : Math.Max(ABytes(capacity, vocabulary),
                ABytes(vocabulary, capacity) + BBytes(width, capacity));
        long transient = Math.Max(ABytes(capacity, width), backwardPanels);
        if (retainedWeights + transient > (long)lane.Options.LossPanelWorkspaceMiB * 1024 * 1024)
            return false;
        result = ArcPackedLossHead(weight, bias, labels, ignore, capacity);
        return true;
    }

    /// <summary>
    /// Keep the whole loss-head logits only when the saved BF16 payload is
    /// bounded and the currently live allocations leave room for backward
    /// scratch. Only the LRU pool can evict cached buffers before allocating.
    /// </summary>
    internal static bool CanCacheArcLossHeadLogits(int rows, int vocabulary, long liveBytes,
        ulong globalBytes, ulong maximumAllocationBytes, long physicalBufferBudgetBytes = 0)
    {
        const long maximumRetainedBytes = 512L * 1024 * 1024;
        const long scratchReserveBytes = 1024L * 1024 * 1024;
        if (rows <= 0 || vocabulary <= 0 || liveBytes < 0 || globalBytes == 0
            || physicalBufferBudgetBytes < 0)
            return false;
        long rowBytes = (long)vocabulary * sizeof(ushort) + 2 * sizeof(float);
        if (rows > maximumRetainedBytes / rowBytes) return false;
        long logitsBytes = (long)rows * vocabulary * sizeof(ushort);
        long statsBytes = (long)rows * 2 * sizeof(float);
        if ((ulong)logitsBytes > maximumAllocationBytes
            || (ulong)statsBytes > maximumAllocationBytes) return false;
        long global = (long)Math.Min(globalBytes, (ulong)long.MaxValue);
        long admission = global / 100 * 85 + global % 100 * 85 / 100;
        if (physicalBufferBudgetBytes > 0)
            admission = Math.Min(admission, physicalBufferBudgetBytes);
        return liveBytes <= admission - scratchReserveBytes - logitsBytes - statsBytes;
    }

    internal static long ArcLossHeadAdmissionLiveBytes(long allocatedBytes, long cachedBytes,
        long retiredBytes, bool lruBufferPool) => lruBufferPool
            ? allocatedBytes
            : checked(allocatedBytes + cachedBytes + retiredBytes);

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
        bool cacheLogits = lane.Options.Mix8_16CachedLossLogits
            && TensorExecutionContext.ActivePrecisionPolicy?.Mode == PrecisionMode.Mix8_16
            && AutogradContext.IsRecordingEnabled
            && CanCacheArcLossHeadLogits(rows, vocabulary,
                ArcLossHeadAdmissionLiveBytes(lane.AllocatedBytes, lane.CachedBytes,
                    lane.RetiredBytes, lane.Options.LruBufferPool),
                lane.Device.GlobalMemoryBytes, lane.Device.MaximumAllocationBytes,
                lane.Options.PhysicalBufferBudgetBytes);
        ArcBuffer? savedLogits = null, savedStats = null;
        void Tile(ArcXmxStorageOperand input, ArcBuffer packedWeights, ArcBuffer biases,
            ArcBuffer logits, ArcBuffer stats, ArcBuffer losses, ArcBuffer targets, int first, int count)
        {
            // Only targets are copied; the input slice addresses its original
            // resident payload and scale origin directly during panel packing.
            lane.CopyBytes(ids, targets, first * 4, 0, count * 4);
            bool fusedRounding = lane.Options.FusedLossHeadLogitRound;
            using (var packedInput = input.PackA(count, width, false))
                ArcXmxStorageOperand.GemmPanels(lane, packedInput, packedWeights, logits, count, vocabulary, width,
                    tb: true, bias: biases, roundBf16Output: fusedRounding);
            if (!fusedRounding)
                lane.Run("round_bf16_values", count * vocabulary, 0, logits, count * vocabulary);
            lane.Run("cross_entropy_rows", count * 128L, 128, logits, targets, stats, losses,
                count, vocabulary, ignore, divisor, new LocalMemory(512));
        }
        try
        {
            if (cacheLogits)
            {
                savedLogits = lane.AllocateBytes(checked(rows * vocabulary * sizeof(ushort)));
                savedStats = lane.Allocate(checked(rows * 2));
            }
            float[] total = new float[1];
            using (var input = ArcMatrixOperand())
            using (var weights = weight.ArcMatrixOperand(cacheWeightPanels: true))
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
                    if (cacheLogits)
                    {
                        int elements = checked(count * vocabulary);
                        lane.Run("loss_head_pack_bf16_at", elements, 0,
                            logits, savedLogits!, elements, checked(first * vocabulary));
                        lane.CopyBytes(stats, savedStats!, 0, checked(first * 2 * sizeof(float)),
                            checked(count * 2 * sizeof(float)));
                    }
                    lane.Run("loss_total", 1, 0, losses, sum, count);
                }
                lane.Run("resident_validate_loss", 1, 0, sum, lane.NumericStatus);
                lane.Read(sum, total);
                if (!float.IsFinite(total[0])) throw new ArithmeticException("Arc loss or BFP8 publication is non-finite; refusing the training step.");
            }
            Tensor result = ArcResult(total, [1], [this, weight, bias], TensorDType.Float32);
            if (result.Node.IsDetached)
            {
                ids.Dispose(); savedLogits?.Dispose(); savedStats?.Dispose();
                return result;
            }
            result.Node.RegisterResource(ids);
            if (savedLogits is not null) result.Node.RegisterResource(savedLogits);
            if (savedStats is not null) result.Node.RegisterResource(savedStats);
            result.Node.BackwardAction = () => {
                using var input = ArcMatrixOperand(); using var weights = weight.ArcMatrixOperand(cacheWeightPanels: true);
                // Cached logits make the forward weight panel and logits GEMM
                // unnecessary during backward. The dX layout is still needed.
                using var forwardWeights = cacheLogits ? null : weights.PackB(vocabulary, width, true);
                using var gradientWeights = weights.PackB(width, vocabulary, false);
                using var biases = cacheLogits ? null : bias.ArcUploadValues();
                using var logits = lane.Allocate(capacity * vocabulary); using var stats = lane.Allocate(capacity * 2);
                using var losses = cacheLogits ? null : lane.Allocate(capacity);
                using var targets = lane.Allocate(capacity);
                using var dLogits = lane.Allocate(capacity * vocabulary); using var dChunk = lane.Allocate(capacity * width);
                ArcBuffer dx = ArcGradient(), dw = weight.ArcGradient(), db = bias.ArcGradient();
                float upstream = result.GradientBuffer[0];
                bool inline = lane.Options.InlineMatrixGradient;
                for (int first = 0; first < rows; first += capacity)
                {
                    int count = Math.Min(capacity, rows - first);
                    using var chunk = input.Slice(checked(first * width), checked(count * width));
                    if (cacheLogits)
                    {
                        int elements = checked(count * vocabulary);
                        lane.Run("loss_head_unpack_bf16_at", elements, 0,
                            savedLogits!, logits, elements, checked(first * vocabulary));
                        lane.CopyBytes(savedStats!, stats, checked(first * 2 * sizeof(float)), 0,
                            checked(count * 2 * sizeof(float)));
                        lane.CopyBytes(ids, targets, checked(first * sizeof(int)), 0,
                            checked(count * sizeof(int)));
                    }
                    else Tile(chunk, forwardWeights!, biases!, logits, stats, losses!, targets, first, count);
                    lane.Run("cross_entropy_gradient", count * vocabulary, 0, logits, targets, stats, dLogits,
                        count, vocabulary, ignore, divisor, upstream);
                    if (!inline) lane.Run("matrix_gradient_bf16", count * vocabulary, 0,
                        dLogits, dLogits, dLogits, count * vocabulary, 0);
                    using var gradient = new ArcXmxStorageOperand(lane, dLogits, null, TensorDType.Float32,
                        checked(count * vocabulary));
                    if (lane.Options.FusedDualGradientPack)
                    {
                        var dual = ArcXmxStorageOperand.PackDualGradientA(lane, dLogits, count, vocabulary);
                        using var packedGradient = dual.Normal;
                        using var packedTransposedGradient = dual.Transposed;
                        ArcXmxStorageOperand.GemmPanels(lane, packedGradient, gradientWeights, dChunk,
                            count, width, vocabulary);
                        lane.Run("copy_range", count * width, 0, dChunk, dx, 0, first * width, count * width, 1);
                        using var packedInput = chunk.PackB(width, count, false);
                        ArcXmxStorageOperand.GemmPanels(lane, packedTransposedGradient, packedInput, dw,
                            vocabulary, width, count, ta: true, accumulate: true);
                    }
                    else
                    {
                        using (var packedGradient = gradient.PackA(count, vocabulary, false))
                            ArcXmxStorageOperand.GemmPanels(lane, packedGradient, gradientWeights, dChunk,
                                count, width, vocabulary);
                        lane.Run("copy_range", count * width, 0, dChunk, dx, 0, first * width, count * width, 1);
                        using (var packedGradient = gradient.PackA(vocabulary, count, true))
                        using (var packedInput = chunk.PackB(width, count, false))
                            ArcXmxStorageOperand.GemmPanels(lane, packedGradient, packedInput, dw,
                                vocabulary, width, count, ta: true, accumulate: true);
                    }
                    ArcBiasGradient(dLogits, null, db, count, vocabulary, inline);
                }
            };
            return result;
        }
        catch { ids.Dispose(); savedLogits?.Dispose(); savedStats?.Dispose(); throw; }
    }
}
