using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

partial class Tensor
{
    private readonly record struct AttentionMatrix(ArcBuffer Buffer, int Row, int Column, int Group, int Batch, int Head, int Offset)
    {
        internal AttentionMatrix Transpose() => this with { Row = Column, Column = Row };
    }

    internal Tensor ArcBatchedAttention(int batch, int sequence, int width, int heads, bool causal, TensorDType? resultDType = null)
    {
        var lane = ArcLane;
        bool blockIo = lane.Options.BlockIoAttention && lane.Options.XmxMatrices && lane.Device.SupportsXmx
            && lane.Device.MinimumSubgroupSize == 16 && lane.Device.Extensions.Split(' ').Contains("cl_intel_subgroup_local_block_io");
        int totalHeads = checked(batch * heads), d = width / heads;
        long bulkElements = totalHeads * ((sequence + 15L) / 16 * 16) * 32;
        bool bulkPanels = lane.Options.BulkAttentionQkPanels && lane.Options.DirectAttentionQk && lane.Options.PanelAttention
            && DType != TensorDType.Float32 && lane.Options.XmxMatrices && lane.Device.SupportsXmx
            && lane.Device.MinimumSubgroupSize == 16 && d == 32 && sequence >= 128
            && bulkElements * 4 <= 256L * 1048576;
        bool xmxProducts = lane.Options.XmxAttentionProducts && DType != TensorDType.Float32
            && lane.Options.XmxMatrices && lane.Device.SupportsXmx && lane.Device.MinimumSubgroupSize == 16;
        int maskMode = causal ? (lane.Options.CausalAttentionBounds ? 2 : 1) : 0;
        bool subgroupReduction = lane.Options.SubgroupAttentionReduction && lane.Options.XmxMatrices
            && lane.Device.SupportsXmx && lane.Device.MinimumSubgroupSize == 16;
        // Only measured fixed-length row caches are enabled. In particular an
        // explicit subgroup-reduction request keeps its old dispatch/fallback.
        bool cachedProbabilities = lane.Options.CachedAttentionProbabilities && !lane.Options.SubgroupAttentionReduction
            && lane.Device.SupportsXmx && lane.Device.MinimumSubgroupSize == 16 && sequence is 512 or 1024;
        string probabilityKernel = subgroupReduction ? "attention_probabilities_subgroup_candidate"
            : cachedProbabilities ? $"attention_probabilities_register_{sequence}" : "attention_probabilities";
        string derivativeKernel = subgroupReduction ? "attention_derivatives_subgroup_candidate" : "attention_derivatives";
        // Two FP32 score/derivative workspaces. A session budget lets more
        // independent heads share each launch without retaining T^2 per layer.
        int tileHeads = Math.Clamp((lane.Options.AttentionWorkspaceMiB * 1024 * 1024 / 2) / checked(sequence * sequence * 4), 1, totalHeads);
        int scoreElements = checked(tileHeads * sequence * sequence);
        // At T1024/D32, dP+dS benefit from a smaller working set than PV.
        // Keep forward occupancy unchanged and never exceed the caller's budget.
        int backwardTileHeads = lane.Options.CacheSizedAttentionBackward && sequence == 1024 && d == 32
            && lane.Device.SupportsXmx && lane.Device.MinimumSubgroupSize == 16 ? Math.Min(tileHeads, 4) : tileHeads;
        AttentionMatrix Qkv(ArcBuffer x, int component) => new(x, 3 * width, 1, 0, sequence * 3 * width, d, component * width);
        AttentionMatrix Activation(ArcBuffer x) => new(x, width, 1, 0, sequence * width, d, 0);
        AttentionMatrix Scores(ArcBuffer x) => new(x, sequence, 1, sequence * sequence, 0, 0, 0);
        void Qk(ArcBuffer input, ArcBuffer scores, int first, int count, ArcBuffer? bulkQ = null, ArcBuffer? bulkK = null)
        {
            // Only the measured D32 BF16 QK operation is eligible. FP32 QK,
            // other head widths, small sequences and non-SG16 devices retain
            // their original dispatch, as do explicit legacy-panel sessions.
            long paddedRows = ((sequence + 15L) / 16) * 16;
            long elements = count * paddedRows * 32;
            bool direct = lane.Options.DirectAttentionQk && lane.Options.PanelAttention
                && DType != TensorDType.Float32 && lane.Options.XmxMatrices
                && lane.Device.SupportsXmx && lane.Device.MinimumSubgroupSize == 16
                && d == 32 && sequence >= 128 && elements * 4 <= 16L * 1024 * 1024;
            if (!direct)
            {
                Gemm(Qkv(input, 0), Qkv(input, 1).Transpose(), Scores(scores), sequence, sequence, d, first, count,
                    bf16: DType != TensorDType.Float32, causalMode: 1);
                return;
            }
            // Panels are scoped to this head tile and this one QK call; neither
            // the backward closure nor the next layer retains a stale QKV pack.
            using var packedQ = bulkQ?.Borrow() ?? lane.AllocateBytes(checked((int)elements * 2));
            using var packedK = bulkK?.Borrow() ?? lane.AllocateBytes(checked((int)elements * 2));
            if (bulkQ is null)
                lane.Run("attention_qk_pack_combined_candidate", elements / 2, 256,
                    input, packedQ, packedK, sequence, width, heads, first, count);
            bool boundedCausal = causal && lane.Options.CausalAttentionBounds;
            int rowsPerTile = boundedCausal ? 256 : 128;
            string qkKernel = bulkQ is null
                ? boundedCausal ? "attention_qk_direct_256x32_candidate" : "attention_qk_direct_128x32_candidate"
                : boundedCausal ? "attention_qk_direct_256x32_offset" : "attention_qk_direct_128x32_offset";
            object[] qkArgs = bulkQ is null ? [packedQ, packedK, scores, sequence, d, 0, boundedCausal ? 1 : 0]
                : [packedQ, packedK, scores, sequence, d, 0, boundedCausal ? 1 : 0, first];
            lane.Run3D(qkKernel,
                ((sequence + 31L) / 32) * 16, ((sequence + rowsPerTile - 1L) / rowsPerTile) * 16, count, 16, 16, 1,
                qkArgs);
        }
        void Gemm(AttentionMatrix a, AttentionMatrix b, AttentionMatrix c, int m, int n, int k, int first, int count, bool add = false, bool bf16 = false, int causalMode = 0, bool fullPrecisionB = false)
        {
            if (xmxProducts && !bf16)
            {
                int columns = n <= 32 ? 32 : 64;
                int cm = causal && lane.Options.CausalAttentionBounds ? causalMode : 0;
                if (lane.Options.DirectXmxAttentionProducts)
                {
                    int depth = checked((k + 15) / 16 * 16);
                    int aElements = checked(count * ((m + 7) / 8 * 8) * depth);
                    int bPairs = checked(count * ((n + 15) / 16 * 16) * depth / 2);
                    using var packedA = lane.AllocateBytes(checked(aElements * 4));
                    using var packedB = lane.AllocateBytes(checked(bPairs * 4 * (fullPrecisionB ? 2 : 1)));
                    lane.Run("attention_products_pack_a", aElements, 256, a.Buffer, packedA, m, k,
                        a.Row, a.Column, a.Group, a.Batch, a.Head, a.Offset, heads, first, count, cm);
                    lane.Run("attention_products_pack_b", bPairs, 256, b.Buffer, packedB, n, k,
                        b.Row, b.Column, b.Group, b.Batch, b.Head, b.Offset, heads, first, count, fullPrecisionB ? 1 : 0);
                    lane.Run3D($"attention_products_direct_n{columns}_b{(fullPrecisionB ? 1 : 0)}",
                        ((n + columns - 1L) / columns) * 16, ((m + 127L) / 128) * 16, count, 16, 16, 1,
                        packedA, packedB, c.Buffer, m, n, k, c.Row, c.Column, c.Group, c.Batch, c.Head, c.Offset,
                        heads, first, add ? 1 : 0, cm);
                    return;
                }
                lane.Run3D($"attention_xmx_products_n{columns}_b{(fullPrecisionB ? 1 : 0)}",
                    ((n + columns - 1L) / columns) * 16, ((m + 63L) / 64) * 8, count, 16, 8, 1,
                    a.Buffer, b.Buffer, c.Buffer, m, n, k,
                    a.Row, a.Column, a.Group, a.Batch, a.Head, a.Offset,
                    b.Row, b.Column, b.Group, b.Batch, b.Head, b.Offset,
                    c.Row, c.Column, c.Group, c.Batch, c.Head, c.Offset, heads, first, add ? 1 : 0,
                    cm);
                return;
            }
            bool xmx = bf16 && lane.Options.XmxMatrices && lane.Device.SupportsXmx;
            bool narrow = lane.Options.WideMatrices && n <= 32;
            // Smaller row tiles improve dQ/dK/dV occupancy without slowing PV forward.
            bool compact = narrow && add && lane.Options.CompactAttentionTiles;
            int rowsPerTile = narrow && !compact ? 64 : 32;
            int sg = lane.Device.MinimumSubgroupSize;
            string kernel = xmx
                ? (lane.Options.PanelAttention && sg == 16 ? "attention_xmx_panel" : "attention_xmx")
                : compact ? (lane.Options.UnrolledAttentionTiles && sg == 16 ? "attention_gemm_compact_unrolled" : "attention_gemm_compact")
                : narrow ? "attention_gemm_narrow" : "attention_gemm";
            int columnsPerTile = 32;
            if (!xmx && lane.Options.TunedFp32Attention && m >= 128)
            {
                if (n > 32 && k <= 64)
                {
                    kernel = "attention_fp32_dp_64x64x16";
                    rowsPerTile = 64; columnsPerTile = 64;
                }
                else if (n <= 32 && a.Column == 1)
                {
                    kernel = "attention_fp32_n_64x32x32";
                    rowsPerTile = 64;
                }
                if (d == 32)
                {
                    string suffix = sequence % 64 == 0 ? "_aligned" : "_special";
                    if (causalMode == 1 && k == 32 && !add && a.Row == width && b.Offset == 2 * width)
                        kernel = "attention_fp32_dp_d32" + suffix;
                    else if (n == 32 && a.Column == 1 && a.Row == sequence && b.Row == 3 * width)
                        kernel = (add && c.Offset == 0 ? "attention_fp32_dq_d32" : "attention_fp32_pv_d32") + suffix;
                }
            }
            lane.Run3D(kernel,
                xmx ? ((n + 63L) / 64) * sg : ((n + columnsPerTile - 1L) / columnsPerTile) * 16,
                xmx ? ((m + 127L) / 128) * 16 : ((m + rowsPerTile - 1L) / rowsPerTile) * 16, count, xmx ? sg : 16, 16, 1,
                a.Buffer, b.Buffer, c.Buffer, m, n, k,
                a.Row, a.Column, a.Group, a.Batch, a.Head, a.Offset,
                b.Row, b.Column, b.Group, b.Batch, b.Head, b.Offset,
                c.Row, c.Column, c.Group, c.Batch, c.Head, c.Offset, heads, first, add ? 1 : 0,
                causal && lane.Options.CausalAttentionBounds ? causalMode : 0);
        }
        var stats = lane.Allocate(checked(totalHeads * sequence * 2));
        try
        {
            Tensor result;
            using (var input = ArcUploadValues(true))
            using (var output = lane.Allocate(checked(batch * sequence * width)))
            using (var p = lane.Allocate(scoreElements))
            using (var bulkQ = bulkPanels ? lane.AllocateBytes(checked((int)bulkElements * 2)) : null)
            using (var bulkK = bulkPanels ? lane.AllocateBytes(checked((int)bulkElements * 2)) : null)
            {
                if (bulkPanels) lane.Run("attention_qk_pack_combined_candidate", bulkElements / 2, 256,
                    input, bulkQ!, bulkK!, sequence, width, heads, 0, totalHeads);
                for (int first = 0; first < totalHeads; first += tileHeads)
                {
                    int count = Math.Min(tileHeads, totalHeads - first);
                    Qk(input, p, first, count, bulkQ, bulkK);
                    if (lane.Options.FusedAttentionPv && d <= 32 && sequence <= 1024)
                        lane.Run3D("attention_prob_pv_fused_candidate", 32, ((sequence + 7L) / 8) * 8, count, 32, 8, 1,
                            input, p, output, stats, sequence, width, heads, first, causal ? 1 : 0, new LocalMemory(8 * sequence * 4));
                    else
                    {
                        lane.Run(probabilityKernel, count * sequence * 64L, 64, p, stats, sequence, width, heads, first, maskMode, 0);
                        Gemm(Scores(p), Qkv(input, 2), Activation(output), sequence, d, sequence, first, count, causalMode: 2);
                    }
                }
                result = ArcDeviceResult(output, Rank == 3 ? [batch, sequence, width] : [sequence, width], [this], resultDType);
            }
            if (result.Node.IsDetached) { stats.Dispose(); return result; }
            result.Node.RegisterResource(stats);
            result.Node.BackwardAction = () => {
                using var input = ArcUploadValues(true);
                using var p = lane.Allocate(checked(backwardTileHeads * sequence * sequence));
                using var ds = lane.Allocate(checked(backwardTileHeads * sequence * sequence));
                using var bulkQ = bulkPanels ? lane.AllocateBytes(checked((int)bulkElements * 2)) : null;
                using var bulkK = bulkPanels ? lane.AllocateBytes(checked((int)bulkElements * 2)) : null;
                if (bulkPanels) lane.Run("attention_qk_pack_combined_candidate", bulkElements / 2, 256,
                    input, bulkQ!, bulkK!, sequence, width, heads, 0, totalHeads);
                ArcBuffer dy = result.ArcGradient(), dx = ArcGradient();
                for (int first = 0; first < totalHeads; first += backwardTileHeads)
                {
                    int count = Math.Min(backwardTileHeads, totalHeads - first);
                    Qk(input, p, first, count, bulkQ, bulkK);
                    lane.Run(probabilityKernel, count * sequence * 64L, 64, p, stats, sequence, width, heads, first, maskMode, 1);
                    if (lane.Options.FusedAttentionDq && d <= 32 && sequence <= 1024)
                        lane.Run3D("attention_dp_ds_dq_fused_candidate", 32, ((sequence + 7L) / 8) * 8, count, 32, 8, 1,
                            input, dy, p, ds, dx, sequence, width, heads, first, causal ? 1 : 0, new LocalMemory(8 * sequence * 4));
                    else
                    {
                        Gemm(Activation(dy), Qkv(input, 2).Transpose(), Scores(ds), sequence, sequence, d, first, count, causalMode: 1);
                        lane.Run(derivativeKernel, count * sequence * 64L, 64, p, ds, sequence, width, heads, maskMode);
                        Gemm(Scores(ds), Qkv(input, 1), Qkv(dx, 0), sequence, d, sequence, first, count, true, causalMode: 2);
                    }
                    if (lane.Options.FusedAttentionDkv && !xmxProducts)
                    {
                        bool specialized = d == 32 && sequence % 64 == 0;
                        int keyTile = specialized && !causal ? 64 : 32;
                        string dkvKernel = specialized
                            ? causal ? "attention_dkv_d32_k32_q32_causal" : "attention_dkv_d32_k64_q32_dense"
                            : "attention_dkv_fused_candidate";
                        int workRows = 16;
                        if (specialized && blockIo && lane.Options.AttentionDkvRows == 16)
                        {
                            keyTile = 32;
                            dkvKernel = causal ? "attention_dkv_block_slm_causal" : "attention_dkv_block_slm_dense";
                        }
                        if (specialized && lane.Options.AttentionDkvRows != 16 && lane.Options.XmxMatrices
                            && lane.Device.SupportsXmx && lane.Device.MinimumSubgroupSize == 16)
                        {
                            workRows = lane.Options.AttentionDkvRows; keyTile = 32;
                            dkvKernel = $"attention_dkv_reg{workRows}_{(causal ? "causal" : "dense")}";
                        }
                        lane.Run3D(dkvKernel, ((d + 31L) / 32) * 16, ((sequence + keyTile - 1L) / keyTile) * workRows, count, 16, workRows, 1,
                            input, dy, p, ds, dx, sequence, width, heads, first, causal ? 1 : 0);
                    }
                    else
                    {
                        Gemm(Scores(ds).Transpose(), Qkv(input, 0), Qkv(dx, 1), sequence, d, sequence, first, count, true, causalMode: 3);
                        Gemm(Scores(p).Transpose(), Activation(dy), Qkv(dx, 2), sequence, d, sequence, first, count, true, causalMode: 3, fullPrecisionB: true);
                    }
                }
            };
            return result;
        }
        catch { stats.Dispose(); throw; }
    }
}
