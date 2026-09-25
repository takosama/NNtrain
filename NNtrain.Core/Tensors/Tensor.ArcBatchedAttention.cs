using static NNtrain.Arc.ArcExecutionLane;
using NNtrain.Runtime.Execution;

namespace NNtrain;

partial class Tensor
{
    internal static bool ArcUseAttentionRowDelta(int sequence, int headWidth, bool causal)
    {
        var lane = ArcLane;
        return AutogradContext.IsRecordingEnabled && lane.Options.Mix8_16AttentionRowDelta
            && sequence == 2048 && headWidth == 32 && causal
            && TensorExecutionContext.ActivePrecisionPolicy is
                { Mode: PrecisionMode.Mix8_16, AllowNonWeightReassociation: true }
            && lane.Options.XmxMatrices && lane.Device.SupportsXmx && lane.Device.MinimumSubgroupSize == 16
            && !lane.Options.FusedAttentionDpDs2048 && !lane.Options.SubgroupAttentionReduction;
    }

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
        bool fusedBfp8DecodeQk = lane.Options.FusedAttentionQkvDecodePack2048 && ArcResident
            && DType == TensorDType.Bfp8 && Bfp8Quantization?.GetEffectiveBlockSize(Numel) == 32
            && bulkPanels && sequence == 2048 && d == 32
            && Numel == checked(batch * sequence * 3 * width);
        var precisionPolicy = TensorExecutionContext.ActivePrecisionPolicy;
        bool fastXmxProducts = lane.Options.Mix8_16XmxAttentionProducts
            && precisionPolicy is { Mode: PrecisionMode.Mix8_16, AllowNonWeightReassociation: true }
            && (precisionPolicy.AllowedNonWeightComputeFormats & NumericFormatSet.BFloat16) != 0
            && lane.Options.DirectXmxAttentionProducts && sequence == 2048 && d == 32;
        bool xmxProducts = (lane.Options.XmxAttentionProducts || fastXmxProducts) && DType != TensorDType.Float32
            && lane.Options.XmxMatrices && lane.Device.SupportsXmx && lane.Device.MinimumSubgroupSize == 16;
        int maskMode = causal ? (lane.Options.CausalAttentionBounds ? 2 : 1) : 0;
        bool subgroupReduction = lane.Options.SubgroupAttentionReduction && lane.Options.XmxMatrices
            && lane.Device.SupportsXmx && lane.Device.MinimumSubgroupSize == 16;
        // Only measured fixed-length row caches are enabled. In particular an
        // explicit subgroup-reduction request keeps its old dispatch/fallback.
        bool cachedProbabilities = lane.Options.CachedAttentionProbabilities && !lane.Options.SubgroupAttentionReduction
            && lane.Device.SupportsXmx && lane.Device.MinimumSubgroupSize == 16
            && (sequence is 512 or 1024 || sequence == 2048 && lane.Options.CachedAttentionProbabilities2048);
        string probabilityKernel = subgroupReduction ? "attention_probabilities_subgroup_candidate"
            : cachedProbabilities ? $"attention_probabilities_register_{sequence}" : "attention_probabilities";
        string savedProbabilityKernel = sequence == 2048 && !subgroupReduction
            && lane.Options.SavedAttentionProbabilitiesDirect2048
            ? "attention_probabilities_saved_direct_2048" : probabilityKernel;
        if (lane.Options.Mix8_16NativeExpAttention && sequence == 2048 && d == 32
            && !cachedProbabilities && !subgroupReduction
            && lane.Device.SupportsXmx && lane.Device.MinimumSubgroupSize == 16
            && precisionPolicy is { Mode: PrecisionMode.Mix8_16, AllowNonWeightReassociation: true })
        {
            probabilityKernel = "attention_probabilities_native_exp_2048";
            savedProbabilityKernel = lane.Options.SavedAttentionProbabilitiesDirect2048
                ? "attention_probabilities_saved_native_exp_2048" : probabilityKernel;
        }
        string derivativeKernel = subgroupReduction ? "attention_derivatives_subgroup_candidate"
            : sequence == 2048 && d == 32 && lane.Options.AttentionDerivativePrefix16T2048
                && lane.Options.XmxMatrices && lane.Device.SupportsXmx && lane.Device.MinimumSubgroupSize == 16
                ? "attention_derivatives_prefix16_2048"
            : sequence == 2048 && d == 32 && lane.Options.AttentionDerivativePrefix8T2048
                && lane.Options.XmxMatrices && lane.Device.SupportsXmx && lane.Device.MinimumSubgroupSize == 16
                ? "attention_derivatives_prefix8_2048"
            : sequence == 2048 && d == 32 && lane.Options.AttentionDerivativePrefix4T2048
                && lane.Options.XmxMatrices && lane.Device.SupportsXmx && lane.Device.MinimumSubgroupSize == 16
                ? "attention_derivatives_prefix4_2048"
            : sequence == 2048 && lane.Options.CachedAttentionDerivatives2048
                && lane.Device.SupportsXmx && lane.Device.MinimumSubgroupSize == 16
                ? "attention_derivatives_register_2048" : "attention_derivatives";
        bool fusedDpDs2048 = sequence == 2048 && d == 32 && lane.Options.FusedAttentionDpDs2048
            && lane.Device.SupportsXmx && lane.Device.MinimumSubgroupSize == 16;
        bool rowDelta = ArcUseAttentionRowDelta(sequence, d, causal);
        bool packedMix8_16Backward = lane.Options.Mix8_16PackedAttentionBackward
            && TensorExecutionContext.ActivePrecisionPolicy?.Mode == NNtrain.Runtime.Execution.PrecisionMode.Mix8_16
            && sequence == 2048 && d == 32 && causal && lane.Options.CausalAttentionBounds
            && blockIo && lane.Options.FusedAttentionDkv && lane.Options.AttentionDkvRows == 16
            && lane.Options.TunedFp32Attention && !xmxProducts
            && !lane.Options.FusedAttentionDq && !fusedDpDs2048
            && !lane.Options.AttentionDqM32T2048 && !lane.Options.SubgroupAttentionReduction
            && lane.Options.AttentionDerivativePrefix8T2048 && !lane.Options.AttentionDerivativePrefix16T2048;
        bool bf16QkvActivations = lane.Options.Mix8_16Bf16QkvActivations
            && TensorExecutionContext.ActivePrecisionPolicy?.Mode == NNtrain.Runtime.Execution.PrecisionMode.Mix8_16
            && fusedBfp8DecodeQk && sequence == 2048 && d == 32 && causal
            && lane.Options.CausalAttentionBounds && blockIo && lane.Options.TunedFp32Attention
            && lane.Options.FusedAttentionDkv && lane.Options.AttentionDkvRows == 16
            && !xmxProducts && !fusedDpDs2048
            && !lane.Options.FusedAttentionDq && !lane.Options.AttentionDqM32T2048
            && !lane.Options.SubgroupAttentionReduction
            && lane.Options.DkvQueryMajorPitch2048 == 0;
        bool fusedRowDelta = rowDelta && packedMix8_16Backward && bf16QkvActivations
            && lane.Options.Mix8_16FusedAttentionDpDs;
        int pvRows = lane.Options.Mix8_16PvM128 ? 128 : 64;
        string bf16PvKernel = lane.Options.Mix8_16PvM128 ? "attention_mix8_16_bf16_pv_m128_d32_2048"
            : lane.Options.Mix8_16PvPackedSlm ? "attention_mix8_16_bf16_pv_bpair_d32_2048"
            : "attention_mix8_16_bf16_pv_d32_2048";
        int dqRows = packedMix8_16Backward && lane.Options.Mix8_16DqM128 ? 128 : 64;
        string bf16DqKernel = !packedMix8_16Backward ? "attention_mix8_16_bf16_dq_d32_2048"
            : lane.Options.Mix8_16DqM128 ? "attention_mix8_16_bf16_dq_packed_m128_2048"
            : lane.Options.Mix8_16DqPackedSlmBoth ? "attention_mix8_16_bf16_dq_packed_ab_2048"
            : lane.Options.Mix8_16DqPackedSlm ? "attention_mix8_16_bf16_dq_packed_bpair_2048"
            : "attention_mix8_16_bf16_dq_packed_2048";
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
        ArcBuffer AttentionInput(ArcBuffer? packedQ, ArcBuffer? packedK)
        {
            if (!fusedBfp8DecodeQk) return ArcUploadValues(true);
            var decoded = bf16QkvActivations
                ? lane.AllocateBytes(checked(Numel * sizeof(ushort))) : lane.Allocate(Numel);
            try
            {
                EnsureArcPacked();
                var resident = ArcOwner();
                lane.Run(bf16QkvActivations
                        ? "attention_bfp8_decode_bf16_qk_2048" : "attention_bfp8_decode_qk_2048_candidate",
                    Numel / 2, 256,
                    resident.Value!, resident.Scales!, decoded, packedQ!, packedK!,
                    batch, sequence, width, heads, 32);
                return decoded;
            }
            catch { decoded.Dispose(); throw; }
        }
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
                if (fastXmxProducts) fullPrecisionB = false;
                int columns = n <= 32 ? 32 : 64;
                int cm = causal && lane.Options.CausalAttentionBounds ? causalMode : 0;
                if (lane.Options.DirectXmxAttentionProducts && !(fastXmxProducts && lane.Options.Mix8_16XmxAttentionSlm))
                {
                    int depth = checked((k + 15) / 16 * 16);
                    int aElements = checked(count * ((m + 7) / 8 * 8) * depth);
                    int bPairs = checked(count * ((n + 15) / 16 * 16) * depth / 2);
                    using var packedA = lane.AllocateBytes(checked(aElements * (fastXmxProducts ? 2 : 4)));
                    using var packedB = lane.AllocateBytes(checked(bPairs * 4 * (fullPrecisionB ? 2 : 1)));
                    if (fastXmxProducts && a.Column != 1)
                        lane.Run3D("attention_products_pack_a_bf16_transpose",
                            ((k + 15L) / 16) * 16, ((m + 15L) / 16) * 16, count, 16, 16, 1,
                            a.Buffer, packedA, m, k, a.Row, a.Column, a.Group, a.Batch, a.Head, a.Offset,
                            heads, first, count, cm);
                    else
                        lane.Run(fastXmxProducts ? "attention_products_pack_a_bf16" : "attention_products_pack_a", aElements, 256, a.Buffer, packedA, m, k,
                            a.Row, a.Column, a.Group, a.Batch, a.Head, a.Offset, heads, first, count, cm);
                    lane.Run("attention_products_pack_b", bPairs, 256, b.Buffer, packedB, n, k,
                        b.Row, b.Column, b.Group, b.Batch, b.Head, b.Offset, heads, first, count, fullPrecisionB ? 1 : 0);
                    lane.Run3D(fastXmxProducts ? $"attention_products_direct_n{columns}_bf16"
                            : $"attention_products_direct_n{columns}_b{(fullPrecisionB ? 1 : 0)}",
                        ((n + columns - 1L) / columns) * 16, ((m + 127L) / 128) * 16, count, 16, 16, 1,
                        packedA, packedB, c.Buffer, m, n, k, c.Row, c.Column, c.Group, c.Batch, c.Head, c.Offset,
                        heads, first, add ? 1 : 0, cm);
                    return;
                }
                lane.Run3D(fastXmxProducts ? $"attention_products_slm_n{columns}_bf16"
                        : $"attention_xmx_products_n{columns}_b{(fullPrecisionB ? 1 : 0)}",
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
                    if (sequence == 2048 && lane.Options.AttentionDqM32T2048
                        && kernel == "attention_fp32_dq_d32_aligned")
                    {
                        kernel = "attention_fp32_dq_d32_t2048_m32k32";
                        rowsPerTile = 32;
                    }
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
        ArcBuffer? savedRaw = null;
        try
        {
            if (rowDelta) savedRaw = lane.AllocateBytes(checked(batch * sequence * width * 2));
            Tensor result;
            // Preserve the legacy allocation/queue order for the disabled A/B
            // path. The fused path needs panel owners before its decode launch.
            using (var earlyInput = fusedBfp8DecodeQk ? null : ArcUploadValues(true))
            using (var output = lane.Allocate(checked(batch * sequence * width)))
            using (var p = lane.Allocate(scoreElements))
            using (var bulkQ = bulkPanels ? lane.AllocateBytes(checked((int)bulkElements * 2)) : null)
            using (var bulkK = bulkPanels ? lane.AllocateBytes(checked((int)bulkElements * 2)) : null)
            using (var input = fusedBfp8DecodeQk ? AttentionInput(bulkQ, bulkK) : earlyInput!.Borrow())
            {
                if (bulkPanels && !fusedBfp8DecodeQk) lane.Run("attention_qk_pack_combined_candidate", bulkElements / 2, 256,
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
                        if (bf16QkvActivations)
                            lane.Run3D(bf16PvKernel, 16,
                                ((sequence + pvRows - 1L) / pvRows) * 16, count, 16, 16, 1,
                                p, input, output, sequence, width, heads, first);
                        else
                            Gemm(Scores(p), Qkv(input, 2), Activation(output), sequence, d, sequence, first, count, causalMode: 2);
                    }
                }
                if (savedRaw is not null)
                    lane.Run("attention_mix8_16_store_raw_bf16_2048", batch * (long)sequence * width, 256,
                        output, savedRaw, checked(batch * sequence * width));
                result = ArcDeviceResult(output, Rank == 3 ? [batch, sequence, width] : [sequence, width], [this], resultDType);
            }
            if (result.Node.IsDetached) { stats.Dispose(); savedRaw?.Dispose(); return result; }
            result.Node.RegisterResource(stats);
            if (savedRaw is not null) result.Node.RegisterResource(savedRaw);
            result.Node.BackwardAction = () => {
                using var earlyInput = fusedBfp8DecodeQk ? null : ArcUploadValues(true);
                using var p = lane.Allocate(checked(backwardTileHeads * sequence * sequence));
                using var ds = fusedRowDelta ? null : lane.Allocate(checked(backwardTileHeads * sequence * sequence));
                using var bulkQ = bulkPanels ? lane.AllocateBytes(checked((int)bulkElements * 2)) : null;
                using var bulkK = bulkPanels ? lane.AllocateBytes(checked((int)bulkElements * 2)) : null;
                using var input = fusedBfp8DecodeQk ? AttentionInput(bulkQ, bulkK) : earlyInput!.Borrow();
                if (bulkPanels && !fusedBfp8DecodeQk) lane.Run("attention_qk_pack_combined_candidate", bulkElements / 2, 256,
                    input, bulkQ!, bulkK!, sequence, width, heads, 0, totalHeads);
                ArcBuffer dy = result.ArcGradient(), dx = ArcGradient();
                using var delta = rowDelta ? lane.Allocate(checked(totalHeads * sequence)) : null;
                if (rowDelta)
                    lane.Run("attention_mix8_16_row_delta_bf16_2048", totalHeads * (long)sequence, 256,
                        dy, savedRaw!, delta!, sequence, width, heads, 0, totalHeads);
                void Derivatives(int first, int count)
                {
                    if (rowDelta)
                        lane.Run(packedMix8_16Backward ? "attention_mix8_16_row_derivatives_packed_2048"
                                : "attention_mix8_16_row_derivatives_fp32_2048",
                            count * (long)sequence * sequence, 256,
                            p, ds!, delta!, sequence, width, heads, first, count, maskMode);
                    else
                        lane.Run(packedMix8_16Backward ? "attention_derivatives_prefix8_pack_bf16_2048" : derivativeKernel,
                            count * sequence * 64L, 64, p, ds!, sequence, width, heads, maskMode);
                }
                for (int first = 0; first < totalHeads; first += backwardTileHeads)
                {
                    int count = Math.Min(backwardTileHeads, totalHeads - first);
                    Qk(input, p, first, count, bulkQ, bulkK);
                    lane.Run(savedProbabilityKernel, count * sequence * 64L, 64, p, stats, sequence, width, heads, first, maskMode, 1);
                    if (lane.Options.FusedAttentionDq && d <= 32 && sequence <= 1024)
                        lane.Run3D("attention_dp_ds_dq_fused_candidate", 32, ((sequence + 7L) / 8) * 8, count, 32, 8, 1,
                            input, dy, p, ds!, dx, sequence, width, heads, first, causal ? 1 : 0, new LocalMemory(8 * sequence * 4));
                    else
                    {
                        if (fusedRowDelta)
                        {
                            bool xmxDpDs = lane.Options.Mix8_16FusedAttentionDpDsXmx;
                            int workRows = xmxDpDs ? 8 : 16;
                            lane.Run3D(xmxDpDs ? "attention_mix8_16_bf16_dpds_xmx_d32_2048"
                                    : lane.Options.Mix8_16DpDsK32 ? "attention_mix8_16_bf16_dpds_k32_d32_2048"
                                        : "attention_mix8_16_bf16_dpds_d32_2048",
                                ((sequence + 63L) / 64) * 16, ((sequence + 63L) / 64) * workRows,
                                count, 16, workRows, 1, dy, input, p, delta!, sequence, width, heads, first);
                        }
                        else if (bf16QkvActivations)
                        {
                            lane.Run3D("attention_mix8_16_bf16_dp_d32_2048",
                                ((sequence + 63L) / 64) * 16,
                                ((sequence + 63L) / 64) * 16, count, 16, 16, 1,
                                dy, input, ds!, sequence, width, heads, first);
                            Derivatives(first, count);
                        }
                        else if (fusedDpDs2048)
                            lane.Run2D("attention_dp_ds_two_pass_d32_2048_candidate",
                                ((sequence + 15L) / 16) * 256, count, 256, 1,
                                input, dy, p, ds!, sequence, width, heads, first, maskMode);
                        else
                        {
                            Gemm(Activation(dy), Qkv(input, 2).Transpose(), Scores(ds!), sequence, sequence, d, first, count, causalMode: 1);
                            Derivatives(first, count);
                        }
                        if (bf16QkvActivations)
                            lane.Run3D(bf16DqKernel, 16,
                                ((sequence + dqRows - 1L) / dqRows) * 16, count, 16, 16, 1,
                                packedMix8_16Backward ? p : ds!, input, dx, sequence, width, heads, first);
                        else if (packedMix8_16Backward)
                            lane.Run3D("attention_dq_packed_bf16_2048", 16,
                                ((sequence + 63L) / 64) * 16, count, 16, 16, 1,
                                p, input, dx, sequence, width, heads, first, maskMode);
                        else
                            Gemm(Scores(ds!), Qkv(input, 1), Qkv(dx, 0), sequence, d, sequence, first, count, true, causalMode: 2);
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
                            if (causal && sequence == 2048 && lane.Options.ExperimentalOptimizationKernels
                                && lane.Options.DkvQueryMajorPitch2048 is 32 or 33)
                                dkvKernel = $"attention_dkv_slm_qmajor{lane.Options.DkvQueryMajorPitch2048}_causal";
                        }
                        if (specialized && lane.Options.AttentionDkvRows != 16 && lane.Options.XmxMatrices
                            && lane.Device.SupportsXmx && lane.Device.MinimumSubgroupSize == 16)
                        {
                            workRows = lane.Options.AttentionDkvRows; keyTile = 32;
                            dkvKernel = $"attention_dkv_reg{workRows}_{(causal ? "causal" : "dense")}";
                        }
                        if (bf16QkvActivations && packedMix8_16Backward)
                            lane.Run3D(lane.Options.Mix8_16DkvPackedSlm ? "attention_mix8_16_bf16_dkv_packed_slm_uint_2048"
                                    : "attention_mix8_16_bf16_dkv_packed_2048", 16,
                                ((sequence + 31L) / 32) * 16, count, 16, 16, 1,
                                input, dy, p, dx, sequence, width, heads, first);
                        else if (bf16QkvActivations)
                            lane.Run3D("attention_mix8_16_bf16_dkv_d32_2048", 16,
                                ((sequence + 31L) / 32) * 16, count, 16, 16, 1,
                                input, dy, p, ds!, dx, sequence, width, heads, first);
                        else if (packedMix8_16Backward)
                            lane.Run3D("attention_dkv_packed_bf16_2048_causal", 16,
                                ((sequence + 31L) / 32) * 16, count, 16, 16, 1,
                                input, dy, p, dx, sequence, width, heads, first);
                        else
                            lane.Run3D(dkvKernel, ((d + 31L) / 32) * 16, ((sequence + keyTile - 1L) / keyTile) * workRows, count, 16, workRows, 1,
                                input, dy, p, ds!, dx, sequence, width, heads, first, causal ? 1 : 0);
                    }
                    else
                    {
                        Gemm(Scores(ds!).Transpose(), Qkv(input, 0), Qkv(dx, 1), sequence, d, sequence, first, count, true, causalMode: 3);
                        Gemm(Scores(p).Transpose(), Activation(dy), Qkv(dx, 2), sequence, d, sequence, first, count, true, causalMode: 3, fullPrecisionB: true);
                    }
                }
            };
            return result;
        }
        catch { stats.Dispose(); savedRaw?.Dispose(); throw; }
    }
}
