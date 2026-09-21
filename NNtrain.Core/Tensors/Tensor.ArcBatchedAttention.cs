using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

partial class Tensor
{
    private readonly record struct AttentionMatrix(ArcBuffer Buffer, int Row, int Column, int Group, int Batch, int Head, int Offset)
    {
        internal AttentionMatrix Transpose() => this with { Row = Column, Column = Row };
    }

    private Tensor ArcBatchedAttention(int batch, int sequence, int width, int heads, bool causal)
    {
        var lane = ArcLane;
        int totalHeads = checked(batch * heads), d = width / heads;
        int maskMode = causal ? (lane.Options.CausalAttentionBounds ? 2 : 1) : 0;
        bool subgroupReduction = lane.Options.SubgroupAttentionReduction && lane.Options.XmxMatrices
            && lane.Device.SupportsXmx && lane.Device.MinimumSubgroupSize == 16;
        string probabilityKernel = subgroupReduction ? "attention_probabilities_subgroup_candidate" : "attention_probabilities";
        string derivativeKernel = subgroupReduction ? "attention_derivatives_subgroup_candidate" : "attention_derivatives";
        // Two FP32 score/derivative workspaces. A session budget lets more
        // independent heads share each launch without retaining T^2 per layer.
        int tileHeads = Math.Clamp((lane.Options.AttentionWorkspaceMiB * 1024 * 1024 / 2) / checked(sequence * sequence * 4), 1, totalHeads);
        int scoreElements = checked(tileHeads * sequence * sequence);
        AttentionMatrix Qkv(ArcBuffer x, int component) => new(x, 3 * width, 1, 0, sequence * 3 * width, d, component * width);
        AttentionMatrix Activation(ArcBuffer x) => new(x, width, 1, 0, sequence * width, d, 0);
        AttentionMatrix Scores(ArcBuffer x) => new(x, sequence, 1, sequence * sequence, 0, 0, 0);
        void Gemm(AttentionMatrix a, AttentionMatrix b, AttentionMatrix c, int m, int n, int k, int first, int count, bool add = false, bool bf16 = false, int causalMode = 0)
        {
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
            {
                for (int first = 0; first < totalHeads; first += tileHeads)
                {
                    int count = Math.Min(tileHeads, totalHeads - first);
                    Gemm(Qkv(input, 0), Qkv(input, 1).Transpose(), Scores(p), sequence, sequence, d, first, count, bf16: DType != TensorDType.Float32, causalMode: 1);
                    if (lane.Options.FusedAttentionPv && d <= 32 && sequence <= 1024)
                        lane.Run3D("attention_prob_pv_fused_candidate", 32, ((sequence + 7L) / 8) * 8, count, 32, 8, 1,
                            input, p, output, stats, sequence, width, heads, first, causal ? 1 : 0, new LocalMemory(8 * sequence * 4));
                    else
                    {
                        lane.Run(probabilityKernel, count * sequence * 64L, 64, p, stats, sequence, width, heads, first, maskMode, 0);
                        Gemm(Scores(p), Qkv(input, 2), Activation(output), sequence, d, sequence, first, count, causalMode: 2);
                    }
                }
                result = ArcDeviceResult(output, Rank == 3 ? [batch, sequence, width] : [sequence, width], [this]);
            }
            if (result.Node.IsDetached) { stats.Dispose(); return result; }
            result.Node.RegisterResource(stats);
            result.Node.BackwardAction = () => {
                using var input = ArcUploadValues(true);
                using var p = lane.Allocate(scoreElements); using var ds = lane.Allocate(scoreElements);
                ArcBuffer dy = result.ArcGradient(), dx = ArcGradient();
                for (int first = 0; first < totalHeads; first += tileHeads)
                {
                    int count = Math.Min(tileHeads, totalHeads - first);
                    Gemm(Qkv(input, 0), Qkv(input, 1).Transpose(), Scores(p), sequence, sequence, d, first, count, bf16: DType != TensorDType.Float32, causalMode: 1);
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
                    if (lane.Options.FusedAttentionDkv)
                    {
                        bool specialized = d == 32 && sequence % 64 == 0;
                        int keyTile = specialized && !causal ? 64 : 32;
                        string dkvKernel = specialized
                            ? causal ? "attention_dkv_d32_k32_q32_causal" : "attention_dkv_d32_k64_q32_dense"
                            : "attention_dkv_fused_candidate";
                        lane.Run3D(dkvKernel, ((d + 31L) / 32) * 16, ((sequence + keyTile - 1L) / keyTile) * 16, count, 16, 16, 1,
                            input, dy, p, ds, dx, sequence, width, heads, first, causal ? 1 : 0);
                    }
                    else
                    {
                        Gemm(Scores(ds).Transpose(), Qkv(input, 0), Qkv(dx, 1), sequence, d, sequence, first, count, true, causalMode: 3);
                        Gemm(Scores(p).Transpose(), Activation(dy), Qkv(dx, 2), sequence, d, sequence, first, count, true, causalMode: 3);
                    }
                }
            };
            return result;
        }
        catch { stats.Dispose(); throw; }
    }
}
