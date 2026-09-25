using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

internal sealed partial class ArcXmxStorageOperand
{
    /// <summary>
    /// Pack the same FP32 row-major gradient for dX (rows x cols) and dW
    /// (cols x rows). Both results retain the existing A-panel layout and
    /// BF16 rounding. The caller owns both returned buffers.
    /// </summary>
    internal static (ArcBuffer Normal, ArcBuffer Transposed) PackDualGradientA(
        ArcExecutionLane lane, ArcBuffer gradient, int rows, int cols)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(cols);
        int normalElements = checked((int)(((rows + 7L) / 8) * ((cols + 15L) / 16) * 128));
        int transposedElements = checked((int)(((cols + 7L) / 8) * ((rows + 15L) / 16) * 128));
        ArcBuffer normal = lane.AllocateBytes(checked(normalElements * 2));
        try
        {
            ArcBuffer transposed = lane.AllocateBytes(checked(transposedElements * 2));
            try
            {
                lane.Run2D("xmx_storage_pack_a_dual_f32",
                    ((cols + 15L) / 16) * 256, (rows + 15L) / 16, 256, 1,
                    gradient, normal, transposed, rows, cols);
                return (normal, transposed);
            }
            catch { transposed.Dispose(); throw; }
        }
        catch { normal.Dispose(); throw; }
    }
}
