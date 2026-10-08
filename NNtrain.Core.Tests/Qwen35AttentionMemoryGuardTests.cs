using NNtrain.Arc;
using Xunit;

namespace NNtrain.Core.Tests;

public sealed class Qwen35AttentionMemoryGuardTests
{
    private const ulong AllocationLimit = 512UL * 1024 * 1024;

    [Fact]
    public void FullBatchAccountsForEverySimultaneouslyLiveBuffer()
    {
        // Scores alone need 2,048 bytes; positions, Q, normalized K and the
        // returned output make the actual peak 7,264 bytes.
        const long peak = (8 * 4 * 16 + 2 * 8 * 4 * 16 + 8 * 2 * 16 + 8 * 3) * 4;
        Assert.Equal(7264, peak);
        Assert.True(Qwen35Gpu.CanAttentionRows(0, 8, 4, 2, 16, AllocationLimit, peak));
        Assert.False(Qwen35Gpu.CanAttentionRows(0, 8, 4, 2, 16, AllocationLimit, peak - 1));
        Assert.False(Qwen35Gpu.CanAttentionRows(0, 8, 4, 2, 16, AllocationLimit, peak, 1));
    }

    [Fact]
    public void TilePlanShrinksForStagingAndLaterPrefixCapacityGrowth()
    {
        const int start = 13, rows = 33, heads = 4, kvHeads = 2, width = 16;
        const long available = 30_000;
        int[] plan = Qwen35Gpu.PlanAttentionRowsTiles(start, rows, 16,
            heads, kvHeads, width, AllocationLimit, available);
        Assert.NotEmpty(plan);
        Assert.Equal(rows, plan.Sum());
        Assert.Equal(11, plan[0]);
        Assert.Contains(plan, count => count < plan[0]);
        long staging = 4L * (plan[0] * (2 * heads * width + 2 * kvHeads * width)
            + rows * heads * width);
        int position = start;
        foreach (int count in plan)
        {
            Assert.InRange(count, 1, plan[0]);
            Assert.True(Qwen35Gpu.CanAttentionRows(position, count, heads, kvHeads, width,
                AllocationLimit, available, staging));
            position += count;
        }
        Assert.Equal(start + rows, position);
    }

    [Fact]
    public void InsufficientOuterStagingHeadroomRejectsAllTilesBeforeExecution()
    {
        // A bare final two-row operation fits, but its persistent Q/K/V tile
        // and 33-row combined output do not. The caller must use scalar mode.
        const long available = 13_335;
        Assert.True(Qwen35Gpu.CanAttentionRows(44, 2, 4, 2, 16, AllocationLimit, available));
        Assert.Empty(Qwen35Gpu.PlanAttentionRowsTiles(13, 33, 16, 4, 2, 16,
            AllocationLimit, available));
    }

    [Fact]
    public void PlentyOfHeadroomKeepsExistingTileSizesAndOneRowTail()
    {
        Assert.Equal(new[] { 16, 16, 1 }, Qwen35Gpu.PlanAttentionRowsTiles(13, 33, 16,
            4, 2, 16, AllocationLimit, 512L * 1024 * 1024));
        Assert.False(Qwen35Gpu.CanAttentionRows(8192, 256, 48, 8, 128,
            AllocationLimit, long.MaxValue)); // Existing 128 MiB score cap.
        Assert.Empty(Qwen35Gpu.PlanAttentionRowsTiles(0, 33, 16, 4, 2, 16,
            1024, long.MaxValue)); // Combined output exceeds allocation limit.
    }

    [Fact]
    public void RejectedDirectBatchLeavesBothKvCachesAndAllocationCountUntouched()
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc GPU required.");
        using var lane = new ArcExecutionLane(0, new()
        {
            Qwen35InferenceKernelsOnly = true, CacheProgramBinary = true
        });
        const int rows = 8, heads = 4, kvHeads = 2, width = 16, start = 3;
        using var query = lane.Upload(new float[rows * heads * width * 2]);
        using var key = lane.Upload(new float[rows * kvHeads * width]);
        using var value = lane.Upload(new float[rows * kvHeads * width]);
        using var norm = lane.Upload(Enumerable.Repeat(1f, width).ToArray());
        float[] initialKeys = Enumerable.Range(0, (start + rows) * kvHeads * width)
            .Select(i => (float)i / 100).ToArray();
        float[] initialValues = initialKeys.Select(x => -x).ToArray();
        using var keys = lane.Upload(initialKeys);
        using var values = lane.Upload(initialValues);
        const long missingOneByte = 7263;
        lane.SetExternalMemoryReservation(lane.EffectivePhysicalBufferBudgetBytes
            - lane.AllocatedBytes - missingOneByte);
        long requested = lane.RequestedBytes;
        Assert.False(Qwen35Gpu.CanAttentionRows(lane, start, rows, heads, kvHeads, width));
        Assert.Throws<InvalidOperationException>(() => Qwen35Gpu.AttentionRows(lane,
            query, key, value, norm, norm, keys, values, start, rows, heads, kvHeads, width,
            0, 10_000, 1e-5f, Enumerable.Range(start, rows).Select(Qwen35Position.Scalar).ToArray()));
        Assert.Equal(requested, lane.RequestedBytes);
        float[] actualKeys = new float[initialKeys.Length], actualValues = new float[initialValues.Length];
        lane.Read(keys, actualKeys);
        lane.Read(values, actualValues);
        Assert.Equal(initialKeys, actualKeys);
        Assert.Equal(initialValues, actualValues);
    }
}
