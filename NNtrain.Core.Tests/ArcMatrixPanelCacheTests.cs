using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcMatrixPanelCacheTests
{
    [Theory]
    [InlineData(TensorPrecisionMode.Mix16_32)]
    [InlineData(TensorPrecisionMode.Mix8_32)]
    public void CacheReusesBothLayoutsAndInvalidatesHostOptimizerAndPrecisionWrites(TensorPrecisionMode precision)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        using var execution = Tensor.BeginArcExecution(precision: precision, options: new() { MatrixPanelCacheMiB = 1 });
        var lane = Tensor.ArcLane;
        var weight = new Tensor(Enumerable.Range(0,128*128).Select(i => (i%31-15)*.03f).ToArray(), [128,128]);
        weight.ConvertStorageInPlace(precision.ToStorageDType(), precision == TensorPrecisionMode.Mix8_32 ? Bfp8QuantizationDescriptor.Block(32) : null);
        ushort[] Pack(bool cached, bool transpose)
        {
            using var operand = weight.ArcMatrixOperand(cacheWeightPanels: cached);
            using var packed = operand.PackB(128,128,transpose);
            var bits = new ushort[128*128]; lane.ReadRaw(packed,bits); return bits;
        }
        foreach (bool transpose in new[]{false,true})
        {
            var expected = Pack(false,transpose);
            Assert.Equal(expected,Pack(true,transpose));
            long kernels = lane.KernelLaunchCount;
            Assert.Equal(expected,Pack(true,transpose));
            Assert.Equal(kernels,lane.KernelLaunchCount);
        }
        Assert.Equal(65536,lane.RetainedMatrixPanelBytes);
        using (var mutation = weight.BeginDataMutation()) mutation.Values[0] += 1f;
        Assert.Equal(0,lane.RetainedMatrixPanelBytes);
        Assert.Equal(Pack(false,true),Pack(true,true));
        using (var master = weight.ArcMaster().Borrow()) lane.Run("copy_scale",weight.Numel,0,master,master,weight.Numel,.75f,0);
        weight.CompleteArcUpdate();
        Assert.Equal(0,lane.RetainedMatrixPanelBytes);
        Assert.Equal(Pack(false,false),Pack(true,false));
        weight.ConvertStorageInPlace(TensorDType.Float32);
        Assert.Equal(0,lane.RetainedMatrixPanelBytes);
        Assert.Equal(Pack(false,true),Pack(true,true));
        weight.ReleaseArcGraph();
        Assert.Equal(0,lane.RetainedMatrixPanelBytes);
    }

    [Fact]
    public void BudgetAndFailureDoNotLeakReservations()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        using var lane = new ArcExecutionLane(options: new() { MatrixPanelCacheMiB = 1 });
        using var cache = new ArcMatrixPanelCache(lane);
        Assert.Throws<InvalidOperationException>(() => cache.GetOrCreate(1,1,false,16,() => throw new InvalidOperationException("injected")));
        Assert.Equal(0,lane.RetainedMatrixPanelBytes);
        using (var a=cache.GetOrCreate(256,2048,false,1048576,()=>lane.AllocateBytes(1048576))) { }
        Assert.Equal(1048576,lane.RetainedMatrixPanelBytes);
        using (var b=cache.GetOrCreate(16,16,true,1024,()=>lane.AllocateBytes(1024))) { }
        Assert.Equal(1048576,lane.RetainedMatrixPanelBytes);
        cache.Clear(); cache.Clear();
        Assert.Equal(0,lane.RetainedMatrixPanelBytes);
        cache.Dispose(); cache.Dispose();
        Assert.Throws<ObjectDisposedException>(() => cache.GetOrCreate(1,1,false,16,()=>lane.AllocateBytes(16)));
    }
}
