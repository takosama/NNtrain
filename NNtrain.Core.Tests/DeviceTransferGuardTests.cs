using NNtrain.Runtime.Execution;
using Xunit;

public sealed class DeviceTransferGuardTests
{
    [Fact]
    public void RejectsNonScalarD2hBeforeItCanBeSubmitted()
    {
        using IDisposable scope = DeviceTransferGuard.EnterTrainingStep(
            cudaDeviceCount: 2);

        InvalidOperationException failure = Assert.Throws<InvalidOperationException>(
            () => DeviceTransferGuard.BeforeDeviceToHost(
                1024,
                "parameter materialization"));

        Assert.Contains("implicit D2H", failure.Message);
        DeviceTransferSnapshot snapshot = Assert.NotNull(
            DeviceTransferGuard.CurrentSnapshot);
        Assert.Equal(0, snapshot.DeviceToHostCopyCount);
        Assert.Equal(0, snapshot.DeviceToHostBytes);
    }

    [Fact]
    public void CountsBatchUploadsAndAllowsOnlyConstantScalarDownloads()
    {
        using IDisposable scope = DeviceTransferGuard.EnterTrainingStep(
            cudaDeviceCount: 1,
            maximumDeviceToHostCopies: 3);

        using (DeviceTransferGuard.AllowBatchHostToDevice())
        {
            DeviceTransferGuard.BeforeHostToDevice(4096, "batch input");
            DeviceTransferGuard.RecordHostToDevice(4096);
            DeviceTransferGuard.BeforeHostToDevice(1024, "batch target");
            DeviceTransferGuard.RecordHostToDevice(1024);
        }
        DeviceTransferGuard.BeforeDeviceToHost(4, "loss");
        DeviceTransferGuard.BeforeDeviceToHost(8, "gradient norm");
        DeviceTransferGuard.BeforeDeviceToHost(4, "finite status");

        DeviceTransferSnapshot snapshot = Assert.NotNull(
            DeviceTransferGuard.CurrentSnapshot);
        Assert.Equal(2, snapshot.HostToDeviceCopyCount);
        Assert.Equal(5120, snapshot.HostToDeviceBytes);
        Assert.Equal(3, snapshot.DeviceToHostCopyCount);
        Assert.Equal(16, snapshot.DeviceToHostBytes);
        Assert.Throws<InvalidOperationException>(() =>
            DeviceTransferGuard.BeforeDeviceToHost(4, "per-parameter stats"));
    }

    [Fact]
    public void RejectsUnclassifiedH2dBeforeItCanBeSubmitted()
    {
        using IDisposable scope = DeviceTransferGuard.EnterTrainingStep(
            cudaDeviceCount: 2);

        InvalidOperationException failure = Assert.Throws<
            InvalidOperationException>(() =>
                DeviceTransferGuard.BeforeHostToDevice(
                    4 * 1024 * 1024,
                    "optimizer state upload"));

        Assert.Contains("unclassified H2D", failure.Message);
        DeviceTransferSnapshot snapshot = Assert.NotNull(
            DeviceTransferGuard.CurrentSnapshot);
        Assert.Equal(0, snapshot.HostToDeviceCopyCount);
        Assert.Equal(0, snapshot.HostToDeviceBytes);
    }

    [Fact]
    public void GradientCollectiveTransportIsSeparateAndAggregatedByDevice()
    {
        using IDisposable scope = DeviceTransferGuard.EnterTrainingStep(2);

        DeviceTransferGuard.GradientCollectiveTransportReservation? first =
            DeviceTransferGuard.ReserveGradientCollectiveTransport(
                sourceDeviceIndex: 1,
                destinationDeviceIndex: 0,
                copyCount: 3,
                byteLength: 96);
        Assert.NotNull(first);
        first.Commit();
        DeviceTransferGuard.GradientCollectiveTransportReservation? second =
            DeviceTransferGuard.ReserveGradientCollectiveTransport(
                sourceDeviceIndex: 0,
                destinationDeviceIndex: 1,
                copyCount: 2,
                byteLength: 64);
        Assert.NotNull(second);
        second.Commit();

        DeviceTransferSnapshot ordinary = Assert.NotNull(
            DeviceTransferGuard.CurrentSnapshot);
        Assert.Equal(default, ordinary);
        DeviceTransferTransportSnapshot? transport =
            DeviceTransferGuard.GetCurrentTransportSnapshot(
                DeviceTransferTransportCategory.GradientCollective);
        Assert.NotNull(transport);
        Assert.Equal(5, transport.Totals.HostToDeviceCopyCount);
        Assert.Equal(160, transport.Totals.HostToDeviceBytes);
        Assert.Equal(5, transport.Totals.DeviceToHostCopyCount);
        Assert.Equal(160, transport.Totals.DeviceToHostBytes);
        Assert.Collection(
            transport.Devices,
            device0 =>
            {
                Assert.Equal(0, device0.DeviceIndex);
                Assert.Equal(3, device0.HostToDeviceCopyCount);
                Assert.Equal(96, device0.HostToDeviceBytes);
                Assert.Equal(2, device0.DeviceToHostCopyCount);
                Assert.Equal(64, device0.DeviceToHostBytes);
            },
            device1 =>
            {
                Assert.Equal(1, device1.DeviceIndex);
                Assert.Equal(2, device1.HostToDeviceCopyCount);
                Assert.Equal(64, device1.HostToDeviceBytes);
                Assert.Equal(3, device1.DeviceToHostCopyCount);
                Assert.Equal(96, device1.DeviceToHostBytes);
            });
    }

    [Fact]
    public void FailedOrDuplicateCollectiveReservationIsNotDoubleCounted()
    {
        using IDisposable scope = DeviceTransferGuard.EnterTrainingStep(2);

        DeviceTransferGuard.GradientCollectiveTransportReservation? abandoned =
            DeviceTransferGuard.ReserveGradientCollectiveTransport(
                sourceDeviceIndex: 1,
                destinationDeviceIndex: 0,
                copyCount: 4,
                byteLength: 128);
        Assert.NotNull(abandoned);
        DeviceTransferTransportSnapshot? beforeCommit =
            DeviceTransferGuard.GetCurrentTransportSnapshot(
                DeviceTransferTransportCategory.GradientCollective);
        Assert.NotNull(beforeCommit);
        Assert.Equal(default, beforeCommit.Totals);
        Assert.Empty(beforeCommit.Devices);

        DeviceTransferGuard.GradientCollectiveTransportReservation? committed =
            DeviceTransferGuard.ReserveGradientCollectiveTransport(
                sourceDeviceIndex: 1,
                destinationDeviceIndex: 0,
                copyCount: 4,
                byteLength: 128);
        Assert.NotNull(committed);
        committed.Commit();
        Assert.Throws<InvalidOperationException>(committed.Commit);

        DeviceTransferTransportSnapshot? afterCommit =
            DeviceTransferGuard.GetCurrentTransportSnapshot(
                DeviceTransferTransportCategory.GradientCollective);
        Assert.NotNull(afterCommit);
        Assert.Equal(4, afterCommit.Totals.HostToDeviceCopyCount);
        Assert.Equal(128, afterCommit.Totals.HostToDeviceBytes);
        Assert.Equal(4, afterCommit.Totals.DeviceToHostCopyCount);
        Assert.Equal(128, afterCommit.Totals.DeviceToHostBytes);
    }

    [Fact]
    public void BatchAuthorizationCannotLeakIntoNestedTrainingStep()
    {
        using IDisposable outer = DeviceTransferGuard.EnterTrainingStep(1);
        using IDisposable batch =
            DeviceTransferGuard.AllowBatchHostToDevice();
        using IDisposable inner = DeviceTransferGuard.EnterTrainingStep(1);

        Assert.Throws<InvalidOperationException>(() =>
            DeviceTransferGuard.BeforeHostToDevice(16, "nested upload"));
    }

    [Fact]
    public async Task ParallelWorkersShareOneStepBudget()
    {
        using IDisposable scope = DeviceTransferGuard.EnterTrainingStep(
            cudaDeviceCount: 2,
            maximumDeviceToHostCopies: 16);

        await Task.WhenAll(Enumerable.Range(0, 16).Select(_ => Task.Run(() =>
        {
            DeviceTransferGuard.BeforeDeviceToHost(4, "worker scalar");
        }, TestContext.Current.CancellationToken)));

        Assert.Equal(
            16,
            Assert.NotNull(DeviceTransferGuard.CurrentSnapshot)
                .DeviceToHostCopyCount);
    }

    [Fact]
    public void NestedOutOfOrderAndDoubleDisposeRestoreTheActiveBudget()
    {
        IDisposable outer = DeviceTransferGuard.EnterTrainingStep(
            cudaDeviceCount: 1,
            maximumDeviceToHostCopies: 2);
        IDisposable inner = DeviceTransferGuard.EnterTrainingStep(
            cudaDeviceCount: 1,
            maximumDeviceToHostCopies: 1);

        outer.Dispose();
        outer.Dispose();
        DeviceTransferGuard.BeforeDeviceToHost(4, "inner");
        Assert.Throws<InvalidOperationException>(() =>
            DeviceTransferGuard.BeforeDeviceToHost(4, "inner overflow"));

        inner.Dispose();
        inner.Dispose();
        Assert.Null(DeviceTransferGuard.CurrentSnapshot);
    }
}
