using NNtrain;
using Xunit;

public sealed class CudaCapturedGradientPublicationStateTests
{
    [Fact]
    public void CapturedPublicationRequiresBegunCurrentStepAndIsExactlyOnce()
    {
        var state = new CudaCapturedGradientPublicationState(2);
        state.BeginStep(17);

        Assert.Throws<InvalidOperationException>(() =>
            state.BeginCapturedPublication(17, 0, 3));
        state.MarkDeviceBegun(17, 0, 3);
        state.MarkDeviceBegun(17, 1, 7);
        Assert.Throws<InvalidOperationException>(() =>
            state.BeginCapturedPublication(16, 0, 3));
        Assert.Throws<InvalidOperationException>(() =>
            state.BeginCapturedPublication(18, 0, 3));

        state.BeginCapturedPublication(17, 0, 3);
        state.CompleteCapturedPublication(17, 0, 3);
        InvalidOperationException duplicate =
            Assert.Throws<InvalidOperationException>(() =>
                state.BeginCapturedPublication(17, 0, 3));
        Assert.Contains("twice", duplicate.Message);
        InvalidOperationException mixed =
            Assert.Throws<InvalidOperationException>(() =>
                state.EnterNotificationPath(17, 1, 7));
        Assert.Contains("mix", mixed.Message);

        state.BeginCapturedPublication(17, 1, 7);
        state.CompleteCapturedPublication(17, 1, 7);
        state.ValidateComplete(17);
        state.EndStep(17);
    }

    [Fact]
    public void CaptureRecordingSuppressesOtherPathsAndMustBeDiscarded()
    {
        var state = new CudaCapturedGradientPublicationState(2);
        state.BeginStep(23);
        state.MarkDeviceBegun(23, 0, 0);
        state.MarkDeviceBegun(23, 1, 1);
        state.BeginCaptureRecording(23, 0, 0);
        Assert.True(state.IsCaptureRecording(23, 0));
        state.EnterCaptureNotificationPath(23, 0, 0);

        Assert.Throws<InvalidOperationException>(() =>
            state.EnterNotificationPath(23, 1, 1));
        Assert.Throws<InvalidOperationException>(() =>
            state.BeginCapturedPublication(23, 1, 1));
        Assert.Throws<InvalidOperationException>(() =>
            state.ValidateComplete(23));
        Assert.Throws<InvalidOperationException>(() =>
            state.ValidateCaptureDiscard(23));

        state.EndCaptureRecording(23, 0, 0);
        state.ValidateCaptureDiscard(23);
        state.EndStep(23);

        state.BeginStep(24);
        state.MarkDeviceBegun(24, 0, 0);
        state.EnterNotificationPath(24, 0, 0);
        state.ValidateComplete(24);
        state.EndStep(24);
    }

    [Fact]
    public void MidPublicationFailureRequiresAbortGenerationBeforeReuse()
    {
        var state = new CudaCapturedGradientPublicationState(2);
        state.BeginStep(31);
        state.MarkDeviceBegun(31, 0, 0);
        state.MarkDeviceBegun(31, 1, 1);
        state.BeginCapturedPublication(31, 0, 0);
        state.FailCapturedPublication(31, 0);

        Assert.Throws<InvalidOperationException>(() =>
            state.BeginCapturedPublication(31, 1, 1));
        Assert.Throws<InvalidOperationException>(() =>
            state.ValidateComplete(31));
        state.EndStep(31);

        state.BeginStep(32);
        state.MarkDeviceBegun(32, 0, 0);
        state.MarkDeviceBegun(32, 1, 1);
        for (int slot = 0; slot < 2; slot++)
        {
            state.BeginCapturedPublication(32, slot, slot);
            state.CompleteCapturedPublication(32, slot, slot);
        }
        state.ValidateComplete(32);
        state.EndStep(32);
    }
}
