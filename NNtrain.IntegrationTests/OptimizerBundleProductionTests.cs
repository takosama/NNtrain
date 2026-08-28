using NNtrain.Cuda.Execution;
using NNtrain.Training.Optimization;
using Xunit;

namespace NNtrain.IntegrationTests;

public sealed class OptimizerBundleProductionTests
{
    [Fact]
    public void ClassificationFactoryNamesNekoMuonGroupsStably()
    {
        var model = new TransformerClassifier(
            seqLen: 2,
            dModel: 4,
            numHeads: 2,
            dHidden: 8,
            numLayers: 1,
            numClasses: 3,
            rng: new Random(13));
        var configuration = new TrainingConfiguration
        {
            Optimizer = TrainingConfiguration.NekoMuonOptimizer,
        };

        OptimizerBundle bundle = Program.CreateOptimizerBundle(
            model,
            configuration);

        Assert.Equal(
            ["hidden", "auxiliary"],
            bundle.Groups.Select(group => group.Name));
        Assert.Equal(
            ["hidden/0000", "auxiliary/0000"],
            bundle.Leaves.Select(leaf => leaf.Name));
        Assert.IsType<CompositeOptimizer>(bundle.RootOptimizer);
    }

    [Fact]
    public void WikiFactoryUsesFrozenLeavesForSchedulerAndCheckpointOrder()
    {
        var model = new GptRinWikiJp(
            BpeTokenizer.BaseVocabularySize,
            contextLength: 4,
            dModel: 8,
            numHeads: 2,
            dHidden: 16,
            numLayers: 1,
            rng: new Random(2));
        var configuration = new WikiTrainingConfiguration
        {
            Optimizer = WikiTrainingConfiguration.NekoMuonOptimizer,
            LearningRate = 4e-4f,
            AuxiliaryLearningRate = 2e-4f,
        };

        OptimizerBundle bundle =
            WikiLanguageModelCommand.CreateOptimizerBundle(
                model,
                configuration);
        var scheduler = lr_scheduler.WarmupCosineProgressLR(bundle, 20f);

        Assert.Equal(
            ["hidden", "auxiliary"],
            bundle.Groups.Select(group => group.Name));
        Assert.Equal(
            bundle.LeafOptimizers,
            OptimizerBundle.GetCheckpointLeafOptimizers(bundle));
        Assert.Equal(2, scheduler.step(0.1d).Count);
    }

    [Fact]
    public void WikiNekoMuonPreflightFailsBeforeModelOrOptimizerAllocation()
    {
        var configuration = new WikiTrainingConfiguration
        {
            Device = WikiTrainingConfiguration.CudaDevice,
            DeviceIndices = [0, 1],
            Optimizer = WikiTrainingConfiguration.NekoMuonOptimizer,
            NekoMuonNewtonSchulzInterval = 1,
            NekoMuonNewtonSchulzDepthMode = "fixed",
            NekoMuonNewtonSchulzDepth = 5f,
        };
        bool allocationStarted = false;
        var capabilities = new CudaKernelCapabilities(
            8,
            6,
            CudaKernelFeature.TensorCores
                | CudaKernelFeature.BFloat16);

        NotSupportedException error = Assert.Throws<NotSupportedException>(
            () =>
            {
                WikiLanguageModelCommand.PreflightCudaOptimizer(
                    configuration,
                    TensorPrecisionMode.Mix16_32,
                    _ => capabilities);
                allocationStarted = true;
            });

        Assert.False(allocationStarted);
        Assert.Contains("BlockReducedMuon", error.Message);
    }
}
