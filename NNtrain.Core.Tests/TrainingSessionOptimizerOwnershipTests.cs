using NNtrain.Runtime.Execution;
using NNtrain.Training.Execution;
using NNtrain.Training.Optimization;
using Xunit;

namespace NNtrain.Core.Tests;

public sealed class TrainingSessionOptimizerOwnershipTests
{
    [Fact]
    public void SessionFreezesOneNamedOptimizerAuthority()
    {
        var first = new AdamW([CreateParameter("first")]);
        var second = new NekoMuon([CreateParameter("second")]);
        var bundle = new OptimizerBundle(
        [
            new OptimizerGroup("auxiliary", first),
            new OptimizerGroup("hidden", second),
        ]);
        using var execution = new ExecutionSession(new ExecutionOptions());
        using var session = new TrainingSession(execution);

        Assert.Same(bundle, session.OwnOptimizer(bundle));
        Assert.Same(bundle, session.OwnOptimizer(bundle));
        Assert.Same(bundle, session.Optimizer);
        Assert.Equal(
            ["auxiliary/0000", "hidden/0000"],
            session.Optimizer!.Leaves.Select(leaf => leaf.Name));

        var replacement = OptimizerBundle.Wrap(
            new Lion([CreateParameter("replacement")]));
        Assert.Throws<InvalidOperationException>(
            () => session.OwnOptimizer(replacement));
    }

    private static Parameter CreateParameter(string name)
        => new(
            [1f],
            [1],
            name,
            WeightDecayPolicy.Exclude);
}
