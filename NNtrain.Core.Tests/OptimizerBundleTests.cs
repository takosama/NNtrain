using System.Text.Json;
using NNtrain;
using NNtrain.Training.Optimization;
using Xunit;

public sealed class OptimizerBundleTests
{
    [Fact]
    public void WrapPreservesCompositeIdentityStateAndCheckpointLeafBytes()
    {
        Parameter matrix = CreateParameter("matrix", 0.25f);
        Parameter auxiliary = CreateParameter("auxiliary", -0.5f);
        matrix.T.MutableGrad[0] = 0.75f;
        auxiliary.T.MutableGrad[0] = -0.25f;
        var first = new NekoMuon(
            [matrix],
            new NekoMuonOptions
            {
                WeightDecay = 0f,
                NewtonSchulzInterval = 3,
            });
        var second = new AdamW(
            [auxiliary],
            new AdamWOptions
            {
                WeightDecay = 0f,
            });
        var composite = new CompositeOptimizer(first, second);
        composite.step();
        byte[][] originalPayloads = SerializeLeaves(composite);
        string originalState = JsonSerializer.Serialize(
            composite.state_dict());

        OptimizerBundle bundle = OptimizerBundle.Wrap(
            composite,
            ["hidden", "auxiliary"]);

        Assert.Same(composite, bundle.RootOptimizer);
        Assert.Equal(["hidden", "auxiliary"],
            bundle.Groups.Select(group => group.Name));
        Assert.Equal(["hidden/0000", "auxiliary/0000"],
            bundle.Leaves.Select(leaf => leaf.Name));
        Assert.Equal([0, 1], bundle.Leaves.Select(leaf => leaf.Index));
        Assert.Equal(
            new IOptimizer[] { first, second },
            bundle.LeafOptimizers);
        Assert.Equal(
            new IOptimizer[] { first, second },
            OptimizerStateStream.GetLeafOptimizers(bundle));
        Assert.Equal(originalState, JsonSerializer.Serialize(bundle.state_dict()));
        AssertPayloadsEqual(originalPayloads, SerializeLeaves(bundle));
    }

    [Fact]
    public void DefaultWrapNamesAreStableAndPositional()
    {
        var first = new Lion([CreateParameter("first", 1f)]);
        var second = new AdamW([CreateParameter("second", 2f)]);
        var composite = new CompositeOptimizer(first, second);

        OptimizerBundle firstBundle = OptimizerBundle.Wrap(composite);
        OptimizerBundle secondBundle = OptimizerBundle.Wrap(composite);

        Assert.Equal(
            ["group-0000", "group-0001"],
            firstBundle.Groups.Select(group => group.Name));
        Assert.Equal(
            firstBundle.Leaves.Select(leaf => leaf.Name),
            secondBundle.Leaves.Select(leaf => leaf.Name));
        Assert.Equal(
            firstBundle.LeafOptimizers,
            secondBundle.LeafOptimizers);
    }

    [Fact]
    public void ExplicitGroupsFlattenNestedCompositesInDeclaredOrder()
    {
        var first = new AdamW([CreateParameter("first", 1f)]);
        var second = new Lion([CreateParameter("second", 2f)]);
        var third = new NekoMuon([CreateParameter("third", 3f)]);
        var nested = new CompositeOptimizer(first, second);

        var bundle = new OptimizerBundle(
            [
                new OptimizerGroup("primary", nested),
                new OptimizerGroup("secondary", third),
            ]);

        Assert.IsType<CompositeOptimizer>(bundle.RootOptimizer);
        Assert.Equal(
            new IOptimizer[] { first, second, third },
            bundle.LeafOptimizers);
        Assert.Equal(
            ["primary/0000", "primary/0001", "secondary/0000"],
            bundle.Leaves.Select(leaf => leaf.Name));
        Assert.Equal([0, 1, 2],
            bundle.Leaves.Select(leaf => leaf.Index));
        Assert.Equal([0, 1, 0],
            bundle.Leaves.Select(leaf => leaf.IndexWithinGroup));
    }

    [Fact]
    public void BundleRejectsDuplicateNamesAndWrongWrapNameCount()
    {
        var first = new AdamW([CreateParameter("first", 1f)]);
        var second = new Lion([CreateParameter("second", 2f)]);

        Assert.Throws<ArgumentException>(() => new OptimizerBundle(
            [
                new OptimizerGroup("duplicate", first),
                new OptimizerGroup("duplicate", second),
            ]));
        Assert.Throws<ArgumentException>(() => OptimizerBundle.Wrap(
            new CompositeOptimizer(first, second),
            ["only-one-name"]));
    }

    [Fact]
    public void BundleForwardsLifecycleToTheWrappedOptimizer()
    {
        Parameter parameter = CreateParameter("weight", 1f);
        parameter.T.MutableGrad[0] = 1f;
        var optimizer = new AdamW(
            [parameter],
            new AdamWOptions
            {
                LearningRate = 0.1f,
                WeightDecay = 0f,
            });
        OptimizerBundle bundle = OptimizerBundle.Wrap(optimizer);

        bundle.step();
        bundle.zero_grad();

        Assert.InRange(parameter.T.Data[0], 0.89998f, 0.90002f);
        Assert.Equal(0f, parameter.T.Grad[0]);
        Assert.Equal("default", bundle.Groups[0].Name);
    }

    [Fact]
    public void NamedBundleStateRoundTripPreservesLeafBytesAndOrder()
    {
        Parameter sourceHidden = CreateParameter("hidden", 0.25f);
        Parameter sourceAuxiliary = CreateParameter("auxiliary", -0.5f);
        sourceHidden.T.MutableGrad[0] = 0.75f;
        sourceAuxiliary.T.MutableGrad[0] = -0.25f;
        var source = new OptimizerBundle(
        [
            new OptimizerGroup("hidden", new NekoMuon([sourceHidden])),
            new OptimizerGroup("auxiliary", new AdamW([sourceAuxiliary])),
        ]);
        source.step();

        var restored = new OptimizerBundle(
        [
            new OptimizerGroup(
                "hidden",
                new NekoMuon([CreateParameter("hidden", 0.25f)])),
            new OptimizerGroup(
                "auxiliary",
                new AdamW([CreateParameter("auxiliary", -0.5f)])),
        ]);
        restored.load_state_dict(source.state_dict());

        Assert.Equal(
            source.Leaves.Select(leaf => leaf.Name),
            restored.Leaves.Select(leaf => leaf.Name));
        AssertPayloadsEqual(
            SerializeLeaves(source),
            SerializeLeaves(restored));
    }

    private static byte[][] SerializeLeaves(IOptimizer optimizer)
        => OptimizerStateStream.GetLeafOptimizers(optimizer)
            .Select(SerializeLeaf)
            .ToArray();

    private static byte[] SerializeLeaf(IOptimizer optimizer)
    {
        using var stream = new MemoryStream();
        OptimizerStateStream.SaveStateBinary(optimizer, stream);
        return stream.ToArray();
    }

    private static void AssertPayloadsEqual(
        IReadOnlyList<byte[]> expected,
        IReadOnlyList<byte[]> actual)
    {
        Assert.Equal(expected.Count, actual.Count);
        for (int index = 0; index < expected.Count; index++)
            Assert.Equal(expected[index], actual[index]);
    }

    private static Parameter CreateParameter(string name, float value)
        => new(
            [value],
            [1],
            name,
            WeightDecayPolicy.Exclude);
}
