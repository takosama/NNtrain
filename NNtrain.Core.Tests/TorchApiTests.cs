using NNtrain;
using Xunit;

public sealed class TorchApiTests
{
    [Fact]
    public void DataLoaderBatchesAndShufflesDataset()
    {
        torch.manual_seed(7);
        DataLoader loader = torch.utils.data.DataLoader(
            new FakeDataset(),
            batch_size: 2,
            shuffle: true,
            training: true,
            generator: torch.generator());

        DataBatch[] batches = loader.ToArray();

        Assert.Equal(2, loader.Count);
        Assert.Equal([2, 1, 2], batches[0].input.shape);
        Assert.Equal(2, batches[0].target.Length);
        Assert.Single(batches[1].target);
    }

    [Fact]
    public void TorchModuleOptimizerAndTensorAliasesDelegateToExistingApi()
    {
        torch.manual_seed(11);
        TransformerClassifier model = nn.transformer_classifier(
            seq_len: 1,
            d_model: 2,
            num_heads: 1,
            dim_feedforward: 2,
            num_layers: 1,
            num_classes: 2,
            generator: torch.generator());
        IOptimizer optimizer = optim.AdamW(
            model.parameters(),
            lr: 1e-3f,
            weight_decay: 0f);
        Tensor input = torch.tensor([0.25f, -0.5f], [1, 1, 2]);

        model.train();
        optimizer.zero_grad();
        Tensor loss = nn.functional.cross_entropy(model.forward(input), [1]);
        loss.backward();
        optimizer.step();
        model.eval();

        Assert.False(model.IsTraining);
        Assert.Equal(1, loss.numel());
        Assert.NotEmpty(model.state_dict().Parameters);
    }

    [Fact]
    public void WarmupCosineSchedulerUpdatesEveryOptimizerGroup()
    {
        var first = new Parameter(
            [1f], [1], "first", WeightDecayPolicy.Apply);
        var second = new Parameter(
            [1f], [1], "second", WeightDecayPolicy.Apply);
        IOptimizer primary = optim.NekoMuon([first], lr: 0.01f);
        IOptimizer auxiliary = optim.AdamW([second], lr: 0.001f);
        IOptimizer optimizer = optim.Composite(primary, auxiliary);
        ILRScheduler scheduler =
            lr_scheduler.LinearWarmupCosineAnnealingLR(
                optimizer,
                total_epochs: 10,
                warmup_epochs: 2,
                min_lr_ratio: 0.01f);

        IReadOnlyList<float> rates = scheduler.step();

        Assert.Equal(0.005f, rates[0], 7);
        Assert.Equal(0.0005f, rates[1], 7);
        Assert.Equal(rates, scheduler.get_last_lr());
    }

    [Fact]
    public void SchedulerStateDictResumesAtTheNextEpoch()
    {
        var parameter = new Parameter(
            [1f], [1], "weight", WeightDecayPolicy.Apply);
        IOptimizer optimizer = optim.AdamW([parameter], lr: 0.01f);
        ILRScheduler scheduler =
            lr_scheduler.LinearWarmupCosineAnnealingLR(
                optimizer,
                total_epochs: 4,
                warmup_epochs: 1);
        scheduler.step();
        scheduler.step();
        LRSchedulerStateDictionary state = scheduler.state_dict();

        IOptimizer restoredOptimizer = optim.AdamW(
            [parameter],
            lr: 0.01f);
        ILRScheduler restored =
            lr_scheduler.LinearWarmupCosineAnnealingLR(
                restoredOptimizer,
                total_epochs: 4,
                warmup_epochs: 1);
        restored.load_state_dict(state);

        Assert.Equal(2, restored.LastEpoch);
        Assert.Equal(scheduler.step(), restored.step());
    }

    [Fact]
    public void ProgressSchedulerStateDictRoundTripsProgress()
    {
        var parameter = new Parameter(
            [1f], [1], "weight", WeightDecayPolicy.Apply);
        IOptimizer optimizer = optim.AdamW([parameter], lr: 0.01f);
        WarmupCosineProgressLRScheduler scheduler =
            lr_scheduler.WarmupCosineProgressLR(
                optimizer,
                warmup_percent: 20f);
        scheduler.step(0.45d);

        LRSchedulerStateDictionary state = scheduler.state_dict();

        var restoredParameter = new Parameter(
            [1f], [1], "weight", WeightDecayPolicy.Apply);
        WarmupCosineProgressLRScheduler restored =
            lr_scheduler.WarmupCosineProgressLR(
                optim.AdamW([restoredParameter], lr: 0.01f),
                warmup_percent: 20f);
        restored.load_state_dict(state);

        Assert.Equal(0.45d, restored.state_dict().LastProgress);
    }

    [Fact]
    public void TokenizerAliasesRoundTripText()
    {
        BpeTokenizer tokenizer = tokenizers.train_bpe(
            ["torch style tokenizer"],
            BpeTokenizer.BaseVocabularySize);

        int[] ids = tokenizer.encode("torch", add_bos: true, add_eos: true);

        Assert.Equal("torch", tokenizer.decode(ids));
        Assert.Equal(BpeTokenizer.BaseVocabularySize, tokenizer.vocab_size);
    }

    [Fact]
    public void FunctionalLossItemAndTorchSerializationMatchCoreBehavior()
    {
        Tensor logits = torch.tensor([2f, -1f], [1, 2]);
        Tensor loss = nn.functional.cross_entropy(logits, [0]);
        string path = Path.Combine(
            Path.GetTempPath(),
            $"nntrain-torch-{Guid.NewGuid():N}.json");
        try
        {
            var state = new SerializableState(3, loss.item());
            torch.save(state, path);

            SerializableState restored = torch.load<SerializableState>(path);

            Assert.Equal(state, restored);
            Assert.True(float.IsFinite(loss.item()));
        }
        finally
        {
            if (File.Exists(path))
                File.Delete(path);
        }
    }

    [Fact]
    public void TrainingCheckpointRoundTripsAllResumeState()
    {
        TransformerClassifier model = nn.transformer_classifier(
            seq_len: 1,
            d_model: 2,
            num_heads: 1,
            dim_feedforward: 2,
            num_layers: 1,
            num_classes: 2,
            generator: new Random(7));
        IOptimizer optimizer = optim.AdamW(
            model.parameters(),
            lr: 0.001f);
        ILRScheduler scheduler =
            lr_scheduler.CosineAnnealingLR(optimizer, T_max: 3);
        scheduler.step();
        var checkpoint = new TrainingCheckpoint(
            1,
            model.state_dict(),
            optimizer.state_dict(),
            scheduler.state_dict());
        string path = Path.Combine(
            Path.GetTempPath(),
            $"nntrain-checkpoint-{Guid.NewGuid():N}.json");
        try
        {
            torch.save(checkpoint, path);

            TrainingCheckpoint restored =
                torch.load<TrainingCheckpoint>(path);

            Assert.Equal(1, restored.Epoch);
            Assert.Equal(
                checkpoint.Model.Parameters.Length,
                restored.Model.Parameters.Length);
            Assert.Equal(
                checkpoint.Optimizer.StateJson,
                restored.Optimizer.StateJson);
            Assert.Equal(1, restored.Scheduler.LastEpoch);
        }
        finally
        {
            if (File.Exists(path))
                File.Delete(path);
        }
    }

    [Fact]
    public void OptimizerStateDictRoundTripsCompositeState()
    {
        var first = new Parameter(
            [1f], [1], "first", WeightDecayPolicy.Apply);
        var second = new Parameter(
            [2f], [1], "second", WeightDecayPolicy.Apply);
        IOptimizer optimizer = optim.Composite(
            optim.NekoMuon([first], lr: 0.01f),
            optim.AdamW([second], lr: 0.001f));
        first.T.backward();
        second.T.backward();
        optimizer.step();
        OptimizerStateDictionary state = optimizer.state_dict();

        IOptimizer restored = optim.Composite(
            optim.NekoMuon([first], lr: 0.01f),
            optim.AdamW([second], lr: 0.001f));
        restored.load_state_dict(state);

        OptimizerStateDictionary roundTrip = restored.state_dict();
        Assert.Equal(state.OptimizerType, roundTrip.OptimizerType);
        Assert.Equal(2, roundTrip.Children.Length);
        Assert.Equal(state.Children[0].StateJson, roundTrip.Children[0].StateJson);
        Assert.Equal(state.Children[1].StateJson, roundTrip.Children[1].StateJson);
    }

    private sealed class FakeDataset : IImageClassificationDataset
    {
        public int Count => 3;
        public int Rows => 1;
        public int Columns => 2;
        public int ImageSize => 2;
        public int ClassCount => 2;

        public int ReadSample(int index, Span<float> destination)
        {
            destination[0] = index;
            destination[1] = -index;
            return index % ClassCount;
        }

        public int ReadTrainingSample(
            int index,
            Span<float> destination,
            Random random)
        {
            int target = ReadSample(index, destination);
            destination[0] += 1f;
            return target;
        }
    }

    public sealed record SerializableState(int Epoch, float Loss);
}
