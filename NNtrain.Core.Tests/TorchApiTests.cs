using NNtrain;
using Xunit;

public sealed class TorchApiTests
{
    [Fact]
    public void DataLoaderBatchesAndShufflesDataset()
    {
        torch.manual_seed(7);
        var loader = new DataLoader(
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
        Tensor loss = model.ForwardBatch(input)
            .CrossEntropyWithLogits([1]);
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
    public void TokenizerAliasesRoundTripText()
    {
        BpeTokenizer tokenizer = tokenizers.train_bpe(
            ["torch style tokenizer"],
            BpeTokenizer.BaseVocabularySize);

        int[] ids = tokenizer.encode("torch", add_bos: true, add_eos: true);

        Assert.Equal("torch", tokenizer.decode(ids));
        Assert.Equal(BpeTokenizer.BaseVocabularySize, tokenizer.vocab_size);
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
}
