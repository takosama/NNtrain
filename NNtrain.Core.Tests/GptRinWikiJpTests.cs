using NNtrain;
using Xunit;

public sealed class GptRinWikiJpTests
{
    [Fact]
    public void FusedTokenAndPositionEmbeddingMatchesComposedOperations()
    {
        const int width = 16;
        const int batchSize = 2;
        const int sequenceLength = 3;
        int[] tokens = [1, 2, 1, 1, 4, 2];
        int[] positions = [0, 1, 2, 0, 1, 2];
        float[] tokenValues = Enumerable.Range(0, 6 * width)
            .Select(index => (index % 19 - 9) * 0.02f)
            .ToArray();
        float[] positionValues = Enumerable.Range(0, 3 * width)
            .Select(index => (index % 11 - 5) * 0.03f)
            .ToArray();
        float[] seed = Enumerable.Range(0, tokens.Length * width)
            .Select(index => (index % 13 - 6) * 0.04f)
            .ToArray();
        var fusedTokens = new Tensor(tokenValues, [6, width]);
        var fusedPositions = new Tensor(positionValues, [3, width]);
        var referenceTokens = new Tensor(tokenValues, [6, width]);
        var referencePositions = new Tensor(positionValues, [3, width]);

        Tensor fused = fusedTokens.EmbeddingLookupWithPositions(
            fusedPositions,
            tokens,
            batchSize,
            sequenceLength);
        Tensor reference = referenceTokens.EmbeddingLookup(
                tokens,
                batchSize,
                sequenceLength)
            + referencePositions.EmbeddingLookup(
                positions,
                batchSize,
                sequenceLength);
        fused.Backward(seed);
        reference.Backward(seed);

        Assert.Equal<int>([batchSize, sequenceLength, width], fused.Shape);
        TensorCharacterizationTests.AssertClose(reference.Data, fused.Data);
        TensorCharacterizationTests.AssertClose(
            referenceTokens.Grad,
            fusedTokens.Grad);
        TensorCharacterizationTests.AssertClose(
            referencePositions.Grad,
            fusedPositions.Grad);
    }

    [Fact]
    public void SeparatesNekoMuonMatrixWeightsFromAuxiliaryParameters()
    {
        var model = new GptRinWikiJp(
            vocabularySize: BpeTokenizer.BaseVocabularySize,
            contextLength: 4,
            dModel: 8,
            numHeads: 2,
            dHidden: 16,
            numLayers: 2,
            rng: new Random(5));

        Parameter[] all = model.Parameters().ToArray();
        Parameter[] hidden = model.HiddenWeightParameters.ToArray();
        Parameter[] auxiliary = model.AuxiliaryParameters.ToArray();

        Assert.NotEmpty(hidden);
        Assert.NotEmpty(auxiliary);
        Assert.All(hidden, parameter => Assert.True(parameter.T.Rank >= 2));
        Assert.Empty(hidden.Intersect(auxiliary));
        Assert.Equal(all.Length, hidden.Length + auxiliary.Length);
        Assert.All(
            all,
            parameter => Assert.Contains(parameter, hidden.Concat(auxiliary)));
    }

    [Fact]
    public void EmbeddingLookupAccumulatesRepeatedRowGradients()
    {
        var table = new Tensor(
            [1f, 2f, 3f, 4f, 5f, 6f],
            [3, 2]);

        Tensor output = table.EmbeddingLookup([1, 0, 1], 3);
        output.Sum().Backward();

        Assert.Equal<int>([3, 2], output.Shape);
        Assert.Equal<float>([1f, 1f, 2f, 2f, 0f, 0f], table.Grad);
    }

    [Fact]
    public void ForwardAndBackwardProduceFiniteLanguageModelValues()
    {
        var model = new GptRinWikiJp(
            vocabularySize: BpeTokenizer.BaseVocabularySize,
            contextLength: 4,
            dModel: 8,
            numHeads: 2,
            dHidden: 16,
            numLayers: 1,
            rng: new Random(7));
        int[] input = [1, 4, 5, 6, 1, 7, 8, 9];
        int[] targets = [4, 5, 6, 2, 7, 8, 9, 2];

        Tensor logits = model.Forward(input, batchSize: 2, sequenceLength: 4);
        Tensor loss = logits.CrossEntropyWithLogits(targets);
        loss.Backward();

        Assert.Equal<int>([8, BpeTokenizer.BaseVocabularySize], logits.Shape);
        Assert.True(float.IsFinite(loss.Data[0]));
        Assert.All(logits.Data, value => Assert.True(float.IsFinite(value)));
        Assert.All(
            model.Parameters().SelectMany(parameter => parameter.T.Grad),
            value => Assert.True(float.IsFinite(value)));
    }

    [Fact]
    public void GreedyGenerationReturnsValidTokensAndRestoresTrainingMode()
    {
        var model = new GptRinWikiJp(
            vocabularySize: BpeTokenizer.BaseVocabularySize,
            contextLength: 3,
            dModel: 4,
            numHeads: 1,
            dHidden: 8,
            numLayers: 1,
            rng: new Random(11));

        int[] generated = model.GenerateTokenIds(
            [BpeTokenizer.BosTokenId],
            maxNewTokens: 4,
            temperature: 0f,
            stopTokenId: null,
            random: new Random(13));

        Assert.True(model.IsTraining);
        Assert.Equal(5, generated.Length);
        Assert.All(
            generated,
            token => Assert.InRange(
                token,
                0,
                BpeTokenizer.BaseVocabularySize - 1));
    }
}
