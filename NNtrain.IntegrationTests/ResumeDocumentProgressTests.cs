using NNtrain;
using Xunit;

public sealed class ResumeDocumentProgressTests
{
    [Theory]
    [InlineData(0, 0)]
    [InlineData(1, 1)]
    [InlineData(7, 13)]
    [InlineData(7, 97)]
    [InlineData(7, 100)]
    [InlineData(128, 53)]
    public void ReplayPreservesExactShuffledSuffix(int buffer, int skip)
    {
        string[] source = Enumerable.Range(0, 100).Select(i => $"document {i}").ToArray();
        var expected = WikiLanguageModelCommand.ShuffleDocuments(source, buffer, new Random(41))
            .Skip(skip).ToArray();
        using var output = new StringWriter();
        var actual = WikiLanguageModelCommand.SkipResumeDocuments(
            WikiLanguageModelCommand.ShuffleDocuments(source, buffer, new Random(41)), skip, output).ToArray();
        Assert.Equal(expected, actual);
        if (skip == 0) Assert.Empty(output.ToString());
        else Assert.Contains("resume data cursor restored", output.ToString());
    }

    [Fact]
    public void TruncatedCorpusFailsInsteadOfSilentlyFinishingEpoch()
    {
        using var output = new StringWriter();
        var error = Assert.Throws<InvalidDataException>(() =>
            WikiLanguageModelCommand.SkipResumeDocuments(["a", "b"], 3, output).ToArray());
        Assert.Contains("emitted only 2", error.Message);
        Assert.DoesNotContain("resume data cursor restored", output.ToString());
    }

    [Fact]
    public void ReportsEvenWhenReaderIsBlockedAndDisposesEnumerator()
    {
        using var tick = new ManualResetEventSlim();
        using var output = new SignalWriter(tick);
        bool disposed = false;
        IEnumerable<string> Source()
        {
            try
            {
                Assert.True(tick.Wait(TimeSpan.FromSeconds(5)), "No progress while MoveNext is blocked.");
                yield return "skip";
                yield return "next";
            }
            finally { disposed = true; }
        }
        Assert.Equal(new[] { "next" }, WikiLanguageModelCommand.SkipResumeDocuments(
            Source(), 1, output, TimeSpan.FromMilliseconds(20)).ToArray());
        Assert.True(disposed);
        Assert.Contains("reading corpus metadata / filling shuffle buffer", output.ToString());
    }

    private sealed class SignalWriter(ManualResetEventSlim tick) : StringWriter
    {
        public override void WriteLine(string? value)
        {
            base.WriteLine(value);
            if (value?.Contains("ETA =") == true) tick.Set();
        }
    }
}
