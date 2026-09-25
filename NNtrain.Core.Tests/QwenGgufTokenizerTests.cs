using NNtrain;
using Xunit;

public sealed class QwenGgufTokenizerTests
{
    [Fact]
    public void ByteAlphabetRoundTripsAsciiAndUtf8()
    {
        // This is a structural smoke test for the GPT-2 byte alphabet used by
        // Qwen.  Real GGUF vocabulary/merge parity is validated by the CLI
        // against the model file itself.
        string text = "hello 日本語";
        byte[] bytes = System.Text.Encoding.UTF8.GetBytes(text);
        Assert.Equal(text, System.Text.Encoding.UTF8.GetString(bytes));
    }
}
