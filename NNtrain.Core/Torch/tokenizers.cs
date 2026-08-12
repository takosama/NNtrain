#pragma warning disable CS8981

namespace NNtrain;

/// <summary>PyTorch/Hugging Face-style tokenizer entry points.</summary>
public static class tokenizers
{
    public static BpeTokenizer train_bpe(
        IEnumerable<string> documents,
        int vocab_size,
        int max_training_bytes = int.MaxValue)
        => BpeTokenizer.Train(documents, vocab_size, max_training_bytes);

    public static BpeTokenizer load_bpe(string path)
        => BpeTokenizer.Load(path);
}
