using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace NNtrain;

internal static class DpoDataSource
{
    internal static string Fingerprint(string path, string dataset)
    {
        if (dataset == "jsonl") { using var file = File.OpenRead(path); return Convert.ToHexString(SHA256.HashData(file)); }
        string[] files = Directory.GetFiles(path, "*.parquet").Order(StringComparer.OrdinalIgnoreCase).ToArray();
        if (files.Length == 0) throw new FileNotFoundException("No FineWeb .parquet shards found.", path);
        // Avoid reading the entire pretraining corpus before its first batch.
        // This is a shard metadata identity, not a cryptographic content identity.
        string manifest = string.Join('\n', files.Select(p => {
            var file = new FileInfo(p);
            return $"{file.Name}\t{file.Length}\t{file.LastWriteTimeUtc.Ticks}";
        }));
        return "parquet-metadata-v1:" + Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(manifest)));
    }
    internal static IEnumerable<string> Read(string path, string dataset, string textColumn, CancellationToken token)
    {
        if (dataset == "jsonl")
        {
            foreach (string line in File.ReadLines(path))
            {
                token.ThrowIfCancellationRequested();
                if (string.IsNullOrWhiteSpace(line)) continue;
                var sample = JsonSerializer.Deserialize<DpoCommand.TextExample>(line, LoraConfiguration.Json);
                if (string.IsNullOrWhiteSpace(sample?.Text)) throw new InvalidDataException("DPO JSONL requires non-empty text.");
                yield return sample.Text;
            }
            yield break;
        }
        var reader = WikiParquetCorpus.ReadTextsAsync(path, textColumn, cancellationToken: token).GetAsyncEnumerator(token);
        try
        {
            while (reader.MoveNextAsync().AsTask().GetAwaiter().GetResult())
                if (!string.IsNullOrWhiteSpace(reader.Current)) yield return reader.Current;
        }
        finally { reader.DisposeAsync().AsTask().GetAwaiter().GetResult(); }
    }
}
