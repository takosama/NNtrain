using System.Text;
using System.Text.RegularExpressions;
namespace NNtrain.Audio;

/// <summary>Local NeMo FastConformer Japanese CTC inference, with FP16 weights and FP32 arithmetic.</summary>
/// <remarks>Algorithm: NVIDIA NeMo v1.23.0, Apache-2.0; model weights: NVIDIA CC BY 4.0.
/// https://github.com/NVIDIA/NeMo/tree/v1.23.0/nemo/collections/asr</remarks>
public sealed partial class ParakeetCtcModel : ILocalAsrModel
{
    private readonly AsrHalfCheckpoint _checkpoint;
    private readonly string[] _pieces;
    private AsrArcLinear? _arc;
    internal int Hidden, Layers, Heads, ConvChannels, Kernel;
    public long HostWeightBytes => _checkpoint.WeightBytes;
    public long PeakDeviceBufferBytes => _arc?.PeakDeviceBufferBytes ?? 0;
    public long ResidentDeviceBytes => _arc?.ResidentDeviceBytes ?? 0;
    public string ExecutionDevice => _arc is null ? "Parakeet JA / CTC / CPU" : $"Parakeet JA / CTC / Arc: {_arc.DeviceName}; CPU frontend, convolution and attention";
    public void EnableArc(int deviceIndex, CancellationToken ct = default) => _arc = _arc is null ? new(_checkpoint.Weights, deviceIndex, ct) : throw new InvalidOperationException("Arc already enabled.");
    public void Dispose() { _arc?.Dispose(); _arc = null; }
    private ParakeetCtcModel(AsrHalfCheckpoint checkpoint, string[] pieces, int hidden, int layers, int heads, int convChannels, int kernel)
    {
        _checkpoint = checkpoint; _pieces = pieces; Hidden = hidden; Layers = layers; Heads = heads; ConvChannels = convChannels; Kernel = kernel;
        if (pieces.Length != Weight("ctc_decoder.decoder_layers.0.bias").Values.Length - 1 || hidden % heads != 0)
            throw new InvalidDataException("Parakeet vocabulary or head shape mismatch.");
    }
    public static ParakeetCtcModel Load(string directory, CancellationToken ct = default)
    {
        string config = File.ReadAllText(Path.Combine(directory, "model_config.yaml"));
        static string Section(string text, string name)
        {
            Match match = Regex.Match(text, "(?m)^" + Regex.Escape(name) + ":\\r?\\n(?<body>(?:[ \\t].*\\r?\\n|\\r?\\n)*)");
            return match.Success ? match.Groups["body"].Value : throw new InvalidDataException("Missing Parakeet configuration section.");
        }
        static string Value(string text, string key)
        {
            Match match = Regex.Match(text, "(?m)^  " + Regex.Escape(key) + ": ([^\\r\\n]+)");
            return match.Success ? match.Groups[1].Value.Trim() : throw new InvalidDataException("Missing Parakeet parameter: " + key);
        }
        string pre = Section(config, "preprocessor"), encoder = Section(config, "encoder");
        if (Value(pre, "sample_rate") != "16000" || Value(pre, "features") != "80" || Value(pre, "normalize") != "per_feature"
            || Value(pre, "window_size") != "0.025" || Value(pre, "window_stride") != "0.01" || Value(pre, "window") != "hann" || Value(pre, "n_fft") != "512"
            || Value(encoder, "subsampling") != "dw_striding" || Value(encoder, "subsampling_factor") != "8"
            || Value(encoder, "self_attention_model") != "rel_pos" || Value(encoder, "conv_norm_type") != "batch_norm" || Value(encoder, "xscaling") != "true")
            throw new InvalidDataException("Unsupported Parakeet frontend or encoder configuration.");
        int Number(string key) => int.Parse(Value(encoder, key), System.Globalization.CultureInfo.InvariantCulture);
        int hidden = Number("d_model"), layers = Number("n_layers"), heads = Number("n_heads"), channels = Number("subsampling_conv_channels"), kernel = Number("conv_kernel_size");
        if (hidden is < 8 or > 4096 || layers is < 1 or > 128 || heads is < 1 or > 64 || channels is < 1 or > 1024 || kernel is < 1 or > 65 || kernel % 2 == 0)
            throw new InvalidDataException("Unsupported Parakeet dimensions.");
        string[] pieces = File.ReadLines(Path.Combine(directory, "tokenizer.vocab")).Select(line => line.Split('\t')[0]).ToArray();
        var checkpoint = AsrHalfCheckpoint.Load(Path.Combine(directory, "model.safetensors"), 2L * 1024 * 1024 * 1024, ct);
        return new(checkpoint, pieces, hidden, layers, heads, channels, kernel);
    }
    internal AsrHalfCheckpoint.Weight Weight(string name) => _checkpoint.Weights.TryGetValue(name, out var weight) ? weight : throw new InvalidDataException("Missing Parakeet tensor: " + name);
    private AsrHalfCheckpoint.Weight? Bias(string name) => _checkpoint.Weights.GetValueOrDefault(name + ".bias");
    private float[][] Linear(float[][] rows, string name, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        if (_arc is not null) return _arc.Linear(rows, name);
        return rows.Select(row => { ct.ThrowIfCancellationRequested(); return AsrCpuMath.Linear(row, Weight(name + ".weight"), Bias(name)); }).ToArray();
    }
    private float[] Norm(float[] row, string name) => AsrCpuMath.LayerNorm(row, Weight(name + ".weight"), Weight(name + ".bias"));
    public string Transcribe(ReadOnlySpan<float> samples, CancellationToken ct = default)
    {
        float[][] mel = ParakeetMel.Extract(samples, ct);
        if (mel.Length == 0) return "";
        return DecodeCtc(Linear(Encode(mel, ct), "ctc_decoder.decoder_layers.0", ct));
    }
    internal string DecodeCtc(float[][] logits)
    {
        var text = new StringBuilder(); var bytes = new List<byte>(); int previous = -1;
        void Flush() { if (bytes.Count > 0) { text.Append(Encoding.UTF8.GetString(bytes.ToArray())); bytes.Clear(); } }
        foreach (float[] row in logits)
        {
            if (row.Length != _pieces.Length + 1) throw new InvalidDataException("CTC vocabulary mismatch.");
            int best = 0; for (int i = 1; i < row.Length; i++) if (row[i] > row[best]) best = i;
            bool emit = best != previous && best != _pieces.Length; previous = best;
            if (!emit) continue;
            string piece = _pieces[best];
            if (piece is "<unk>" or "<s>" or "</s>") { Flush(); continue; }
            if (piece.Length == 6 && piece.StartsWith("<0x", StringComparison.Ordinal) && piece.EndsWith('>')
                && byte.TryParse(piece.AsSpan(3, 2), System.Globalization.NumberStyles.HexNumber, null, out byte value)) bytes.Add(value);
            else { Flush(); text.Append(piece.Replace('\u2581', ' ')); }
        }
        Flush(); return text.ToString().TrimStart();
    }
    public ILocalAsrStream CreateStream() => new BufferedPreview(this);
    // This published model has global attention, not Nemotron's cache-aware streaming contract.
    // Preview reprocesses the current utterance; previous text may be revised. Bound working memory and duration.
    private sealed class BufferedPreview(ParakeetCtcModel model) : ILocalAsrStream
    {
        private readonly List<float> _samples = []; private int _lastPreview; private string _text = ""; private bool _closed;
        public long CacheBytes => _samples.Count * 4L;
        public string Append(ReadOnlySpan<float> samples, bool final = false, Action<string>? partial = null, CancellationToken ct = default)
        {
            if (_closed) throw new InvalidOperationException("Parakeet utterance is closed.");
            try
            {
                ct.ThrowIfCancellationRequested();
                if (_samples.Count + samples.Length > 16000 * 30) throw new InvalidOperationException("Parakeet CTC preview supports utterances up to 30 seconds. Stop and start a new utterance.");
                foreach (float value in samples) _samples.Add(value);
                if (final || _samples.Count - _lastPreview >= 16000 * 4)
                { _text = model.Transcribe(_samples.ToArray(), ct); _lastPreview = _samples.Count; partial?.Invoke(_text); }
                _closed = final; return _text;
            }
            catch { _closed = true; throw; }
        }
    }
}
