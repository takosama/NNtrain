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
    internal Action<string, double>? TimingObserver { get; set; }
    internal int Hidden, Layers, Heads, ConvChannels, Kernel;
    public long HostWeightBytes => _checkpoint.WeightBytes;
    public long PeakDeviceBufferBytes => _arc?.PeakDeviceBufferBytes ?? 0;
    public long ResidentDeviceBytes => _arc?.ResidentDeviceBytes ?? 0;
    public string ExecutionDevice => _arc is null ? "Parakeet JA / CTC / CPU" : $"Parakeet JA / CTC / Arc: {_arc.DeviceName}; CPU frontend/convolution; Arc global attention";
    public void EnableArc(int deviceIndex, CancellationToken ct = default) => EnableArc(deviceIndex, ct, null);
    internal void EnableArc(int deviceIndex, CancellationToken ct, Action<string, double>? loadingObserver)
        => _arc = _arc is null ? new(_checkpoint.Weights, deviceIndex, ct, loadingObserver) : throw new InvalidOperationException("Arc already enabled.");
    public void Dispose() { _arc?.Dispose(); _arc = null; }
    private ParakeetCtcModel(AsrHalfCheckpoint checkpoint, string[] pieces, int hidden, int layers, int heads, int convChannels, int kernel)
    {
        _checkpoint = checkpoint; _pieces = pieces; Hidden = hidden; Layers = layers; Heads = heads; ConvChannels = convChannels; Kernel = kernel;
        if (pieces.Length != Weight("ctc_decoder.decoder_layers.0.bias").Values.Length - 1 || hidden % heads != 0)
            throw new InvalidDataException("Parakeet vocabulary or head shape mismatch.");
    }
    public static ParakeetCtcModel Load(string directory, CancellationToken ct = default)
        => Load(directory, ct, null);

    internal static ParakeetCtcModel Load(string directory, CancellationToken ct, Action<string, double>? timingObserver)
    {
        ct.ThrowIfCancellationRequested();
        long started = System.Diagnostics.Stopwatch.GetTimestamp();
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
        timingObserver?.Invoke("config", System.Diagnostics.Stopwatch.GetElapsedTime(started).TotalMilliseconds);
        started = System.Diagnostics.Stopwatch.GetTimestamp();
        string[] pieces = File.ReadLines(Path.Combine(directory, "tokenizer.vocab")).Select(line => line.Split('\t')[0]).ToArray();
        timingObserver?.Invoke("tokenizer", System.Diagnostics.Stopwatch.GetElapsedTime(started).TotalMilliseconds);
        var checkpoint = AsrHalfCheckpoint.Load(Path.Combine(directory, "model.safetensors"), 2L * 1024 * 1024 * 1024, ct, timingObserver);
        started = System.Diagnostics.Stopwatch.GetTimestamp();
        var model = new ParakeetCtcModel(checkpoint, pieces, hidden, layers, heads, channels, kernel);
        ct.ThrowIfCancellationRequested();
        timingObserver?.Invoke("initialization", System.Diagnostics.Stopwatch.GetElapsedTime(started).TotalMilliseconds);
        return model;
    }
    internal AsrHalfCheckpoint.Weight Weight(string name) => _checkpoint.Weights.TryGetValue(name, out var weight) ? weight : throw new InvalidDataException("Missing Parakeet tensor: " + name);
    private AsrHalfCheckpoint.Weight? Bias(string name) => _checkpoint.Weights.GetValueOrDefault(name + ".bias");
    private float[][] Linear(float[][] rows, string name, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        long start = System.Diagnostics.Stopwatch.GetTimestamp();
        float[][] result;
        if (_arc is not null) { _arc.TimingObserver = TimingObserver; result = _arc.Linear(rows, name); }
        else result = rows.Select(row => { ct.ThrowIfCancellationRequested(); return AsrCpuMath.Linear(row, Weight(name + ".weight"), Bias(name)); }).ToArray();
        TimingObserver?.Invoke("linear:" + name, System.Diagnostics.Stopwatch.GetElapsedTime(start).TotalMilliseconds);
        return result;
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
    private sealed class BufferedPreview(ParakeetCtcModel model) : ILocalAsrStream, ILocalAsrPreviewControl, IDisposable
    {
        private readonly List<float> _samples = []; private int _lastPreview;
        private int _previewInterval = 16000; private string _text = ""; private bool _closed;
        private readonly CancellationTokenSource _previewStop = new();
        public void Dispose() { _closed = true; _previewStop.Dispose(); _samples.Clear(); }
        public void RequestFinalization()
        {
            try { _previewStop.Cancel(); } catch (ObjectDisposedException) { }
        }
        public long CacheBytes => _samples.Count * 4L;
        public string Append(ReadOnlySpan<float> samples, bool final = false, Action<string>? partial = null, CancellationToken ct = default)
        {
            if (_closed) throw new InvalidOperationException("Parakeet utterance is closed.");
            try
            {
                ct.ThrowIfCancellationRequested();
                if (_samples.Count + samples.Length > 16000 * 30) throw new InvalidOperationException("Parakeet CTC preview supports utterances up to 30 seconds. Stop and start a new utterance.");
                foreach (float value in samples) _samples.Add(value);
                if ((final && _samples.Count != _lastPreview) || (!final && !_previewStop.IsCancellationRequested && _samples.Count - _lastPreview >= _previewInterval))
                {
                    long started = System.Diagnostics.Stopwatch.GetTimestamp();
                    using var preview = CancellationTokenSource.CreateLinkedTokenSource(ct, final ? CancellationToken.None : _previewStop.Token);
                    try { _text = model.Transcribe(_samples.ToArray(), preview.Token); _lastPreview = _samples.Count; }
                    catch (OperationCanceledException) when (!final && _previewStop.IsCancellationRequested && !ct.IsCancellationRequested)
                    {
                        model.TimingObserver?.Invoke($"stream.previewCancelled:{_samples.Count}", System.Diagnostics.Stopwatch.GetElapsedTime(started).TotalMilliseconds);
                        return _text;
                    }
                    // Recompute the complete prefix; global attention cannot reuse
                    // previous encoder frames. Avoid queuing obsolete short previews.
                    double seconds = System.Diagnostics.Stopwatch.GetElapsedTime(started).TotalSeconds;
                    model.TimingObserver?.Invoke($"stream.{(final ? "final" : "preview")}:{_samples.Count}", seconds * 1000);
                    _previewInterval = (int)(16000 * Math.Clamp(Math.Max(2, seconds * 1.5), 2, 8));
                    partial?.Invoke(_text);
                }
                else if (final) model.TimingObserver?.Invoke($"stream.finalReused:{_samples.Count}", 0);
                _closed = final; if (final) _previewStop.Dispose(); return _text;
            }
            catch { _closed = true; _previewStop.Dispose(); throw; }
        }
    }
}
