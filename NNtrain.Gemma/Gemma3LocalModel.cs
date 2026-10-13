using System.Diagnostics;
using System.Text;
using LLama;
using LLama.Common;
using LLama.Native;
using LLama.Sampling;

namespace NNtrain.Gemma;

public sealed record Gemma3ChatTurn(string Role, string Content);

public sealed record Gemma3GenerationResult(
    string Text, int CompletionTokens, double? FirstTokenMilliseconds)
{
    public int PromptTokens { get; init; }
    public bool EndedByStop { get; init; }
}

public enum Gemma3Backend
{
    Cpu,
    OpenCl
}

/// <summary>
/// In-process Gemma 3 GGUF inference. Q8_0 weights remain in the GGUF format;
/// the local native backend owns tokenization, Gemma attention and CPU/OpenCL execution.
/// No network service or model conversion is involved in generation.
/// </summary>
public sealed class Gemma3LocalModel : IDisposable
{
    private readonly LLamaWeights _weights;
    private readonly ModelParams _parameters;
    private readonly SemaphoreSlim _operation = new(1, 1);
    private bool _disposed;

    private Gemma3LocalModel(LLamaWeights weights, ModelParams parameters)
        => (_weights, _parameters) = (weights, parameters);

    public static Gemma3LocalModel Load(string path, int contextSize = 4096,
        int gpuLayers = 0, int? threads = null, Gemma3Backend backend = Gemma3Backend.Cpu,
        bool allOpenClDevices = false, IReadOnlyList<float>? tensorSplits = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(contextSize);
        ArgumentOutOfRangeException.ThrowIfNegative(gpuLayers);
        if (threads is <= 0) throw new ArgumentOutOfRangeException(nameof(threads));
        if (!Enum.IsDefined(backend)) throw new ArgumentOutOfRangeException(nameof(backend));
        if (backend == Gemma3Backend.Cpu && gpuLayers != 0)
            throw new ArgumentException("GPU layers require the explicit OpenCL backend.", nameof(gpuLayers));
        if (backend == Gemma3Backend.OpenCl && gpuLayers == 0)
            throw new ArgumentException("OpenCL requires at least one GPU layer.", nameof(gpuLayers));
        if (backend != Gemma3Backend.OpenCl && allOpenClDevices)
            throw new ArgumentException("All OpenCL devices require the OpenCL backend.", nameof(allOpenClDevices));
        if (tensorSplits is not null)
        {
            if (!allOpenClDevices || tensorSplits.Count < 2)
                throw new ArgumentException("Tensor splits require all OpenCL devices and at least two values.", nameof(tensorSplits));
            if (tensorSplits.Any(value => !float.IsFinite(value) || value <= 0))
                throw new ArgumentException("Tensor splits must contain positive finite values.", nameof(tensorSplits));
        }
        path = Path.GetFullPath(path);
        if (!File.Exists(path)) throw new FileNotFoundException("Gemma 3 GGUF not found.", path);
        if (File.Exists(path + ".aria2"))
            throw new IOException("Gemma 3 GGUF download is still in progress.");

        using (var gguf = new GgufReader(path))
        {
            if (!gguf.Metadata.TryGetValue("general.architecture", out object? architecture)
                || architecture is not string name || name != "gemma3")
                throw new InvalidDataException($"Expected gemma3 GGUF, got '{architecture}'.");
            if (!gguf.Tensors.Any(tensor => tensor.Type == 8))
                throw new NotSupportedException("This path requires a Gemma 3 GGUF with Q8_0 weights.");
            if (gguf.Tensors.Any(tensor => tensor.Type is not (0 or 1 or 8 or 30)))
                throw new NotSupportedException(
                    "This path requires Q8_0 weights with only unquantized F32/F16/BF16 tensors.");
            if (gguf.Metadata.ContainsKey("split.count") || gguf.Metadata.ContainsKey("split.no"))
                throw new NotSupportedException("Join split GGUF files before loading Gemma 3.");
        }

        lock (Gemma3NativeBackend.SyncRoot)
        {
            Gemma3NativeBackend.Configure(backend, allOpenClDevices);
            var parameters = new ModelParams(path)
            {
                ContextSize = checked((uint)contextSize),
                GpuLayerCount = gpuLayers,
                SplitMode = gpuLayers > 0 ? GPUSplitMode.Layer : null,
                Threads = threads ?? Math.Max(1, Environment.ProcessorCount / 2),
                BatchThreads = threads ?? Math.Max(1, Environment.ProcessorCount / 2),
                UseMemorymap = true
            };
            if (tensorSplits is not null)
            {
                if (tensorSplits.Count > parameters.TensorSplits.Length)
                    throw new ArgumentException("Too many tensor split values for the native backend.", nameof(tensorSplits));
                double splitTotal = tensorSplits.Sum(value => (double)value);
                for (int i = 0; i < tensorSplits.Count; i++)
                    parameters.TensorSplits[i] = (float)(tensorSplits[i] / splitTotal);
            }
            return new Gemma3LocalModel(LLamaWeights.LoadFromFile(parameters), parameters);
        }
    }

    public static string FormatConversation(IReadOnlyList<Gemma3ChatTurn> turns)
        => Gemma3ConversationFormatter.Format(turns);

    public async Task<Gemma3GenerationResult> GenerateAsync(string prompt,
        int maxNewTokens, Action<string> onText, CancellationToken ct,
        float temperature = 0f, float topP = 1f, int topK = 1)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(prompt);
        ArgumentNullException.ThrowIfNull(onText);
        ArgumentOutOfRangeException.ThrowIfNegative(maxNewTokens);
        if (!float.IsFinite(temperature) || temperature < 0f)
            throw new ArgumentOutOfRangeException(nameof(temperature));
        if (!float.IsFinite(topP) || topP <= 0f || topP > 1f)
            throw new ArgumentOutOfRangeException(nameof(topP));
        if (topK < 1) throw new ArgumentOutOfRangeException(nameof(topK));
        ObjectDisposedException.ThrowIf(_disposed, this);
        await _operation.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (maxNewTokens == 0) return new Gemma3GenerationResult("", 0, null);
            using LLamaContext context = _weights.CreateContext(_parameters);
            int promptTokens = context.Tokenize(prompt, addBos: true, special: true).Length;
            if (promptTokens + maxNewTokens > _parameters.ContextSize)
                throw new ArgumentException(
                    $"Prompt ({promptTokens} tokens) plus output budget ({maxNewTokens}) " +
                    $"exceeds context size ({_parameters.ContextSize}).", nameof(maxNewTokens));
            var executor = new InteractiveExecutor(context);
            using var sampling = new DefaultSamplingPipeline
            {
                Temperature = temperature,
                TopP = topP,
                TopK = topK,
                MinP = 0f,
                RepeatPenalty = 1f
            };
            var timer = Stopwatch.StartNew();
            var countedSampling = new CountingSamplingPipeline(sampling, timer);
            var inference = new InferenceParams
            {
                MaxTokens = maxNewTokens,
                AntiPrompts = ["<end_of_turn>"],
                SamplingPipeline = countedSampling
            };
            var text = new StringBuilder();
            await foreach (string chunk in executor.InferAsync(prompt, inference, ct)
                               .WithCancellation(ct).ConfigureAwait(false))
            {
                ct.ThrowIfCancellationRequested();
                if (chunk.Length == 0) continue;
                text.Append(chunk);
                onText(chunk);
            }
            string output = text.ToString();
            return new Gemma3GenerationResult(output, countedSampling.SampledTokens,
                countedSampling.FirstSampleMilliseconds)
            {
                PromptTokens = promptTokens,
                EndedByStop = countedSampling.LastSampleWasEndOfGeneration
                    || countedSampling.SampledTokens < maxNewTokens
            };
        }
        finally { _operation.Release(); }
    }

    public void Dispose()
    {
        _operation.Wait();
        try
        {
            if (_disposed) return;
            _disposed = true;
            _weights.Dispose();
        }
        finally
        {
            _operation.Release();
        }
    }

}
