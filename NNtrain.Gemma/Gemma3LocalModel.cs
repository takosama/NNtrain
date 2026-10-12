using System.Diagnostics;
using System.Runtime.InteropServices;
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
    // LLamaSharp selects one native backend per process. ModelParams itself calls
    // native code, so select the backend before constructing it.
    private static readonly object NativeBackendLock = new();
    private static Gemma3Backend? _selectedBackend;
    private static bool _selectedAllOpenClDevices;
    private static bool _nativeBackendConfigurationFailed;

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

        lock (NativeBackendLock)
        {
            ConfigureNativeBackend(backend, allOpenClDevices);
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
    {
        ArgumentNullException.ThrowIfNull(turns);
        if (turns.Count == 0 || turns[^1].Role != "user")
            throw new ArgumentException("Gemma conversation must end with a user turn.", nameof(turns));

        var prompt = new StringBuilder();
        var system = new StringBuilder();
        bool firstUser = true;
        foreach (Gemma3ChatTurn turn in turns)
        {
            ArgumentNullException.ThrowIfNull(turn);
            if (turn.Role == "system")
            {
                if (!firstUser) throw new ArgumentException("System text must precede user turns.", nameof(turns));
                if (system.Length > 0) system.Append("\n\n");
                system.Append(EscapeMarkers(turn.Content));
                continue;
            }
            string role = turn.Role switch
            {
                "user" => "user",
                "assistant" => "model",
                _ => throw new ArgumentException($"Unsupported Gemma chat role '{turn.Role}'.", nameof(turns))
            };
            prompt.Append("<start_of_turn>").Append(role).Append('\n');
            if (firstUser && role == "user" && system.Length > 0)
                prompt.Append(system).Append("\n\n");
            prompt.Append(EscapeMarkers(turn.Content)).Append("<end_of_turn>\n");
            if (role == "user") firstUser = false;
        }
        prompt.Append("<start_of_turn>model\n");
        return prompt.ToString();
    }

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

    private static string EscapeMarkers(string text)
        => (text ?? string.Empty)
            .Replace("<start_of_turn>", "<start_of_\u200bturn>", StringComparison.Ordinal)
            .Replace("<end_of_turn>", "<end_of_\u200bturn>", StringComparison.Ordinal);

    private sealed class CountingSamplingPipeline(DefaultSamplingPipeline inner,
        Stopwatch timer) : ISamplingPipeline
    {
        public int SampledTokens { get; private set; }
        public double? FirstSampleMilliseconds { get; private set; }
        public bool LastSampleWasEndOfGeneration { get; private set; }

        public LLamaToken Sample(SafeLLamaContextHandle context, int index)
        {
            LLamaToken token = inner.Sample(context, index);
            SampledTokens++;
            FirstSampleMilliseconds ??= timer.Elapsed.TotalMilliseconds;
            LastSampleWasEndOfGeneration = token.IsEndOfGeneration(context.Vocab);
            return token;
        }

        public void Apply(SafeLLamaContextHandle context, LLamaTokenDataArray data)
            => inner.Apply(context, data);

        public void Accept(LLamaToken token) => inner.Accept(token);

        public void Reset() => inner.Reset();

        // The outer GenerateAsync scope owns and disposes the native pipeline.
        public void Dispose() { }
    }

    private static void ConfigureNativeBackend(Gemma3Backend backend, bool allOpenClDevices)
    {
        if (_nativeBackendConfigurationFailed)
            throw new InvalidOperationException(
                "The Gemma 3 native backend failed to initialize. Restart NNtrain before retrying.");
        if (_selectedBackend is Gemma3Backend alreadySelected)
        {
            if (alreadySelected != backend)
                throw new InvalidOperationException(
                    "LLamaSharp cannot switch between CPU and OpenCL after its native backend loads. " +
                    "Restart NNtrain to change the Gemma 3 backend.");
            if (backend == Gemma3Backend.OpenCl && _selectedAllOpenClDevices != allOpenClDevices)
                throw new InvalidOperationException(
                    "Changing the OpenCL device set requires restarting NNtrain.");
            return;
        }

        var config = NativeLibraryConfig.LLama;
        if (config.LibraryHasLoaded)
            throw new InvalidOperationException(
                "LLamaSharp native backend was loaded before Gemma 3 selected CPU or OpenCL. " +
                "Restart NNtrain and load Gemma 3 first.");

        if (backend == Gemma3Backend.Cpu)
        {
            config.WithCuda(false).WithVulkan(false).WithAutoFallback(true);
            _selectedBackend = backend;
            return;
        }

        if (!OperatingSystem.IsWindows())
            throw new PlatformNotSupportedException("The bundled Gemma 3 OpenCL backend requires Windows and Intel Arc.");

        string directory = Path.Combine(AppContext.BaseDirectory, "gemma-opencl");
        string library = Path.Combine(directory, "llama.dll");
        string openClPlugin = Path.Combine(directory, "ggml-opencl.dll");
        if (!File.Exists(library) || !File.Exists(openClPlugin))
            throw new FileNotFoundException(
                "The bundled Gemma 3 OpenCL backend is incomplete. " +
                "Build NNtrain with gemma-opencl/llama.dll and gemma-opencl/ggml-opencl.dll.",
                !File.Exists(library) ? library : openClPlugin);

        // Select Intel Arc instead of the older NVIDIA OpenCL platform. The
        // default uses one B580; the opt-in mode lets the OpenCL backend see
        // all Intel GPUs for a measured two-device layer split.
        if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("GGML_OPENCL_PLATFORM")))
            SetOpenClEnvironment("GGML_OPENCL_PLATFORM", "Intel");
        if (allOpenClDevices)
        {
            SetOpenClEnvironment("GGML_OPENCL_DEVICE", null);
            // The measured dual-Arc path uses a 16x32 XMX tile for Q8_0 prefill and reuses
            // programs compiled for the shared OpenCL context. Explicit zero
            // values retain the reference kernels for comparisons.
            if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("GGML_OPENCL_Q8_XMX_F16")))
                SetOpenClEnvironment("GGML_OPENCL_Q8_XMX_F16", "1");
            if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("GGML_OPENCL_Q8_XMX_F16_M16N32")))
                SetOpenClEnvironment("GGML_OPENCL_Q8_XMX_F16_M16N32", "1");
            if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("GGML_OPENCL_PROGRAM_CACHE")))
                SetOpenClEnvironment("GGML_OPENCL_PROGRAM_CACHE", "1");
        }
        else if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("GGML_OPENCL_DEVICE")))
            SetOpenClEnvironment("GGML_OPENCL_DEVICE", "B580");

        // WithLibrary is an explicit full-path override: LLamaSharp ignores its
        // ordinary CPU/Vulkan selection and cannot silently choose those DLLs.
        try
        {
            config.WithLibrary(library);
            if (!config.DryRun(out var nativeLibrary) || nativeLibrary is null)
                throw new InvalidOperationException(
                    $"The bundled Gemma 3 OpenCL backend could not be loaded from '{library}'.");
            if (!NativeApi.llama_supports_gpu_offload())
                throw new InvalidOperationException(
                    "The bundled Gemma 3 backend did not expose GPU offload. " +
                    "Check the Intel OpenCL driver and ggml-opencl.dll.");
            _selectedBackend = backend;
            _selectedAllOpenClDevices = allOpenClDevices;
        }
        catch
        {
            // LLamaSharp's explicit library override cannot be safely undone.
            // Do not let a later CPU request accidentally use this GPU DLL.
            _nativeBackendConfigurationFailed = true;
            throw;
        }
    }

    private static void SetOpenClEnvironment(string name, string? value)
    {
        Environment.SetEnvironmentVariable(name, value, EnvironmentVariableTarget.Process);
        // ggml-opencl reads getenv() from the Windows UCRT. Updating only the
        // Win32 process environment after startup does not update that cache.
        int result = PutEnvironmentVariable(name, value ?? string.Empty);
        if (result != 0)
            throw new InvalidOperationException($"Could not set native OpenCL environment {name}: {result}.");
    }

    [DllImport("ucrtbase.dll", EntryPoint = "_putenv_s", CallingConvention = CallingConvention.Cdecl,
        CharSet = CharSet.Ansi)]
    private static extern int PutEnvironmentVariable(string name, string value);
}
