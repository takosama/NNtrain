using NNtrain.Gemma;

namespace NNtrain.Gui;

/// <summary>
/// Owns one resident Qwen3.5 or Gemma 3 model and serializes GPU loading, generation and
/// disposal. The window controls the five-minute idle timeout via UnloadAsync.
/// </summary>
public sealed partial class InferenceSession : IDisposable
{
    private readonly SemaphoreSlim _operation = new(1, 1);
    private readonly InferenceSessionOptions _options;
    private Qwen35QuantizedModel? _model;
    private Gemma3LocalModel? _gemmaModel;
    private Qwen2GgufTokenizer? _tokenizer;
    private string? _loadedModelPath;
    private string? _loadedAdapterPath;
    private string? _loadedMmprojPath;
    private Qwen35VisionEncoder? _vision;
    private readonly Dictionary<string, Qwen35VisionEmbedding> _imageEmbeddings = [];
    private int[]? _loadedDevices;
    private string? _loadedGemmaBackend;
    private int _disposed;

    public InferenceSession(int inferencePrefillChunkTokens = 1024, bool useOptimizedVisionKernels = true,
        bool inferenceBatchMixedAttention = true, bool collectKernelTimings = false,
        int inferenceProjectionRows = 4, bool prepareVisionOnLoad = true,
        bool inferenceXmxPrefill = true, int inferenceBufferPoolMiB = 512,
        int inferenceDeferredReleaseMiB = 256, bool inferenceBatchRecurrent = true,
        bool inferenceXmxPackedPrefill = false, bool useXmxVisionLinear = true,
        bool useXmxVisionAttention = true, bool inferenceXmxFactoredPrefill = true,
        bool useFlashVisionAttention = false, bool inferenceResidentIq2Panels = false,
        bool inferenceSubgroupRecurrentRms = true, bool inferenceXmxGgufBslmPrefill = true,
        bool inferenceSplitOutputHead = true)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(inferencePrefillChunkTokens);
        _options = new InferenceSessionOptions
        {
            InferencePrefillChunkTokens = inferencePrefillChunkTokens,
            UseOptimizedVisionKernels = useOptimizedVisionKernels,
            UseXmxVisionLinear = useXmxVisionLinear,
            UseXmxVisionAttention = useXmxVisionAttention,
            UseFlashVisionAttention = useFlashVisionAttention,
            InferenceBatchMixedAttention = inferenceBatchMixedAttention,
            CollectKernelTimings = collectKernelTimings,
            InferenceProjectionRows = inferenceProjectionRows,
            InferenceXmxPrefill = inferenceXmxPrefill,
            InferenceXmxPackedPrefill = inferenceXmxPackedPrefill,
            InferenceXmxFactoredPrefill = inferenceXmxFactoredPrefill,
            InferenceResidentIq2Panels = inferenceResidentIq2Panels,
            InferenceXmxGgufBslmPrefill = inferenceXmxGgufBslmPrefill,
            InferenceSubgroupRecurrentRms = inferenceSubgroupRecurrentRms,
            InferenceBufferPoolMiB = inferenceBufferPoolMiB,
            InferenceDeferredReleaseMiB = inferenceDeferredReleaseMiB,
            InferenceBatchRecurrent = inferenceBatchRecurrent,
            InferenceSplitOutputHead = inferenceSplitOutputHead,
            PrepareVisionOnLoad = prepareVisionOnLoad,
        };
    }

    public bool IsLoaded => Volatile.Read(ref _model) is not null || Volatile.Read(ref _gemmaModel) is not null;
    public string? LoadedModelPath => Volatile.Read(ref _loadedModelPath);
    public string? LoadedAdapterPath => Volatile.Read(ref _loadedAdapterPath);
    public string? LoadedMmprojPath => Volatile.Read(ref _loadedMmprojPath);
    public IReadOnlyList<int>? LoadedDevices => Volatile.Read(ref _loadedDevices);
    public string? LoadedGemmaBackend => Volatile.Read(ref _loadedGemmaBackend);
    public GenerationStats? LastGenerationStats { get; private set; }
    /// <summary>Decode, resize and encode work for the most recent generation or preparation; cache hits take zero.</summary>
    public double LastImagePreparationMilliseconds { get; private set; }
    /// <summary>Time spent validating and uploading the currently loaded vision tower.</summary>
    public double VisionInitializationMilliseconds { get; private set; }
    public IReadOnlyDictionary<string, double> KernelMilliseconds => _model?.KernelMilliseconds
        ?? new Dictionary<string, double>();

    public async Task UnloadAsync()
    {
        ThrowIfDisposed();
        await _operation.WaitAsync().ConfigureAwait(false);
        try { await UnloadCoreAsync().ConfigureAwait(false); }
        finally { _operation.Release(); }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _operation.Wait();
        try { UnloadCore(); }
        finally
        {
            _operation.Release();
            _operation.Dispose();
        }
    }

    private Task UnloadCoreAsync() => Task.Run(UnloadCore);

    private void UnloadCore()
    {
        Qwen35QuantizedModel? model = Interlocked.Exchange(ref _model, null);
        Gemma3LocalModel? gemma = Interlocked.Exchange(ref _gemmaModel, null);
        Volatile.Write(ref _tokenizer, null);
        Volatile.Write(ref _loadedModelPath, null);
        Volatile.Write(ref _loadedAdapterPath, null);
        Volatile.Write(ref _loadedMmprojPath, null);
        _imageEmbeddings.Clear();
        _vision?.Dispose();
        _vision = null;
        LastImagePreparationMilliseconds = 0;
        VisionInitializationMilliseconds = 0;
        Volatile.Write(ref _loadedDevices, null);
        Volatile.Write(ref _loadedGemmaBackend, null);
        model?.Dispose();
        gemma?.Dispose();
    }

    private void RecoverGenerationFailure(Qwen35QuantizedModel model)
    {
        try { model.Reset(); }
        catch
        {
            // Reset may fail after a GPU error. Release residency while keeping
            // the original operation failure as the exception seen by callers.
            try { UnloadCore(); } catch { }
        }
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

}
