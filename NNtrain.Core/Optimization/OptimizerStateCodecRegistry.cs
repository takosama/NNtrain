namespace NNtrain;

/// <summary>
/// A strongly typed adapter between an optimizer and its streamed state
/// representation. The non-generic surface is kept private to the registry so
/// registrations cannot accidentally pair one optimizer type with another.
/// </summary>
internal sealed class OptimizerStateCodec<TOptimizer>
    : IOptimizerStateCodec
    where TOptimizer : class, IOptimizer
{
    private readonly Action<TOptimizer, Stream> _loadJson;
    private readonly Action<TOptimizer, BinaryReader, Stream> _loadBinary;
    private readonly Action<TOptimizer, Stream> _saveJson;
    private readonly Action<TOptimizer, BinaryWriter, Stream> _saveBinary;

    internal OptimizerStateCodec(
        string stateType,
        Action<TOptimizer, Stream> loadJson,
        Action<TOptimizer, BinaryReader, Stream> loadBinary,
        Action<TOptimizer, Stream> saveJson,
        Action<TOptimizer, BinaryWriter, Stream> saveBinary)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(stateType);
        StateType = stateType;
        _loadJson = loadJson
            ?? throw new ArgumentNullException(nameof(loadJson));
        _loadBinary = loadBinary
            ?? throw new ArgumentNullException(nameof(loadBinary));
        _saveJson = saveJson
            ?? throw new ArgumentNullException(nameof(saveJson));
        _saveBinary = saveBinary
            ?? throw new ArgumentNullException(nameof(saveBinary));
    }

    public Type OptimizerType => typeof(TOptimizer);

    public string StateType { get; }

    public void LoadJson(IOptimizer optimizer, Stream stream)
        => _loadJson(RequireOptimizer(optimizer), stream);

    public void LoadBinary(
        IOptimizer optimizer,
        BinaryReader reader,
        Stream stream)
        => _loadBinary(RequireOptimizer(optimizer), reader, stream);

    public void SaveJson(IOptimizer optimizer, Stream stream)
        => _saveJson(RequireOptimizer(optimizer), stream);

    public void SaveBinary(
        IOptimizer optimizer,
        BinaryWriter writer,
        Stream stream)
        => _saveBinary(RequireOptimizer(optimizer), writer, stream);

    private static TOptimizer RequireOptimizer(IOptimizer optimizer)
        => optimizer as TOptimizer
            ?? throw new ArgumentException(
                $"Codec for '{typeof(TOptimizer).Name}' cannot process " +
                $"optimizer '{optimizer.GetType().Name}'.",
                nameof(optimizer));
}

internal interface IOptimizerStateCodec
{
    Type OptimizerType { get; }

    string StateType { get; }

    void LoadJson(IOptimizer optimizer, Stream stream);

    void LoadBinary(
        IOptimizer optimizer,
        BinaryReader reader,
        Stream stream);

    void SaveJson(IOptimizer optimizer, Stream stream);

    void SaveBinary(
        IOptimizer optimizer,
        BinaryWriter writer,
        Stream stream);
}

/// <summary>
/// Exact-type optimizer codec registry. Both optimizer types and serialized
/// type names are unique so checkpoint dispatch remains deterministic.
/// </summary>
internal sealed class OptimizerStateCodecRegistry
{
    private readonly object _sync = new();
    private readonly Dictionary<Type, IOptimizerStateCodec> _byOptimizer = [];
    private readonly Dictionary<string, IOptimizerStateCodec> _byStateType =
        new(StringComparer.Ordinal);

    internal void Register<TOptimizer>(
        OptimizerStateCodec<TOptimizer> codec)
        where TOptimizer : class, IOptimizer
    {
        ArgumentNullException.ThrowIfNull(codec);
        lock (_sync)
        {
            if (_byOptimizer.ContainsKey(codec.OptimizerType))
            {
                throw new InvalidOperationException(
                    $"A streaming-state codec for optimizer " +
                    $"'{codec.OptimizerType.Name}' is already registered.");
            }
            if (_byStateType.ContainsKey(codec.StateType))
            {
                throw new InvalidOperationException(
                    $"A streaming-state codec named '{codec.StateType}' " +
                    "is already registered.");
            }
            _byOptimizer.Add(codec.OptimizerType, codec);
            _byStateType.Add(codec.StateType, codec);
        }
    }

    internal bool TryResolve(
        IOptimizer optimizer,
        out IOptimizerStateCodec? codec)
    {
        ArgumentNullException.ThrowIfNull(optimizer);
        lock (_sync)
            return _byOptimizer.TryGetValue(optimizer.GetType(), out codec);
    }
}
