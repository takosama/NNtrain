#pragma warning disable CS8981

using System.Text.Json;
using System.Text.Json.Serialization;

namespace NNtrain;

/// <summary>
/// PyTorch-style entry points for tensor creation, seeding and autograd scopes.
/// </summary>
public static class torch
{
    public const TensorDType float32 = TensorDType.Float32;
    public const TensorDType float16 = TensorDType.Float16;
    public const TensorDType half = TensorDType.Float16;

    public static class utils
    {
        public static class data
        {
            public static DataLoader DataLoader(
                IImageClassificationDataset dataset,
                int batch_size = 1,
                bool shuffle = false,
                bool drop_last = false,
                bool training = false,
                Random? generator = null,
                Random? augmentation_generator = null)
                => new(
                    dataset,
                    batch_size,
                    shuffle,
                    drop_last,
                    training,
                    generator,
                    augmentation_generator);
        }
    }

    private static readonly JsonSerializerOptions SerializationOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals,
    };
    private static readonly object SeedLock = new();
    private static int _seed = 1;
    private static int _generatorIndex;

    public static int initial_seed
    {
        get
        {
            lock (SeedLock)
                return _seed;
        }
    }

    public static void manual_seed(int seed)
    {
        lock (SeedLock)
        {
            _seed = seed;
            _generatorIndex = 0;
        }
    }

    public static Random generator(int stream = 0)
    {
        lock (SeedLock)
        {
            int index = stream == 0 ? _generatorIndex++ : stream;
            return new Random(HashCode.Combine(_seed, index));
        }
    }

    public static Tensor tensor(
        float[] data,
        int[] shape,
        string name = "",
        TensorDType dtype = TensorDType.Float32)
        => new(data, shape, name, dtype);

    public static Tensor zeros(params int[] shape) => Tensor.Zeros(shape);

    public static Tensor zeros(
        int[] shape,
        TensorDType dtype)
        => Tensor.Zeros(dtype, shape);

    public static Tensor scalar(
        float value,
        string name = "",
        TensorDType dtype = TensorDType.Float32)
        => Tensor.Scalar(value, name, dtype);

    public static IDisposable no_grad() => AutogradContext.NoGrad();

    public static void save<T>(T value, string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        string fullPath = Path.GetFullPath(path);
        string? directory = Path.GetDirectoryName(fullPath);
        if (!string.IsNullOrEmpty(directory))
            Directory.CreateDirectory(directory);

        string temporaryPath = fullPath + ".tmp";
        File.WriteAllText(
            temporaryPath,
            JsonSerializer.Serialize(value, SerializationOptions));
        File.Move(temporaryPath, fullPath, overwrite: true);
    }

    public static T load<T>(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        return JsonSerializer.Deserialize<T>(
            File.ReadAllText(path),
            SerializationOptions)
            ?? throw new InvalidDataException(
                $"Serialized torch object '{path}' was JSON null.");
    }

    public static void save_safetensors(ModuleState state, string path)
        => SafeTensorFile.Save(state, path);

    public static ModuleState load_safetensors(string path)
        => SafeTensorFile.Load(path);
}
