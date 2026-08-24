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
    public const TensorDType bfloat16 = TensorDType.BFloat16;

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

    public static TorchDevice device(string value) => TorchDevice.Parse(value);

    public static IDisposable use_device(TorchDevice device)
        => TensorExecutionContext.Push(device);

    public static class cuda
    {
        public static bool is_available(int device_index = 0)
            => TensorBackends
                .Get(TensorDevice.Cuda)
                .IsAvailable(device_index);

        public static int device_count() => Tensor.CudaDeviceCount;
    }

    public static Tensor tensor(
        float[] data,
        int[] shape,
        string name = "",
        TensorDType dtype = TensorDType.Float32,
        TorchDevice? device = null)
    {
        var result = new Tensor(data, shape, name, dtype);
        return device is TorchDevice target ? result.to(target) : result;
    }

    public static Tensor zeros(params int[] shape) => Tensor.Zeros(shape);

    public static Tensor zeros(
        int[] shape,
        TensorDType dtype,
        TorchDevice? device = null)
    {
        Tensor result = Tensor.Zeros(dtype, shape);
        return device is TorchDevice target ? result.to(target) : result;
    }

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

        // Serialize straight into the file. Going through a string would
        // materialize the whole checkpoint twice - once as UTF-16 and once as
        // UTF-8 - which for a multi-hundred-megabyte checkpoint is gigabytes of
        // transient allocation.
        string temporaryPath = fullPath + ".tmp";
        using (var stream = new FileStream(
            temporaryPath,
            FileMode.Create,
            FileAccess.Write,
            FileShare.None,
            bufferSize: 1024 * 1024,
            FileOptions.SequentialScan))
        {
            JsonSerializer.Serialize(stream, value, SerializationOptions);
        }
        File.Move(temporaryPath, fullPath, overwrite: true);
    }

    public static T load<T>(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        using var stream = new FileStream(
            Path.GetFullPath(path),
            FileMode.Open,
            FileAccess.Read,
            FileShare.Read,
            bufferSize: 1024 * 1024,
            FileOptions.SequentialScan);
        return JsonSerializer.Deserialize<T>(stream, SerializationOptions)
            ?? throw new InvalidDataException(
                $"Serialized torch object '{path}' was JSON null.");
    }

    public static void save_safetensors(ModuleState state, string path)
        => SafeTensorFile.Save(state, path);

    public static ModuleState load_safetensors(string path)
        => SafeTensorFile.Load(path);
}
