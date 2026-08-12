#pragma warning disable CS8981

namespace NNtrain;

/// <summary>
/// PyTorch-style entry points for tensor creation, seeding and autograd scopes.
/// </summary>
public static class torch
{
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
        string name = "")
        => new(data, shape, name);

    public static Tensor zeros(params int[] shape) => Tensor.Zeros(shape);

    public static Tensor scalar(float value, string name = "")
        => Tensor.Scalar(value, name);

    public static IDisposable no_grad() => AutogradContext.NoGrad();
}
