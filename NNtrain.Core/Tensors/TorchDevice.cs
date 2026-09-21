namespace NNtrain;

/// <summary>Identifies a CPU or indexed CUDA device.</summary>
public readonly record struct TorchDevice
{
    public TorchDevice(TensorDevice type, int index = 0)
    {
        if (!Enum.IsDefined(type))
            throw new ArgumentOutOfRangeException(nameof(type));
        if (index < 0)
            throw new ArgumentOutOfRangeException(nameof(index));
        if (type == TensorDevice.Cpu && index != 0)
        {
            throw new ArgumentException(
                "CPU devices do not have an adapter index.",
                nameof(index));
        }
        Type = type;
        Index = index;
    }

    public TensorDevice Type { get; }

    public int Index { get; }

    public bool IsCuda => Type == TensorDevice.Cuda;
    public bool IsArc => Type == TensorDevice.Arc;

    public static TorchDevice Parse(string value)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(value);
        if (string.Equals(value, "arc", StringComparison.OrdinalIgnoreCase))
            return new TorchDevice(TensorDevice.Arc);
        if (value.StartsWith("arc:", StringComparison.OrdinalIgnoreCase)
            && int.TryParse(value[4..], out int arcIndex) && arcIndex >= 0)
            return new TorchDevice(TensorDevice.Arc, arcIndex);
        if (string.Equals(value, "cpu", StringComparison.OrdinalIgnoreCase))
            return new TorchDevice(TensorDevice.Cpu);
        if (string.Equals(value, "cuda", StringComparison.OrdinalIgnoreCase))
            return new TorchDevice(TensorDevice.Cuda);
        const string Prefix = "cuda:";
        if (value.StartsWith(Prefix, StringComparison.OrdinalIgnoreCase)
            && int.TryParse(value[Prefix.Length..], out int index)
            && index >= 0)
        {
            return new TorchDevice(TensorDevice.Cuda, index);
        }
        throw new FormatException(
            $"Invalid device '{value}'. Use 'cpu', 'cuda[:index]', or 'arc[:index]'.");
    }

    public override string ToString()
        => IsCuda ? $"cuda:{Index}" : IsArc ? $"arc:{Index}" : "cpu";
}
