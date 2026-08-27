using System.Collections;

namespace NNtrain.Runtime.Execution;

/// <summary>
/// An immutable, ordered set of device indices. Selecting a set does not
/// select the execution device; that choice belongs to <see cref="ExecutionOptions"/>.
/// </summary>
public sealed class DeviceSet : IReadOnlyList<int>, IEquatable<DeviceSet>
{
    private readonly int[] _indices;

    public DeviceSet(IEnumerable<int> deviceIndices)
    {
        ArgumentNullException.ThrowIfNull(deviceIndices);
        _indices = deviceIndices.ToArray();
        if (_indices.Length == 0)
            throw new ArgumentException(
                "At least one device index is required.",
                nameof(deviceIndices));
        if (_indices.Any(static index => index < 0))
            throw new ArgumentOutOfRangeException(
                nameof(deviceIndices),
                "Device indices must be non-negative.");
        if (_indices.Distinct().Count() != _indices.Length)
            throw new ArgumentException(
                "Device indices must be unique.",
                nameof(deviceIndices));
    }

    public DeviceSet(params int[] deviceIndices)
        : this((IEnumerable<int>)deviceIndices)
    {
    }

    public static DeviceSet Default { get; } = new(0);

    public int Count => _indices.Length;

    public int this[int index] => _indices[index];

    public bool Contains(int deviceIndex)
        => Array.IndexOf(_indices, deviceIndex) >= 0;

    public int[] ToArray() => (int[])_indices.Clone();

    public IEnumerator<int> GetEnumerator()
        => ((IEnumerable<int>)_indices).GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => _indices.GetEnumerator();

    public bool Equals(DeviceSet? other)
        => other is not null && _indices.SequenceEqual(other._indices);

    public override bool Equals(object? obj) => Equals(obj as DeviceSet);

    public override int GetHashCode()
    {
        var hash = new HashCode();
        foreach (int index in _indices)
            hash.Add(index);
        return hash.ToHashCode();
    }

    public override string ToString() => string.Join(",", _indices);
}
