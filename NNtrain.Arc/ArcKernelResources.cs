namespace NNtrain.Arc;

public sealed record ArcKernelResources(string Name, ulong? MaximumWorkGroupSize,
    ulong? LocalMemoryBytes, ulong? PrivateMemoryBytes, ulong? SpillMemoryBytes);
