namespace NNtrain;

public sealed partial class NekoMuon
{
    internal int MaterializedCpuWorkspaceCount
        => _workspaces.Count(workspace => workspace is not null);

    internal long LegacyCudaScratchBytesPerDevice
    {
        get
        {
            long floatCount = 0;
            foreach (Parameter parameter in _parameters)
            {
                GetMatrixShape(
                    parameter,
                    out int originalRows,
                    out int originalColumns);
                int rows = Math.Min(originalRows, originalColumns);
                floatCount = checked(
                    floatCount
                    + 4L * parameter.T.Numel
                    + 2L * rows * rows);
            }
            return checked(floatCount * sizeof(float));
        }
    }

    internal long SharedCudaScratchBytesPerDevice
    {
        get
        {
            int maximumLength = 0;
            int maximumGramLength = 0;
            foreach (Parameter parameter in _parameters)
            {
                GetMatrixShape(
                    parameter,
                    out int originalRows,
                    out int originalColumns);
                int rows = Math.Min(originalRows, originalColumns);
                maximumLength = Math.Max(maximumLength, parameter.T.Numel);
                maximumGramLength = Math.Max(
                    maximumGramLength,
                    checked(rows * rows));
            }
            long floatCount = checked(
                2L * maximumLength + 2L * maximumGramLength);
            return checked(floatCount * sizeof(float));
        }
    }

    internal long ConfiguredCudaScratchBytesPerDevice
        => checked(SharedCudaScratchBytesPerDevice * _cudaBatchCapacity);

    internal int CudaBatchCapacity => _cudaBatchCapacity;

    private static int ResolveCudaBatchCapacity()
    {
        if (Environment.GetEnvironmentVariable(
            "NNTRAIN_DISABLE_BATCHED_NEKOMUON") == "1")
        {
            return 1;
        }
        string? configured = Environment.GetEnvironmentVariable(
            "NNTRAIN_NEKOMUON_BATCH_SIZE");
        return int.TryParse(configured, out int value)
            ? Math.Clamp(value, 1, 32)
            : 8;
    }

    private static NekoMuonWorkspace CreateWorkspace(Parameter parameter)
    {
        int length = parameter.T.Numel;
        GetMatrixShape(parameter, out int originalRows, out int originalColumns);
        int rows = Math.Min(originalRows, originalColumns);
        int gramLength = checked(rows * rows);
        return new NekoMuonWorkspace(
            new float[length],
            new float[length],
            new float[length],
            new float[length],
            new float[gramLength],
            new float[gramLength]);
    }

    private static void GetMatrixShape(
        Parameter parameter,
        out int rows,
        out int columns)
    {
        if (parameter.T.Rank >= 2)
        {
            rows = parameter.T.Shape[0];
            columns = parameter.T.Numel / rows;
            return;
        }

        rows = 1;
        columns = parameter.T.Numel;
    }

    private sealed record NekoMuonWorkspace(
        float[] FastHat,
        float[] SlowHat,
        float[] X,
        float[] Next,
        float[] Gram,
        float[] GramSquared);
}
