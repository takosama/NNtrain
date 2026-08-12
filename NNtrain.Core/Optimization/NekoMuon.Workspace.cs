namespace NNtrain;

public sealed partial class NekoMuon
{
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
