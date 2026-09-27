using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

internal sealed class Qwen35LoraMatrix : IDisposable
{
    internal readonly ArcExecutionLane Lane;
    internal readonly int Input, Output, Rank;
    internal readonly float Scale;
    internal readonly ArcBuffer A, B;
    private ArcBuffer? _da, _db, _ma, _mb, _va, _vb;
    internal long ParameterCount => (long)Rank * (Input + Output);

    internal Qwen35LoraMatrix(ArcExecutionLane lane, int input, int output, Qwen35LoraOptions options, Random random)
    {
        Lane = lane; Input = input; Output = output; Rank = options.Rank; Scale = options.Alpha / Rank;
        float bound = 1f / MathF.Sqrt(input);
        A = lane.Upload(Enumerable.Range(0, checked(input * Rank))
            .Select(_ => ((float)random.NextDouble() * 2 - 1) * bound).ToArray());
        try { B = Zero(checked(output * Rank)); }
        catch { A.Dispose(); throw; }
    }
    private ArcBuffer Zero(int count)
    {
        ArcBuffer buffer = Lane.Allocate(count);
        try { Lane.Run("q35a_zero", count, 0, buffer, count); return buffer; }
        catch { buffer.Dispose(); throw; }
    }
    private void PrepareTraining()
    {
        if (_da is not null) return;
        if (!Lane.Options.Qwen35TrainingKernels)
            throw new InvalidOperationException("Load the model with LoraTraining=true to train adapters.");
        var buffers = new List<ArcBuffer>();
        try
        {
            foreach (int count in new[] { Input * Rank, Output * Rank, Input * Rank, Output * Rank, Input * Rank, Output * Rank })
                buffers.Add(Zero(count));
            (_da, _db, _ma, _mb, _va, _vb) = (buffers[0], buffers[1], buffers[2], buffers[3], buffers[4], buffers[5]);
        }
        catch { foreach (var buffer in buffers) buffer.Dispose(); throw; }
    }
    internal ArcBuffer Forward(ArcBuffer x, ArcBuffer y, int rows)
    {
        ArcBuffer z = Lane.Allocate(checked(rows * Rank));
        try
        {
            if (Lane.Options.Qwen35CooperativeLora)
                Lane.Run("q35l_lora_a_coop", (long)rows * Rank * 128, 128, x, A, z, rows, Input, Rank);
            else Lane.Run("q35l_lora_a", rows * Rank, 0, x, A, z, rows, Input, Rank);
            Lane.Run("q35l_lora_b", (long)rows * Output, 0, z, B, y, rows, Output, Rank, Scale);
            return z;
        }
        catch { z.Dispose(); throw; }
    }
    internal void Backward(ArcBuffer x, ArcBuffer z, ArcBuffer dy, ArcBuffer? dx, int rows)
    {
        PrepareTraining();
        using ArcBuffer dz = Lane.Allocate(checked(rows * Rank));
        if (Lane.Options.Qwen35CooperativeLora)
            Lane.Run("q35t_lora_dz_coop", (long)rows * Rank * 128, 128, dy, B, dz, rows, Output, Rank, Scale);
        else Lane.Run("q35t_lora_dz", rows * Rank, 0, dy, B, dz, rows, Output, Rank, Scale);
        Lane.Run("q35t_lora_db", (long)Output * Rank, 0, dy, z, _db!, rows, Output, Rank, Scale);
        Lane.Run("q35t_lora_da", (long)Input * Rank, 0, dz, x, _da!, rows, Input, Rank);
        if (dx is not null) Lane.Run("q35t_lora_dx", (long)rows * Input, 0, dz, A, dx, rows, Input, Rank);
    }
    internal void ZeroGrad()
    {
        PrepareTraining();
        Lane.Run("q35a_zero", Input * Rank, 0, _da!, Input * Rank);
        Lane.Run("q35a_zero", Output * Rank, 0, _db!, Output * Rank);
    }
    internal double GradientSquaredNorm()
    {
        using ArcBuffer result = Lane.Allocate(1);
        var value = new float[1]; double sum = 0;
        foreach (ArcBuffer buffer in new[] { _da!, _db! })
        {
            Lane.Run("q35t_norm2", 128, 128, buffer, result, checked((int)(buffer.ByteLength / 4)));
            Lane.Read(result, value); sum += value[0];
        }
        return sum;
    }
    internal void EnqueueGradientSquaredNorm(ArcBuffer results, int offset, int splits)
    {
        Lane.Run("q35t_norm2_split", splits * 128, 128, _da!, results, Input * Rank, offset, splits);
        Lane.Run("q35t_norm2_split", splits * 128, 128, _db!, results, Output * Rank, offset + splits, splits);
    }
    internal void Update(Qwen35LoraOptions options, int step, float clip)
    {
        float c1 = (float)(1 - Math.Pow(.9, step)), c2 = (float)(1 - Math.Pow(.999, step));
        Lane.Run("q35t_adam", Input * Rank, 0, A, _da!, _ma!, _va!, Input * Rank,
            options.LearningRate, options.WeightDecay, clip, c1, c2);
        Lane.Run("q35t_adam", Output * Rank, 0, B, _db!, _mb!, _vb!, Output * Rank,
            options.LearningRate, options.WeightDecay, clip, c1, c2);
    }
    internal float[][] ReadState()
    {
        var buffers = new[] { A, B, _ma, _mb, _va, _vb };
        return buffers.Select((buffer, i) =>
        {
            var values = new float[(i % 2 == 0 ? Input : Output) * Rank];
            if (buffer is not null) Lane.Read(buffer, values);
            return values;
        }).ToArray();
    }
    internal void RestoreState(float[][] state, bool training)
    {
        if (training) PrepareTraining();
        ArcBuffer?[] buffers = [A, B, _ma, _mb, _va, _vb];
        for (int i = 0; i < buffers.Length; i++)
            if (buffers[i] is { } buffer) Lane.Write(buffer, state[i]);
    }
    internal float[][] ReadGradients()
    {
        var a = new float[Input * Rank]; var b = new float[Output * Rank];
        if (_da is not null) { Lane.Read(_da, a); Lane.Read(_db!, b); }
        return [a, b];
    }
    public void Dispose()
    {
        foreach (ArcBuffer? buffer in new[] { A, B, _da, _db, _ma, _mb, _va, _vb }) buffer?.Dispose();
    }
}
