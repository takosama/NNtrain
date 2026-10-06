namespace NNtrain.Audio;

/// <summary>Small CPU reference primitives. FP16 parameter storage, FP32 accumulation/activations.</summary>
internal static class AsrCpuMath
{
    internal static float[] Linear(float[] input, AsrHalfCheckpoint.Weight weight, AsrHalfCheckpoint.Weight? bias = null)
    {
        if (weight.Shape.Length < 2 || weight.Shape.Skip(2).Any(x => x != 1)) throw new InvalidDataException("Expected matrix or pointwise convolution.");
        int outputs = weight.Shape[0], inputs = weight.Shape[1];
        if (input.Length != inputs || bias is not null && bias.Values.Length != outputs) throw new InvalidDataException("Linear shape mismatch.");
        var result = new float[outputs];
        for (int row = 0; row < outputs; row++)
        {
            float sum = bias is null ? 0 : (float)bias.Values[row];
            int start = row * inputs;
            for (int col = 0; col < inputs; col++) sum += input[col] * (float)weight.Values[start + col];
            result[row] = sum;
        }
        return result;
    }

    internal static float[] LayerNorm(float[] input, AsrHalfCheckpoint.Weight weight, AsrHalfCheckpoint.Weight bias)
    {
        if (input.Length != weight.Values.Length || input.Length != bias.Values.Length) throw new InvalidDataException("LayerNorm shape mismatch.");
        float mean = 0, variance = 0;
        foreach (float value in input) mean += value;
        mean /= input.Length;
        foreach (float value in input) variance += (value - mean) * (value - mean);
        float scale = 1 / MathF.Sqrt(variance / input.Length + 1e-5f);
        var result = new float[input.Length];
        for (int i = 0; i < input.Length; i++) result[i] = (input[i] - mean) * scale * (float)weight.Values[i] + (float)bias.Values[i];
        return result;
    }

    internal static float Sigmoid(float value)
    {
        float exponential = MathF.Exp(-MathF.Abs(value));
        return value >= 0 ? 1 / (1 + exponential) : exponential / (1 + exponential);
    }
    internal static void Silu(float[] values) { for (int i = 0; i < values.Length; i++) values[i] *= Sigmoid(values[i]); }
    internal static void Relu(float[] values) { for (int i = 0; i < values.Length; i++) values[i] = Math.Max(0, values[i]); }
    internal static void Add(float[] target, float[] source, float scale = 1)
    {
        if (target.Length != source.Length) throw new InvalidDataException("Residual shape mismatch.");
        for (int i = 0; i < target.Length; i++) target[i] += source[i] * scale;
    }

    /// <summary>PyTorch LSTM gate order i,f,g,o; updates h,c for a single token.</summary>
    internal static float[] Lstm(float[] input, float[] hidden, float[] cell,
        AsrHalfCheckpoint.Weight inputWeight, AsrHalfCheckpoint.Weight hiddenWeight,
        AsrHalfCheckpoint.Weight inputBias, AsrHalfCheckpoint.Weight hiddenBias)
    {
        int size = hidden.Length;
        if (cell.Length != size || inputWeight.Shape[0] != 4 * size || hiddenWeight.Shape[0] != 4 * size)
            throw new InvalidDataException("LSTM shape mismatch.");
        float[] gates = Linear(input, inputWeight, inputBias);
        Add(gates, Linear(hidden, hiddenWeight, hiddenBias));
        for (int i = 0; i < size; i++)
        {
            cell[i] = Sigmoid(gates[size + i]) * cell[i] + Sigmoid(gates[i]) * MathF.Tanh(gates[2 * size + i]);
            hidden[i] = Sigmoid(gates[3 * size + i]) * MathF.Tanh(cell[i]);
        }
        return (float[])hidden.Clone();
    }
}
