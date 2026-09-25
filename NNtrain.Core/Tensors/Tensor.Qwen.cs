using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

public partial class Tensor
{
    /// <summary>RMS normalization over the final dimension.</summary>
    public Tensor RmsNormLastDim(Tensor weight, float epsilon = 1e-6f)
    {
        ArgumentNullException.ThrowIfNull(weight);
        if (weight.Rank != 1 || weight.Numel != _shape[^1])
            throw new ArgumentException("RMSNorm weight must match the final dimension.", nameof(weight));
        if (!(epsilon > 0f) || !float.IsFinite(epsilon))
            throw new ArgumentOutOfRangeException(nameof(epsilon));

        int width = _shape[^1], rows = Numel / width;
        if (ExecutionDevice == TensorDevice.Arc)
        {
            var lane = ArcLane;
            using var x = ArcUploadValues();
            using var w = weight.ArcUploadValues();
            using var output = lane.Allocate(Numel);
            var inv = lane.Allocate(rows);
            try
            {
                lane.Run("qwen_rmsnorm", rows, 0, x, w, output, inv, rows, width, epsilon);
                Tensor result = ArcDeviceResult(output, _shape, [this, weight]);
                if (result.Node.IsDetached) { inv.Dispose(); return result; }
                result.Node.RegisterResource(inv);
                result.Node.BackwardAction = () =>
                {
                    using var xv = ArcUploadValues();
                    using var wv = weight.ArcUploadValues();
                    ArcBuffer dy = result.ArcGradient();
                    lane.Run("qwen_rmsnorm_dx", rows, 0, xv, wv, dy, inv, ArcGradient(), rows, width);
                    lane.Run("qwen_rmsnorm_dw", width, 0, xv, dy, inv, weight.ArcGradient(), rows, width);
                };
                return result;
            }
            catch { inv.Dispose(); throw; }
        }

        ThrowIfCudaHostFallback(nameof(RmsNormLastDim));
        EnsureHostDataCurrent();
        weight.EnsureHostDataCurrent();
        var values = new float[Numel];
        var invRms = new float[rows];
        for (int r = 0; r < rows; ++r)
        {
            float square = 0f;
            int offset = r * width;
            for (int c = 0; c < width; ++c)
            {
                float value = _data[offset + c];
                square += value * value;
            }
            float inv = 1f / MathF.Sqrt(square / width + epsilon);
            invRms[r] = inv;
            for (int c = 0; c < width; ++c)
                values[offset + c] = _data[offset + c] * inv * weight._data[c];
        }

        var cpu = new Tensor(values, _shape, [this, weight]);
        cpu.Node.BackwardAction = () =>
        {
            for (int r = 0; r < rows; ++r)
            {
                int offset = r * width;
                float inv = invRms[r], dot = 0f;
                for (int c = 0; c < width; ++c)
                    dot += cpu._grad[offset + c] * weight._data[c] * _data[offset + c];
                float correction = inv * inv * inv * dot / width;
                for (int c = 0; c < width; ++c)
                {
                    int i = offset + c;
                    float dy = cpu._grad[i];
                    _grad[i] += dy * weight._data[c] * inv - _data[i] * correction;
                    weight._grad[c] += dy * _data[i] * inv;
                }
            }
        };
        return cpu;
    }

    /// <summary>Computes SiLU(gate) * up with an exact backward.</summary>
    public Tensor SiluMultiply(Tensor up)
    {
        ArgumentNullException.ThrowIfNull(up);
        if (!_shape.AsSpan().SequenceEqual(up._shape))
            throw ShapeMismatch(this, up, "SiLU multiply");

        if (ExecutionDevice == TensorDevice.Arc)
        {
            var lane = ArcLane;
            using var gate = ArcUploadValues();
            using var upValues = up.ArcUploadValues();
            using var output = lane.Allocate(Numel);
            lane.Run("qwen_silu_mul", Numel, 0, gate, upValues, output, Numel);
            Tensor result = ArcDeviceResult(output, _shape, [this, up]);
            result.Node.BackwardAction = () =>
            {
                using var gateValues = ArcUploadValues();
                using var upValuesBackward = up.ArcUploadValues();
                lane.Run("qwen_silu_mul_back", Numel, 0,
                    gateValues, upValuesBackward, result.ArcGradient(),
                    ArcGradient(), up.ArcGradient(), Numel);
            };
            return result;
        }

        ThrowIfCudaHostFallback(nameof(SiluMultiply));
        EnsureHostDataCurrent();
        up.EnsureHostDataCurrent();
        var values = new float[Numel];
        for (int i = 0; i < Numel; ++i)
        {
            float sigmoid = 1f / (1f + MathF.Exp(-_data[i]));
            values[i] = _data[i] * sigmoid * up._data[i];
        }
        var cpu = new Tensor(values, _shape, [this, up]);
        cpu.Node.BackwardAction = () =>
        {
            for (int i = 0; i < Numel; ++i)
            {
                float x = _data[i];
                float sigmoid = 1f / (1f + MathF.Exp(-x));
                float silu = x * sigmoid;
                _grad[i] += cpu._grad[i] * up._data[i]
                    * sigmoid * (1f + x * (1f - sigmoid));
                up._grad[i] += cpu._grad[i] * silu;
            }
        };
        return cpu;
    }

    /// <summary>
    /// Grouped-query causal attention with Qwen/Llama rotary embeddings.
    /// q=[B,S,H*D], k/v=[B,S,KV*D].
    /// </summary>
    public Tensor QwenGroupedQueryAttention(
        Tensor key, Tensor value, int queryHeads, int kvHeads,
        float ropeTheta = 1_000_000f, bool causal = true)
    {
        ArgumentNullException.ThrowIfNull(key);
        ArgumentNullException.ThrowIfNull(value);
        if (Rank != 3 || key.Rank != 3 || value.Rank != 3)
            throw new InvalidOperationException("Qwen GQA requires rank-3 [batch, sequence, width] tensors.");
        if (_shape[0] != key._shape[0] || _shape[1] != key._shape[1]
            || !key._shape.AsSpan().SequenceEqual(value._shape))
            throw new ArgumentException("Qwen GQA batch/sequence/key/value shapes must match.");
        if (queryHeads <= 0 || kvHeads <= 0 || queryHeads % kvHeads != 0)
            throw new ArgumentException("Query heads must be a positive multiple of KV heads.");
        if (_shape[2] % queryHeads != 0)
            throw new ArgumentException("Query width must be divisible by queryHeads.");
        int headWidth = _shape[2] / queryHeads;
        if ((headWidth & 1) != 0 || key._shape[2] != kvHeads * headWidth)
            throw new ArgumentException("Q/K head widths must match and be even.");
        if (!float.IsFinite(ropeTheta) || ropeTheta <= 0f)
            throw new ArgumentOutOfRangeException(nameof(ropeTheta));

        int batch = _shape[0], sequence = _shape[1];
        int groups = checked(batch * queryHeads * sequence);
        if (ExecutionDevice == TensorDevice.Arc)
        {
            // Both GQA softmax kernels keep a complete row in OpenCL local memory.
            // The backward kernel needs one additional element for its reduction.
            // Keep this bound explicit until a tiled implementation is available.
            const int maxLocalMemorySequence = 4096;
            if (sequence > maxLocalMemorySequence)
                throw new NotSupportedException(
                    $"Arc Qwen GQA supports at most {maxLocalMemorySequence} tokens " +
                    "because its softmax kernels use per-row local memory.");

            var lane = ArcLane;
            bool saveProbabilities = AutogradContext.IsRecordingEnabled;
            using var q = ArcUploadValues(true);
            using var k = key.ArcUploadValues(true);
            using var v = value.ArcUploadValues(true);
            using var output = lane.Allocate(Numel);
            // Inference only needs the row-local softmax values. Backward alone
            // retains the quadratic probability matrix.
            var probabilities = lane.Allocate(saveProbabilities
                ? checked(groups * sequence) : 1);
            try
            {
                lane.Run("qwen_gqa", (long)groups * 64, 64, q, k, v, output,
                    probabilities, batch, sequence, queryHeads, kvHeads,
                    headWidth, causal ? 1 : 0, ropeTheta,
                    saveProbabilities ? 1 : 0,
                    new LocalMemory(checked(sequence * sizeof(float))));
                Tensor result = ArcDeviceResult(output, _shape, [this, key, value]);
                if (result.Node.IsDetached) { probabilities.Dispose(); return result; }
                result.Node.RegisterResource(probabilities);
                result.Node.BackwardAction = () =>
                {
                    using var qv = ArcUploadValues(true);
                    using var kv = key.ArcUploadValues(true);
                    using var vv = value.ArcUploadValues(true);
                    using var ds = lane.Allocate(checked(groups * sequence));
                    lane.Run("qwen_gqa_ds", (long)groups * 64, 64, vv,
                        result.ArcGradient(), probabilities, ds, batch, sequence,
                        queryHeads, kvHeads, headWidth, causal ? 1 : 0,
                        new LocalMemory(checked((sequence + 1) * sizeof(float))));
                    int half = headWidth / 2;
                    lane.Run("qwen_gqa_dq", (long)batch * sequence * queryHeads * half,
                        0, kv, ds, ArcGradient(), batch, sequence, queryHeads,
                        kvHeads, headWidth, causal ? 1 : 0, ropeTheta);
                    lane.Run("qwen_gqa_dkv", (long)batch * sequence * kvHeads * half,
                        0, qv, result.ArcGradient(), probabilities, ds,
                        key.ArcGradient(), value.ArcGradient(), batch, sequence,
                        queryHeads, kvHeads, headWidth, causal ? 1 : 0, ropeTheta);
                };
                return result;
            }
            catch { probabilities.Dispose(); throw; }
        }

        throw new NotSupportedException(
            "Qwen grouped-query attention currently has an Intel Arc implementation; " +
            "CUDA support will use the same operation boundary later.");
    }
}
