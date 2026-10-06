using System.Runtime.Intrinsics;

namespace NNtrain;

partial class Tensor
{
    private static void SiluMultiplyValues(TensorStorage gate, TensorStorage up, float[] output)
    {
        int i = 0;
        if (output.Length >= 32 && CanUseSimd(output.Length))
        {
            var one = Vector256.Create(1f);
            for (; i <= output.Length - Vector256<float>.Count; i += Vector256<float>.Count)
            {
                var x = LoadVector256(gate, i);
                var sigmoid = one / (one + Vector256.Exp(-x));
                StoreVector256(x * sigmoid * LoadVector256(up, i), output, i);
            }
        }
        for (; i < output.Length; i++)
        {
            float x = gate[i];
            output[i] = x * (1f / (1f + MathF.Exp(-x))) * up[i];
        }
    }

    private static void SiluMultiplyGradients(TensorStorage gate, TensorStorage up,
        float[] gradient, float[] gateGradient, float[] upGradient)
    {
        int i = 0;
        if (gradient.Length >= 32 && CanUseSimd(gradient.Length))
        {
            var one = Vector256.Create(1f);
            for (; i <= gradient.Length - Vector256<float>.Count; i += Vector256<float>.Count)
            {
                var x = LoadVector256(gate, i);
                var sigmoid = one / (one + Vector256.Exp(-x));
                var dy = LoadVector256(gradient, i);
                StoreVector256(LoadVector256(gateGradient, i) + dy * LoadVector256(up, i)
                    * sigmoid * (one + x * (one - sigmoid)), gateGradient, i);
                StoreVector256(LoadVector256(upGradient, i) + dy * (x * sigmoid), upGradient, i);
            }
        }
        for (; i < gradient.Length; i++)
        {
            float x = gate[i];
            float sigmoid = 1f / (1f + MathF.Exp(-x));
            gateGradient[i] += gradient[i] * up[i] * sigmoid * (1f + x * (1f - sigmoid));
            upGradient[i] += gradient[i] * (x * sigmoid);
        }
    }

    // Use the existing storage codec directly: no decoded full-tensor copy.
    // Small rows and SimdEnabled=false preserve the scalar path.
    private static float RmsSquareSum(TensorStorage values, int offset, int length)
    {
        int i = 0;
        float sum = 0f;
        if (CanUseSimd(length))
        {
            var sums = Vector256<float>.Zero;
            for (; i <= length - Vector256<float>.Count; i += Vector256<float>.Count)
            {
                var v = LoadVector256(values, offset + i);
                sums += v * v;
            }
            sum = Vector256.Sum(sums);
        }
        for (; i < length; i++)
        {
            float v = values[offset + i];
            sum += v * v;
        }
        return sum;
    }

    private static void RmsNormalizeValues(TensorStorage values, TensorStorage weights,
        float[] output, int offset, int length, float inverse)
    {
        int i = 0;
        if (CanUseSimd(length))
        {
            var inv = Vector256.Create(inverse);
            for (; i <= length - Vector256<float>.Count; i += Vector256<float>.Count)
                StoreVector256(LoadVector256(values, offset + i) * inv
                    * LoadVector256(weights, i), output, offset + i);
        }
        for (; i < length; i++)
            output[offset + i] = values[offset + i] * inverse * weights[i];
    }
}
