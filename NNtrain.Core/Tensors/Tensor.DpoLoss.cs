namespace NNtrain;

partial class Tensor
{
    internal void ScaleDpoGradient(float scale)
    {
        if (!float.IsFinite(scale) || scale <= 0) throw new ArgumentOutOfRangeException(nameof(scale));
        if (!HasGradientBuffer || scale == 1) return;
        if (ExecutionDevice == TensorDevice.Cuda)
        {
            int device = CudaDeviceIndex;
            var gradient = EnsureCudaGradientBuffer(device);
            CudaTensorNative.Scale(device, gradient.NativePtr, Numel, scale);
            MarkCudaGradientMutated(device);
        }
        else
        {
            var gradient = EnsureGradientBuffer();
            for (int i = 0; i < gradient.Length; i++) gradient[i] *= scale;
        }
    }

    // CE is mean over unmasked completion tokens. DPO uses their SUM of log probabilities.
    // Only scalar losses cross to the host; logits/activations/gradients stay on CUDA.
    internal static Tensor DpoLoss(Tensor[] crossEntropies, int[] counts, float[] referenceCe, float beta)
    {
        int n = crossEntropies.Length;
        if (n == 0 || n % 2 != 0 || counts.Length != n || referenceCe.Length != n ||
            counts.Any(c => c <= 0) || !float.IsFinite(beta) || beta <= 0 ||
            crossEntropies.Any(t => t.Numel != 1)) throw new ArgumentException("Invalid DPO pair scores.");
        var derivatives = new float[n];
        double total = 0;
        for (int i = 0; i < n; i += 2)
        {
            double c = crossEntropies[i].item(), r = crossEntropies[i + 1].item();
            double margin = beta * ((referenceCe[i] - c) * counts[i]
                - (referenceCe[i + 1] - r) * counts[i + 1]);
            if (!double.IsFinite(margin)) throw new InvalidOperationException("Non-finite DPO score.");
            total += Math.Max(-margin, 0) + Math.Log(1 + Math.Exp(-Math.Abs(margin)));
            double inverseSigmoid = margin >= 0 ? Math.Exp(-margin) / (1 + Math.Exp(-margin))
                : 1 / (1 + Math.Exp(margin));
            double scale = beta * inverseSigmoid / (n / 2);
            derivatives[i] = (float)(scale * counts[i]);
            derivatives[i + 1] = (float)(-scale * counts[i + 1]);
        }
        var result = new Tensor([(float)(total / (n / 2))], [1], crossEntropies, dtype: TensorDType.Float32);
        result.Node.BackwardAction = () =>
        {
            float seed = result.Grad[0];
            for (int i = 0; i < n; i++) crossEntropies[i].EnsureGradientBuffer()[0] += derivatives[i] * seed;
        };
        return result;
    }
}
