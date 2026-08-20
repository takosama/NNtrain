namespace NNtrain;

public sealed partial class NekoMuon
{
    private static void UpdateMoments(
        float[] gradientBuffer,
        float[] fast,
        float[] slow,
        float[] fastHat,
        float[] slowHat,
        NekoMuonOptions options,
        float fastCorrection,
        float slowCorrection)
    {
        int length = fast.Length;
        if (Tensor.ExecutionDevice == TensorDevice.Cuda)
        {
            CudaOptimizerKernels.NekoMuonMoments(
                gradientBuffer.Length == 0
                    ? new float[length]
                    : gradientBuffer,
                fast,
                slow,
                fastHat,
                slowHat,
                options.BetaFast,
                options.BetaSlow,
                fastCorrection,
                slowCorrection);
            return;
        }
        int index = 0;
        if (Tensor.SimdEnabled
            && Vector256.IsHardwareAccelerated
            && length >= Vector256<float>.Count)
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = length - length % width;
            Vector256<float> betaFast = Vector256.Create(options.BetaFast);
            Vector256<float> betaSlow = Vector256.Create(options.BetaSlow);
            Vector256<float> oneMinusBetaFast =
                Vector256.Create(1f - options.BetaFast);
            Vector256<float> oneMinusBetaSlow =
                Vector256.Create(1f - options.BetaSlow);
            Vector256<float> inverseFastCorrection =
                Vector256.Create(1f / fastCorrection);
            Vector256<float> inverseSlowCorrection =
                Vector256.Create(1f / slowCorrection);

            for (; index < vectorizedLength; index += width)
            {
                Vector256<float> gradient = gradientBuffer.Length == 0
                    ? Vector256<float>.Zero
                    : Vector256.LoadUnsafe(ref gradientBuffer[index]);
                Vector256<float> nextFast =
                    betaFast * Vector256.LoadUnsafe(ref fast[index])
                    + oneMinusBetaFast * gradient;
                Vector256<float> nextSlow =
                    betaSlow * Vector256.LoadUnsafe(ref slow[index])
                    + oneMinusBetaSlow * gradient;
                nextFast.StoreUnsafe(ref fast[index]);
                nextSlow.StoreUnsafe(ref slow[index]);
                (nextFast * inverseFastCorrection)
                    .StoreUnsafe(ref fastHat[index]);
                (nextSlow * inverseSlowCorrection)
                    .StoreUnsafe(ref slowHat[index]);
            }
        }

        for (; index < length; index++)
        {
            float gradient = gradientBuffer.Length == 0
                ? 0f
                : gradientBuffer[index];
            fast[index] = options.BetaFast * fast[index]
                + (1f - options.BetaFast) * gradient;
            slow[index] = options.BetaSlow * slow[index]
                + (1f - options.BetaSlow) * gradient;
            fastHat[index] = fast[index] / fastCorrection;
            slowHat[index] = slow[index] / slowCorrection;
        }
    }

    private static float CalculateConfidenceRaw(
        float[] fastHat,
        float[] slowHat,
        float epsilon)
    {
        double dot = 0d;
        double fastNormSquared = 0d;
        double slowNormSquared = 0d;
        double residualNormSquared = 0d;

        int index = 0;
        if (Tensor.SimdEnabled
            && Vector256.IsHardwareAccelerated
            && fastHat.Length >= Vector256<float>.Count)
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = fastHat.Length - fastHat.Length % width;
            for (; index < vectorizedLength; index += width)
            {
                Vector256<float> fast =
                    Vector256.LoadUnsafe(ref fastHat[index]);
                Vector256<float> slow =
                    Vector256.LoadUnsafe(ref slowHat[index]);
                Vector256<float> residual = fast - slow;
                dot += Vector256.Sum(fast * slow);
                fastNormSquared += Vector256.Sum(fast * fast);
                slowNormSquared += Vector256.Sum(slow * slow);
                residualNormSquared += Vector256.Sum(residual * residual);
            }
        }

        for (; index < fastHat.Length; index++)
        {
            double fast = fastHat[index];
            double slow = slowHat[index];
            double residual = fast - slow;
            dot += fast * slow;
            fastNormSquared += fast * fast;
            slowNormSquared += slow * slow;
            residualNormSquared += residual * residual;
        }

        double alignmentDenominator =
            Math.Sqrt(fastNormSquared)
            * Math.Sqrt(slowNormSquared)
            + epsilon;
        double alignment = Math.Max(0d, dot / alignmentDenominator);
        double persistence = slowNormSquared
            / (slowNormSquared + residualNormSquared + epsilon);
        return (float)Math.Clamp(alignment * persistence, 0d, 1d);
    }
}
