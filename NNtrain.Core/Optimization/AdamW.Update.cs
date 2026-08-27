namespace NNtrain;

public partial class AdamW
{
    private void UpdateWorkItem(int workItemIndex)
    {
        AdamWOptions options = _stepOptions;
        float updateScale = _stepUpdateScale;
        float scaledEpsilon = _stepScaledEpsilon;
        AdamWWorkItem workItem = _workItems[workItemIndex];
        AdamWParameterRuntime runtime =
            _parameterRuntime[workItem.ParameterIndex];
        float[] data = runtime.Data;
        float[] grad = runtime.Gradient;
        float[] m = runtime.FirstMoment;
        short[]? mBFloat16 = runtime.FirstMomentBFloat16;
        float[] v = runtime.SecondMoment;
        short[]? vBFloat16 = runtime.SecondMomentBFloat16;
        bool applyWeightDecay = runtime.ApplyWeightDecay;
        int end = workItem.Start + workItem.Length;
        int index = workItem.Start;

        if (Tensor.SimdEnabled
            && Vector256.IsHardwareAccelerated
            && (mBFloat16 is null
                && vBFloat16 is null
                || System.Runtime.Intrinsics.X86.Avx2.IsSupported)
            && workItem.Length >= Vector256<float>.Count)
        {
            int vectorWidth = Vector256<float>.Count;
            int vectorizedLength = end - workItem.Length % vectorWidth;
            Vector256<float> beta1 = Vector256.Create(options.Beta1);
            Vector256<float> beta2 = Vector256.Create(options.Beta2);
            Vector256<float> oneMinusBeta1 =
                Vector256.Create(1f - options.Beta1);
            Vector256<float> oneMinusBeta2 =
                Vector256.Create(1f - options.Beta2);
            Vector256<float> updateScaleVector =
                Vector256.Create(updateScale);
            Vector256<float> epsilon =
                Vector256.Create(scaledEpsilon);
            Vector256<float> parameterScale = Vector256.Create(
                applyWeightDecay
                    ? 1f - options.LearningRate * options.WeightDecay
                    : 1f);
            Vector256<float> one = Vector256.Create(1f);
            ref float dataStart = ref System.Runtime.InteropServices
                .MemoryMarshal.GetArrayDataReference(data);
            ref float gradientStart = ref System.Runtime.InteropServices
                .MemoryMarshal.GetArrayDataReference(grad);
            ref float firstMomentStart = ref System.Runtime.InteropServices
                .MemoryMarshal.GetArrayDataReference(m);
            ref float secondMomentStart = ref System.Runtime.InteropServices
                .MemoryMarshal.GetArrayDataReference(v);

            for (; index < vectorizedLength; index += vectorWidth)
            {
                Vector256<float> gradient = grad.Length == 0
                    ? Vector256<float>.Zero
                    : Vector256.LoadUnsafe(
                        ref System.Runtime.CompilerServices.Unsafe.Add(
                            ref gradientStart,
                            index));
                Vector256<float> firstMoment =
                    Vector256.FusedMultiplyAdd(
                        oneMinusBeta1,
                        gradient,
                        beta1 * (mBFloat16 is null
                            ? Vector256.LoadUnsafe(
                                ref System.Runtime.CompilerServices.Unsafe.Add(
                                    ref firstMomentStart,
                                    index))
                            : LoadBFloat16(mBFloat16, index)));
                Vector256<float> secondMoment =
                    Vector256.FusedMultiplyAdd(
                        oneMinusBeta2 * gradient,
                        gradient,
                        beta2 * (vBFloat16 is null
                            ? Vector256.LoadUnsafe(
                                ref System.Runtime.CompilerServices.Unsafe.Add(
                                    ref secondMomentStart,
                                    index))
                            : LoadBFloat16(vBFloat16, index)));
                if (mBFloat16 is null)
                {
                    firstMoment.StoreUnsafe(
                        ref System.Runtime.CompilerServices.Unsafe.Add(
                            ref firstMomentStart,
                            index));
                }
                else
                {
                    StoreBFloat16(firstMoment, mBFloat16, index);
                }
                if (vBFloat16 is null)
                {
                    secondMoment.StoreUnsafe(
                        ref System.Runtime.CompilerServices.Unsafe.Add(
                            ref secondMomentStart,
                            index));
                }
                else
                {
                    StoreBFloat16(secondMoment, vBFloat16, index);
                }

                Vector256<float> parameter =
                    Vector256.LoadUnsafe(
                        ref System.Runtime.CompilerServices.Unsafe.Add(
                            ref dataStart,
                            index))
                    * parameterScale;
                Vector256<float> inverseDenominator =
                    one / (Vector256.Sqrt(secondMoment) + epsilon);
                parameter -= updateScaleVector
                    * firstMoment
                    * inverseDenominator;
                parameter.StoreUnsafe(
                    ref System.Runtime.CompilerServices.Unsafe.Add(
                        ref dataStart,
                        index));
            }
        }

        for (; index < end; index++)
        {
            float g = grad.Length == 0 ? 0f : grad[index];
            float previousFirstMoment = mBFloat16 is null
                ? m[index]
                : BFloat16ToSingle(mBFloat16[index]);
            float firstMoment = options.Beta1 * previousFirstMoment
                + (1f - options.Beta1) * g;
            if (mBFloat16 is null)
                m[index] = firstMoment;
            else
                mBFloat16[index] = SingleToBFloat16(firstMoment);
            float previousSecondMoment = vBFloat16 is null
                ? v[index]
                : BFloat16ToSingle(vBFloat16[index]);
            float secondMoment = options.Beta2 * previousSecondMoment
                + (1f - options.Beta2) * g * g;
            if (vBFloat16 is null)
                v[index] = secondMoment;
            else
                vBFloat16[index] = SingleToBFloat16(secondMoment);

            if (applyWeightDecay)
                data[index] *= 1f - options.LearningRate
                    * options.WeightDecay;

            data[index] -= updateScale * firstMoment
                / (MathF.Sqrt(secondMoment) + scaledEpsilon);
        }
    }

    private static AdamWWorkItem[] CreateWorkItems(
        IReadOnlyList<Parameter> parameters)
    {
        // Split large matrices so one embedding or projection cannot leave
        // the remaining workers idle near the end of an optimizer step.
        const int ChunkElements = 65_536;
        var workItems = new List<AdamWWorkItem>();
        for (int parameterIndex = 0;
            parameterIndex < parameters.Count;
            parameterIndex++)
        {
            int length = parameters[parameterIndex].T.Numel;
            for (int start = 0; start < length; start += ChunkElements)
            {
                workItems.Add(
                    new AdamWWorkItem(
                        parameterIndex,
                        start,
                        Math.Min(ChunkElements, length - start)));
            }
        }
        return workItems.ToArray();
    }
}
