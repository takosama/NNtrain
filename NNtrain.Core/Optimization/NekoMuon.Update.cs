namespace NNtrain;

public sealed partial class NekoMuon
{
    private static void ApplyUpdate(
        Parameter parameter,
        float[] update,
        float finalScale,
        NekoMuonOptions options)
    {
        using Tensor.DataMutation mutation = parameter.BeginUpdate();
        Span<float> data = mutation.Values;
        bool applyWeightDecay =
            parameter.WeightDecay == WeightDecayPolicy.Apply
            || (options.Decay1D && parameter.T.Rank == 1);
        int index = 0;

        if (Tensor.SimdEnabled
            && Vector256.IsHardwareAccelerated
            && data.Length >= Vector256<float>.Count)
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = data.Length - data.Length % width;
            Vector256<float> learningRate =
                Vector256.Create(options.LearningRate);
            Vector256<float> updateScale =
                Vector256.Create(options.LearningRate * finalScale);
            Vector256<float> weightDecay =
                Vector256.Create(options.WeightDecay);
            for (; index < vectorizedLength; index += width)
            {
                Vector256<float> parameterValues =
                    Vector256.LoadUnsafe(ref data[index]);
                if (applyWeightDecay)
                {
                    parameterValues -= learningRate
                        * weightDecay
                        * parameterValues;
                }

                parameterValues -= updateScale
                    * Vector256.LoadUnsafe(ref update[index]);
                parameterValues.StoreUnsafe(ref data[index]);
            }
        }

        for (; index < data.Length; index++)
        {
            if (applyWeightDecay)
            {
                data[index] -= options.LearningRate
                    * options.WeightDecay
                    * data[index];
            }

            data[index] -= options.LearningRate
                * finalScale
                * update[index];
        }
    }
}
