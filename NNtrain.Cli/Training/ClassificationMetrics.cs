namespace NNtrain;

internal static class ClassificationMetrics
{
    private static int CountCorrect(
        IReadOnlyList<float> logits,
        IReadOnlyList<int> answers,
        int classCount)
    {
        int correct = 0;
        for (int sample = 0; sample < answers.Count; sample++)
        {
            if (ArgMax(logits, sample * classCount, classCount)
                == answers[sample])
            {
                correct++;
            }
        }

        return correct;
    }

    internal static CudaClassificationCorrectCountReadback?
        BeginCountCorrect(
            Tensor logits,
            int[] answers,
            int classCount)
        => Tensor.ExecutionDevice == TensorDevice.Cuda
            ? logits.BeginCudaClassificationCorrectCount(
                answers,
                classCount)
            : null;

    internal static int CompleteCountCorrect(
        Tensor logits,
        IReadOnlyList<int> answers,
        int classCount,
        CudaClassificationCorrectCountReadback? readback)
        => readback is not null
            ? readback.CompleteAndReturn()
            : CountCorrect(logits.Data, answers, classCount);

    internal static int CountCorrectForMetrics(
        Tensor logits,
        int[] answers,
        int classCount)
    {
        ArgumentNullException.ThrowIfNull(logits);
        ArgumentNullException.ThrowIfNull(answers);
        using CudaClassificationCorrectCountReadback? readback =
            BeginCountCorrect(logits, answers, classCount);
        return CompleteCountCorrect(
            logits,
            answers,
            classCount,
            readback);
    }

    private static int ArgMax(
        IReadOnlyList<float> values,
        int offset,
        int count)
    {
        int result = 0;
        float best = values[offset];
        for (int index = 1; index < count; index++)
        {
            if (values[offset + index] > best)
            {
                best = values[offset + index];
                result = index;
            }
        }

        return result;
    }

}
