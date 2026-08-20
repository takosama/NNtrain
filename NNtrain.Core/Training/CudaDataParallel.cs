namespace NNtrain;

/// <summary>CUDA data-parallel forward/backward for language-model batches.</summary>
public static class CudaDataParallel
{
    public static float ForwardBackward(
        IWikiLanguageModel model,
        int[] input,
        int[] target,
        int batchSize,
        int sequenceLength,
        int ignoreIndex = Tensor.DefaultCrossEntropyIgnoreIndex)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(target);
        if (Tensor.ExecutionDevice != TensorDevice.Cuda)
            throw new InvalidOperationException("CUDA execution must be selected.");
        if (input.Length != target.Length
            || input.Length != checked(batchSize * sequenceLength))
        {
            throw new ArgumentException("Input and target must match the batch shape.");
        }

        int[] devices = Tensor.CudaDeviceIndices
            .Take(Math.Min(batchSize, Tensor.CudaDeviceIndices.Count))
            .ToArray();
        if (devices.Length == 1)
        {
            Tensor logits = model.Forward(input, batchSize, sequenceLength);
            Tensor loss = logits.CrossEntropyWithLogits(target, ignoreIndex: ignoreIndex);
            float value = loss.item();
            loss.BackwardAndRelease();
            return value;
        }

        Parameter[] parameters = model.Parameters().ToArray();
        foreach (Parameter parameter in parameters)
            parameter.T.PrepareCudaGradientBuffers(devices);

        int totalValid = target.Count(value => value != ignoreIndex);
        if (totalValid == 0)
            throw new ArgumentException("At least one target must be valid.", nameof(target));
        var weightedLosses = new double[devices.Length];
        Parallel.For(0, devices.Length, shard =>
        {
            int batchStart = batchSize * shard / devices.Length;
            int batchEnd = batchSize * (shard + 1) / devices.Length;
            int shardBatch = batchEnd - batchStart;
            int elementStart = batchStart * sequenceLength;
            int elementCount = shardBatch * sequenceLength;
            int[] shardInput = input.AsSpan(elementStart, elementCount).ToArray();
            int[] shardTarget = target.AsSpan(elementStart, elementCount).ToArray();
            int shardValid = shardTarget.Count(value => value != ignoreIndex);

            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndex = devices[shard];
            Tensor logits = model.Forward(
                shardInput,
                shardBatch,
                sequenceLength);
            Tensor loss = logits.CrossEntropyWithLogits(
                shardTarget,
                ignoreIndex: ignoreIndex);
            float weight = (float)shardValid / totalValid;
            weightedLosses[shard] = loss.item() * shardValid;
            loss.BackwardAndRelease([weight]);
        });

        TensorCudaKernels.AllReduceGradientsResident(parameters, devices);
        return (float)(weightedLosses.Sum() / totalValid);
    }
}
