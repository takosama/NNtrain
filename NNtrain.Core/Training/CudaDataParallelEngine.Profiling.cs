using NNtrain.Cuda.Execution;
using NNtrain.Cuda.Memory;
using NNtrain.Runtime.Execution;

namespace NNtrain;

public sealed partial class CudaDataParallelEngine
{
    /// <summary>
    /// Diagnostic variant that synchronizes CUDA after every major phase.
    /// It deliberately calls the same ForwardLoss entry point as compiled
    /// training so BFP8 models retain their direct-BF16 loss head. The old
    /// decomposed Forward -> BFP8 logits -> CrossEntropy sequence is not a
    /// production path and must not be used for bottleneck attribution.
    /// </summary>
    internal CudaDataParallelProfile ForwardBackwardProfiled(
        int[] input,
        int[] target,
        int batchSize,
        int sequenceLength,
        int ignoreIndex = Tensor.DefaultCrossEntropyIgnoreIndex)
    {
        lock (_sync)
        {
            ThrowIfDisposed();
            // A profiled step deliberately executes the eager autograd path so
            // each phase can be synchronized independently. Keeping a compiled
            // training graph alive here pins its capture reservation while the
            // eager path asks the same lane for a second activation workspace.
            // At production shapes that unnecessary overlap can exhaust VRAM
            // after an otherwise successful benchmark. Retire only the cached
            // shape/graph resources; parameters, optimizer state and gradient
            // reduction plans remain resident and warm.
            ReleaseTrainingShapePlans();
            PrepareParameterResidency();
            using IDisposable precisionScope =
                TensorExecutionContext.PushPrecisionPolicy(
                    ResolvePrecisionPolicy(_model.PrecisionMode));
            ValidateBatch(input, target, batchSize, sequenceLength);

            int[] devices = GetDevices(batchSize);
            int[] shardBatches = AllocateShardBatches(batchSize, devices);
            int[] shardStarts = GetShardStarts(shardBatches);
            void SynchronizeAll()
            {
                foreach (int device in devices)
                    ForgetMemoryV2Cuda.GetAccelerator(device).Synchronize();
            }

            SynchronizeAll();
            var totalTimer = System.Diagnostics.Stopwatch.StartNew();
            var prepareTimer = System.Diagnostics.Stopwatch.StartNew();
            Parameter[] parameters = _parameters;
            CudaBFloat16GradientAllReducePlan? bfloat16Plan =
                GetBFloat16GradientPlan(parameters, devices);
            CudaBfp8GradientAllReducePlan? bfp8Plan =
                GetBfp8GradientPlan(parameters, devices);
            ICudaGradientReductionPlan? reductionPlan =
                (ICudaGradientReductionPlan?)bfp8Plan ?? bfloat16Plan;
            if (reductionPlan is null)
            {
                foreach (Parameter parameter in parameters)
                    parameter.T.PrepareCudaGradientBuffers(devices);
            }
            int totalValid = target.Count(value => value != ignoreIndex);
            if (totalValid == 0)
            {
                throw new ArgumentException(
                    "At least one target must be valid.", nameof(target));
            }
            long reductionStepId = bfp8Plan?.BeginStep()
                ?? bfloat16Plan?.BeginStep()
                ?? 0;
            SynchronizeAll();
            prepareTimer.Stop();

            var weightedLosses = new double[devices.Length];
            var shards = new CudaShardProfile[devices.Length];
            try
            {
                Parallel.For(0, devices.Length, shard =>
                {
                    var shardTimer = System.Diagnostics.Stopwatch.StartNew();
                int batchStart = shardStarts[shard];
                int shardBatch = shardBatches[shard];
                int elementStart = batchStart * sequenceLength;
                int elementCount = shardBatch * sequenceLength;
                int[] shardInput = input.AsSpan(
                    elementStart, elementCount).ToArray();
                int[] shardTarget = target.AsSpan(
                    elementStart, elementCount).ToArray();
                int shardValid = shardTarget.Count(
                    value => value != ignoreIndex);
                double dataPreparation = shardTimer.Elapsed.TotalMilliseconds;

                using IDisposable shardPrecisionScope =
                    TensorExecutionContext.PushPrecisionPolicy(
                        ResolvePrecisionPolicy(_model.PrecisionMode));
                using IDisposable scope = TensorExecutionContext.Push(
                    new TorchDevice(TensorDevice.Cuda, devices[shard]));
                bfp8Plan?.BeginDeviceStep(
                    reductionStepId, devices[shard]);
                bfloat16Plan?.BeginDeviceStep(
                    reductionStepId, devices[shard]);
                using IDisposable? reductionScope = reductionPlan is null
                    ? null
                    : CudaGradientReductionContext.Push(
                        reductionPlan, devices[shard], reductionStepId);
                NativeCudaDevice accelerator =
                    ForgetMemoryV2Cuda.GetAccelerator(devices[shard]);

                var phaseTimer = System.Diagnostics.Stopwatch.StartNew();
                Tensor loss = _model.ForwardLoss(
                    shardInput,
                    shardTarget,
                    shardBatch,
                    sequenceLength,
                    ignoreIndex);
                accelerator.Synchronize();
                double forward = phaseTimer.Elapsed.TotalMilliseconds;

                phaseTimer.Restart();
                weightedLosses[shard] = loss.item() * shardValid;
                accelerator.Synchronize();
                double lossMilliseconds = phaseTimer.Elapsed.TotalMilliseconds;

                phaseTimer.Restart();
                float weight = (float)shardValid / totalValid;
                loss.BackwardAndRelease([weight]);
                accelerator.Synchronize();
                double backward = phaseTimer.Elapsed.TotalMilliseconds;
                    shards[shard] = new CudaShardProfile(
                        devices[shard],
                        shardBatch,
                        dataPreparation,
                        forward,
                        lossMilliseconds,
                        backward);
                });
            }
            catch
            {
                bfp8Plan?.Abort(reductionStepId);
                bfloat16Plan?.Abort(reductionStepId);
                throw;
            }

            var allReduceTimer = System.Diagnostics.Stopwatch.StartNew();
            if (bfp8Plan is not null)
            {
                bfp8Plan.Complete(reductionStepId);
            }
            else if (bfloat16Plan is not null)
            {
                bfloat16Plan.Complete(reductionStepId);
            }
            else
            {
                TensorCudaKernels.FlatGradientPlan plan =
                    GetFlatGradientPlan(parameters, devices);
                TensorCudaKernels.AllReduceGradientsResident(
                    parameters, devices, plan);
            }
            SynchronizeAll();
            allReduceTimer.Stop();
            totalTimer.Stop();
            return new CudaDataParallelProfile(
                (float)(weightedLosses.Sum() / totalValid),
                prepareTimer.Elapsed.TotalMilliseconds,
                allReduceTimer.Elapsed.TotalMilliseconds,
                totalTimer.Elapsed.TotalMilliseconds,
                shards);
        }
    }

}
