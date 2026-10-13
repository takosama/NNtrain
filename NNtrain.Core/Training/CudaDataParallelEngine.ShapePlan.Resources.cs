using NNtrain.Cuda.Execution;
using NNtrain.Cuda.Memory;
using NNtrain.Runtime.Execution;

namespace NNtrain;

public sealed partial class CudaDataParallelEngine
{
    private sealed partial class CudaTrainingShapePlan
    {
        private void DisposeDeviceGraph(int replicaIndex)
        {
            CompiledDeviceGraph? graph = Interlocked.Exchange(
                ref _compiledGraphs[replicaIndex],
                null);
            var failures = new List<Exception>();
            TryDispose(graph, failures);
            CudaGraphCaptureScope? captureMemory = Interlocked.Exchange(
                ref _captureMemoryScopes[replicaIndex],
                null);
            TryDispose(captureMemory, failures);
            NativeCudaBuffer<float>? lossSlot = Interlocked.Exchange(
                ref _graphLossSlots[replicaIndex],
                null);
            TryDispose(lossSlot, failures);
            CudaBfp8GraphGradientPublication? bfp8Publication =
                Interlocked.Exchange(
                    ref _graphBfp8Publications[replicaIndex],
                    null);
            TryDispose(bfp8Publication, failures);
            CudaGraphRngState? rng = Interlocked.Exchange(
                ref _graphRngStates[replicaIndex],
                null);
            TryDispose(rng, failures);
            CudaGraphBatchInputs? inputs = Interlocked.Exchange(
                ref _graphInputs[replicaIndex],
                null);
            TryDispose(inputs, failures);
            if (failures.Count != 0)
            {
                throw new AggregateException(
                    "CUDA device graph cleanup failed.",
                    failures);
            }
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            var failures = new List<Exception>();
            for (int device = 0; device < _compiledGraphs.Length; device++)
            {
                try
                {
                    DisposeDeviceGraph(device);
                }
                catch (Exception exception)
                {
                    failures.Add(exception);
                }
            }
            if (failures.Count != 0)
            {
                throw new AggregateException(
                    "CUDA training shape plan cleanup failed.",
                    failures);
            }
        }

        private static void TryDispose(
            IDisposable? resource,
            List<Exception> failures)
        {
            if (resource is null)
                return;
            try
            {
                resource.Dispose();
            }
            catch (Exception exception)
            {
                failures.Add(exception);
            }
        }

    }
}
