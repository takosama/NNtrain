using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;

namespace NNtrain.Benchmarks;

internal static class DrnForwardKernelProbe
{
    // Run the identical executable with old/new native DLLs. Hash every
    // output/history/state byte, including continuation from nonzero state.
    internal static void Run(string outputPath)
    {
        if (File.Exists(outputPath)) throw new IOException("Use a new output file.");
        if (!Tensor.IsCudaAvailable(0)) throw new InvalidOperationException("CUDA0 required.");
        using var device = TensorExecutionContext.Push(new TorchDevice(TensorDevice.Cuda, 0));
        using var policy = CudaDispatchPolicy.Push(CudaDispatchPolicy.Defaults);
        var accelerator = ForgetMemoryV2Cuda.GetAccelerator(0);
        var results = new List<object>();
        foreach (var (batch, sequence, key, value) in new[]
        { (3,33,16,16), (2,257,16,48), (8,2048,32,32), (2,65,32,128), (2,33,48,32) })
        {
            int width = 2 * key + 3 * value;
            var rng = new Random(7301);
            ushort[] bits = Enumerable.Range(0, batch * sequence * width).Select(_ =>
                (ushort)(BitConverter.SingleToInt32Bits(TensorStorageCodec.RoundToBFloat16(
                    (float)(rng.NextDouble() - .5) * 1.7f)) >> 16)).ToArray();
            float[] initial = Enumerable.Range(0, batch * key * value)
                .Select(_ => (float)(rng.NextDouble() - .5) * .2f).ToArray();
            using var projected = accelerator.Allocate1D(bits);
            using var output = accelerator.Allocate1D<ushort>(batch * sequence * value);
            using var history = accelerator.Allocate1D<float>(batch * sequence * key * value);
            using var state = accelerator.Allocate1D(initial);
            void Launch()
            {
                if (!CudaForgetMemoryTensorCore.TryForward(accelerator, projected, output,
                    history, state, batch, sequence, width, key, value, .37f, 2))
                    throw new InvalidOperationException("Tensor Core path required.");
            }
            Launch(); accelerator.Synchronize();
            var actualOutput = new ushort[output.Length]; output.CopyToCPU(actualOutput);
            var actualHistory = new float[history.Length]; history.CopyToCPU(actualHistory);
            var actualState = new float[state.Length]; state.CopyToCPU(actualState);
            for (int i = 0; i < 5; i++) Launch();
            accelerator.Synchronize(); var watch = Stopwatch.StartNew();
            for (int i = 0; i < 20; i++) Launch();
            accelerator.Synchronize(); watch.Stop();
            var result = new { Batch = batch, Sequence = sequence, Key = key, Value = value,
                MeanMilliseconds = watch.Elapsed.TotalMilliseconds / 20,
                OutputHash = Hash(actualOutput), HistoryHash = Hash(actualHistory), StateHash = Hash(actualState) };
            results.Add(result); Console.WriteLine(JsonSerializer.Serialize(result));
        }
        using var file = new FileStream(outputPath, FileMode.CreateNew);
        JsonSerializer.Serialize(file, results, new JsonSerializerOptions { WriteIndented = true });
    }

    private static string Hash<T>(T[] values) where T : unmanaged
        => Convert.ToHexString(SHA256.HashData(MemoryMarshal.AsBytes(values.AsSpan())));
}
