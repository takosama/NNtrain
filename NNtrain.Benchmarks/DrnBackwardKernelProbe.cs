using System.Diagnostics;
using System.Text.Json;

namespace NNtrain.Benchmarks;

internal static class DrnBackwardKernelProbe
{
    internal static void RunBias(string outputPath)
    {
        if (File.Exists(outputPath)) throw new IOException("Use a new output path.");
        if (!Tensor.IsCudaAvailable(0)) throw new InvalidOperationException("CUDA0 required.");
        using var device = TensorExecutionContext.Push(new TorchDevice(TensorDevice.Cuda, 0));
        var accelerator = ForgetMemoryV2Cuda.GetAccelerator(0);
        var results = new List<object>();
        foreach (bool bf16 in new[] { false, true })
        foreach (var (rows, width) in new[] { (16384,512), (16384,1536), (16384,160), (4097,129), (1024,45) })
        {
            var rng = new Random(437);
            float[] source = Enumerable.Range(0, rows * width)
                .Select(_ => (float)(rng.NextDouble() - .5) * .02f).ToArray();
            if (bf16) for (int i = 0; i < source.Length; i++) source[i] = TensorStorageCodec.RoundToBFloat16(source[i]);
            using var fp = bf16 ? null : accelerator.Allocate1D(source);
            using var bp = bf16 ? accelerator.Allocate1D(source.Select(TensorStorageCodec.EncodeBFloat16).ToArray()) : null;
            using var bias = accelerator.Allocate1D<float>(width);
            void Launch()
            {
                bias.MemSetToZero();
                CudaTensorNative.LinearBiasBackward(0, fp?.NativePtr ?? bp!.NativePtr, bias.NativePtr, rows, width, bf16);
            }
            Launch(); accelerator.Synchronize();
            var actual = new float[width]; bias.CopyToCPU(actual);
            var expected = new double[width];
            for (int i = 0; i < source.Length; i++) expected[i % width] += source[i];
            double error = actual.Select((v, i) => Math.Abs(v - expected[i])).Max();
            if (error > 1e-5) throw new InvalidOperationException($"Bias error {error} exceeds 1e-5.");
            for (int i = 0; i < 5; i++) Launch();
            accelerator.Synchronize(); var watch = Stopwatch.StartNew();
            for (int i = 0; i < 50; i++) Launch();
            accelerator.Synchronize(); watch.Stop();
            var result = new { BFloat16 = bf16, Rows = rows, Width = width,
                MeanMilliseconds = watch.Elapsed.TotalMilliseconds / 50, MaxAbsoluteError = error };
            results.Add(result); Console.WriteLine(JsonSerializer.Serialize(result));
        }
        using var file = new FileStream(outputPath, FileMode.CreateNew);
        JsonSerializer.Serialize(file, results, new JsonSerializerOptions { WriteIndented = true });
    }

    internal static void Run(string outputPath, string? referencePath = null)
    {
        if (File.Exists(outputPath)) throw new IOException("Use a new output path.");
        if (!Tensor.IsCudaAvailable(0)) throw new InvalidOperationException("CUDA0 required.");
        using var device = TensorExecutionContext.Push(new TorchDevice(TensorDevice.Cuda, 0));
        using var policy = CudaDispatchPolicy.Push(CudaDispatchPolicy.Defaults);
        var accelerator = ForgetMemoryV2Cuda.GetAccelerator(0);
        var results = new List<object>();
        int caseIndex = 0;
        foreach (var (batch, sequence, key, value) in new[]
        { (8,2048,32,32), (16,512,16,16), (8,129,17,17), (8,65,1,19), (2,33,32,32), (2,33,48,32) })
        {
            int width = 2 * key + 3 * value;
            var rng = new Random(8301);
            float[] Values(int length) => Enumerable.Range(0, length)
                .Select(_ => (float)(rng.NextDouble() - .5) * .6f).ToArray();
            ushort[] bits = Values(batch * sequence * width).Select(v =>
                (ushort)(BitConverter.SingleToInt32Bits(TensorStorageCodec.RoundToBFloat16(v)) >> 16)).ToArray();
            using var projected = accelerator.Allocate1D(bits);
            using var output = accelerator.Allocate1D<ushort>(batch * sequence * value);
            using var history = accelerator.Allocate1D<float>(batch * sequence * key * value);
            using var state = accelerator.Allocate1D<float>(batch * key * value);
            state.MemSetToZero();
            if (!CudaForgetMemoryTensorCore.TryForward(accelerator, projected, output,
                history, state, batch, sequence, width, key, value, .37f, 2))
                CudaForgetMemoryNative.Forward(accelerator, 0, projected.NativePtr, 0,
                    output.NativePtr, history.NativePtr, state.NativePtr, batch, sequence,
                    width, key, value, .37f, 2, true);
            using var dy = accelerator.Allocate1D(Values(output.Length));
            using var dp = accelerator.Allocate1D(Values(projected.Length));
            using var ds = accelerator.Allocate1D(Values(state.Length));
            using var previous = accelerator.Allocate1D<float>(state.Length);
            int preparedLength = CudaForgetMemoryNative.PreparedBackwardScratchLength(batch, sequence, key, value);
            using var prepared = preparedLength > 0 ? accelerator.Allocate1D<float>(preparedLength) : null;
            void Launch() => CudaForgetMemoryNative.Backward(accelerator, 0, projected.NativePtr,
                dp.NativePtr, dy.NativePtr, history.NativePtr, ds.NativePtr, previous.NativePtr,
                batch, sequence, width, key, value, .37f, 2, true, prepared?.NativePtr ?? 0);
            Launch(); accelerator.Synchronize();
            float[] gradients = new float[dp.Length]; dp.CopyToCPU(gradients);
            float[] stateGradient = new float[ds.Length]; ds.CopyToCPU(stateGradient);
            string artifact = outputPath + $".case{caseIndex}.bin";
            double square = 0, error = 0; float maxError = 0;
            using (var writer = new BinaryWriter(new FileStream(artifact, FileMode.CreateNew)))
            using (var reader = referencePath is null ? null : new BinaryReader(File.OpenRead(referencePath + $".case{caseIndex}.bin")))
            {
                foreach (float actual in gradients.Concat(stateGradient))
                {
                    if (!float.IsFinite(actual)) throw new InvalidOperationException("Nonfinite gradient.");
                    writer.Write(actual);
                    if (reader is null) continue;
                    float expected = reader.ReadSingle();
                    double delta = (double)actual - expected;
                    maxError = Math.Max(maxError, (float)Math.Abs(delta));
                    square += (double)expected * expected; error += delta * delta;
                }
                if (reader is not null && reader.BaseStream.Position != reader.BaseStream.Length)
                    throw new InvalidDataException("Reference gradient length mismatch.");
            }
            if (referencePath is not null && maxError > 6e-5f)
                throw new InvalidOperationException($"Gradient error {maxError} exceeds existing 6e-5 threshold.");
            // Reset buffers before each timed backward; memset work is included in both variants.
            void ResetLaunch() { dp.MemSetToZero(); ds.MemSetToZero(); Launch(); }
            for (int i = 0; i < 5; i++) ResetLaunch();
            accelerator.Synchronize(); var watch = Stopwatch.StartNew();
            for (int i = 0; i < 20; i++) ResetLaunch();
            accelerator.Synchronize(); watch.Stop();
            var result = new { Batch = batch, Sequence = sequence, Key = key, Value = value,
                PreparedScratchBytes = (long)preparedLength * sizeof(float),
                MeanMilliseconds = watch.Elapsed.TotalMilliseconds / 20,
                Compared = referencePath is not null, MaxAbsoluteError = maxError,
                RelativeL2Error = Math.Sqrt(error / Math.Max(square, 1e-30)) };
            results.Add(result); Console.WriteLine(JsonSerializer.Serialize(result)); caseIndex++;
        }
        using var file = new FileStream(outputPath, FileMode.CreateNew);
        JsonSerializer.Serialize(file, results, new JsonSerializerOptions { WriteIndented = true });
    }
}
