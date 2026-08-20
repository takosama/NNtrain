using System.Diagnostics;
using NNtrain;

namespace NNtrain.Benchmarks;

internal static class TenStepCompareProfiler
{
    private const int Steps = 10;
    private const int Vocabulary = 256;
    private const int Batch = 2;
    private const int Sequence = 16;
    private const int Width = 32;
    private const int Hidden = 64;
    private const int Layers = 2;
    private const int KeyWidth = 4;
    private const int ValueWidth = 4;
    private const int Seed = 20260820;

    internal static void Run()
    {
        Console.WriteLine("10-step ForgetMemoryV2 CPU/CUDA training comparison");
        Console.WriteLine(
            $"conditions: steps={Steps}, batch={Batch}, sequence={Sequence}, "
            + $"vocab={Vocabulary}, width={Width}, hidden={Hidden}, "
            + $"layers={Layers}, key/value={KeyWidth}/{ValueWidth}, "
            + "dtype=bfloat16, dropout=0, lr=0.001, weight_decay=0.01, "
            + "optimizer=Composite(NekoMuon+AdamW), seed=" + Seed);
        Console.WriteLine(
            $"cuda adapters={Tensor.CudaDeviceCount}, "
            + "cuda devices=[0,1] when two adapters are available");

        int[] tokens = CreateTokens();
        int[] targets = CreateTargets();
        Result cpu = RunDevice(
            "CPU",
            TensorDevice.Cpu,
            [0],
            tokens,
            targets);
        Console.WriteLine();
        Result gpu = RunDevice(
            "CUDA",
            TensorDevice.Cuda,
            Tensor.CudaDeviceCount >= 2 ? [0, 1] : [0],
            tokens,
            targets);

        Console.WriteLine();
        Console.WriteLine("summary");
        Console.WriteLine(
            $"CPU : total={cpu.TotalMs:F2} ms, mean={cpu.MeanMs:F2} ms/step, "
            + $"final_loss={cpu.FinalLoss:F6}");
        Console.WriteLine(
            $"CUDA: total={gpu.TotalMs:F2} ms, mean={gpu.MeanMs:F2} ms/step, "
            + $"final_loss={gpu.FinalLoss:F6}, speedup={cpu.MeanMs / gpu.MeanMs:F2}x");
        Console.WriteLine(
            "note: CUDA step 1 includes ILGPU kernel compilation; "
            + "GPU tensors use resident compute-view caches and BF16 host storage.");
    }

    private static Result RunDevice(
        string name,
        TensorDevice device,
        int[] deviceIndices,
        int[] tokens,
        int[] targets)
    {
        Tensor.ExecutionDevice = device;
        Tensor.CudaDeviceIndices = deviceIndices;
        var model = new ForgetMemoryV2Gpt(
            Vocabulary,
            Sequence,
            Width,
            Hidden,
            Layers,
            KeyWidth,
            ValueWidth,
            retentionMinimum: 0.5f,
            retentionMaximum: 0.99f,
            random: new Random(Seed),
            initializationScale: 0.02f,
            dropout: 0f,
            dtype: TensorDType.BFloat16);
        IOptimizer optimizer = optim.Composite(
            optim.NekoMuon(
                model.HiddenWeightParameters,
                lr: 0.001f,
                newton_schulz_steps: 2,
                newton_schulz_interval: 2,
                weight_decay: 0.01f),
            optim.AdamW(
                model.AuxiliaryParameters,
                lr: 0.001f,
                weight_decay: 0.01f,
                bf16_first_moment: true,
                bf16_second_moment: true));

        var times = new double[Steps];
        float finalLoss = float.NaN;
        for (int step = 0; step < Steps; step++)
        {
            model.ZeroGrad();
            long started = Stopwatch.GetTimestamp();
            Tensor logits = model.Forward(tokens, Batch, Sequence);
            Tensor loss = logits.CrossEntropyWithLogits(targets);
            finalLoss = loss.item();
            loss.Backward();
            nn.utils.clip_grad_norm_(model.parameters(), max_norm: 1f);
            optimizer.Step();
            times[step] = Stopwatch.GetElapsedTime(started).TotalMilliseconds;
            Console.WriteLine(
                $"{name,-4} step {step + 1,2}: {times[step],10:F2} ms, "
                + $"loss={finalLoss:F6}");
        }

        return new Result(times.Sum(), times.Average(), finalLoss);
    }

    private static int[] CreateTokens()
        => Enumerable.Range(0, Batch * Sequence)
            .Select(index => (index * 37 + 11) % Vocabulary)
            .ToArray();

    private static int[] CreateTargets()
        => Enumerable.Range(0, Batch * Sequence)
            .Select(index => (index * 53 + 7) % Vocabulary)
            .ToArray();

    private readonly record struct Result(
        double TotalMs,
        double MeanMs,
        float FinalLoss);
}
