using System.Diagnostics;
using NNtrain;

namespace NNtrain.Benchmarks;

internal static class TenStepCompareProfiler
{
    private const int DefaultSteps = 10;
    private const int Vocabulary = 4096;
    private const int Batch = 2;
    private const int Sequence = 1024;
    private const int Width = 256;
    private const int Hidden = 512;
    private const int Layers = 8;
    private const int KeyWidth = 16;
    private const int ValueWidth = 16;
    private const int Seed = 1234;
    private const float Dropout = 0.1f;
    private const float LearningRate = 0.0003f;
    private const float WeightDecay = 0.01f;

    internal static void Run(int steps = DefaultSteps, bool cudaOnly = false)
    {
        Console.WriteLine("10-step ForgetMemoryV2 CPU/CUDA training comparison");
        Console.WriteLine(
            $"conditions: steps={steps}, batch={Batch}, sequence={Sequence}, "
            + $"vocab={Vocabulary}, width={Width}, hidden={Hidden}, "
            + $"layers={Layers}, key/value={KeyWidth}/{ValueWidth}, "
            + $"dtype=bfloat16, dropout={Dropout}, lr={LearningRate}, "
            + $"weight_decay={WeightDecay}, "
            + "optimizer=Composite(NekoMuon+AdamW), "
            + "adamw_moments=f32/f32, seed=" + Seed);
        Console.WriteLine(
            $"cuda adapters={Tensor.CudaDeviceCount}, "
            + "cuda devices=[0,1] when two adapters are available");

        int[] tokens = CreateTokens();
        int[] targets = CreateTargets();
        Result? cpu = null;
        if (!cudaOnly)
        {
            cpu = RunDevice(
                "CPU",
                TensorDevice.Cpu,
                [0],
                tokens,
                targets,
                steps);
            Console.WriteLine();
        }
        Result gpu1 = RunDevice(
            "GPU1",
            TensorDevice.Cuda,
            [0],
            tokens,
            targets,
            steps);
        Result? gpu2 = null;
        if (Tensor.CudaDeviceCount >= 2)
        {
            Console.WriteLine();
            gpu2 = RunDevice(
                "GPU2",
                TensorDevice.Cuda,
                [0, 1],
                tokens,
                targets,
                steps);
        }

        Console.WriteLine();
        Console.WriteLine("summary");
        if (cpu is { } cpuResult)
        {
            Console.WriteLine(
                $"CPU : total={cpuResult.TotalMs:F2} ms, "
                + $"mean={cpuResult.MeanMs:F2} ms/step, "
                + $"final_loss={cpuResult.FinalLoss:F6}");
        }
        Console.WriteLine(
            $"GPU1: total={gpu1.TotalMs:F2} ms, mean={gpu1.MeanMs:F2} ms/step, "
            + $"final_loss={gpu1.FinalLoss:F6}"
            + (cpu is { } speedupCpu
                ? $", speedup={speedupCpu.MeanMs / gpu1.MeanMs:F2}x"
                : ""));
        if (gpu2 is { } two)
        {
            Console.WriteLine(
                $"GPU2: total={two.TotalMs:F2} ms, mean={two.MeanMs:F2} ms/step, "
                + $"final_loss={two.FinalLoss:F6}"
                + (cpu is { } twoGpuSpeedupCpu
                    ? $", speedup={twoGpuSpeedupCpu.MeanMs / two.MeanMs:F2}x"
                    : ""));
        }
        Console.WriteLine(
            "note: CUDA step 1 includes kernel/library initialization; "
            + "Linear uses cuBLAS CUBLAS_COMPUTE_32F_FAST_16BF "
            + "(FP32 buffers with BF16-fast compute, FP32 accumulation); "
            + "AdamW/NekoMuon FP32 "
            + "master weights, moments, and workspaces remain GPU-resident.");
    }

    private static Result RunDevice(
        string name,
        TensorDevice device,
        int[] deviceIndices,
        int[] tokens,
        int[] targets,
        int steps)
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
            dropout: Dropout,
            dtype: TensorDType.BFloat16);
        NekoMuon neko = (NekoMuon)optim.NekoMuon(
                model.HiddenWeightParameters,
                lr: LearningRate,
                newton_schulz_steps: 5,
                newton_schulz_interval: 5,
                weight_decay: WeightDecay);
        AdamW adam = (AdamW)optim.AdamW(
                model.AuxiliaryParameters,
                lr: LearningRate,
                weight_decay: WeightDecay,
                bf16_first_moment: false,
                bf16_second_moment: false);

        var times = new double[steps];
        var phase = new double[5];
        float finalLoss = float.NaN;
        for (int step = 0; step < steps; step++)
        {
            var currentPhase = new double[5];
            int gen0Before = GC.CollectionCount(0);
            int gen1Before = GC.CollectionCount(1);
            int gen2Before = GC.CollectionCount(2);
            long phaseStart = Stopwatch.GetTimestamp();
            model.ZeroGrad();
            currentPhase[0] = Stopwatch.GetElapsedTime(phaseStart).TotalMilliseconds;
            phase[0] += currentPhase[0];
            long started = Stopwatch.GetTimestamp();
            phaseStart = started;
            if (device == TensorDevice.Cuda && deviceIndices.Length > 1)
            {
                finalLoss = CudaDataParallel.ForwardBackward(
                    model,
                    tokens,
                    targets,
                    Batch,
                    Sequence);
            }
            else
            {
                Tensor logits = model.Forward(tokens, Batch, Sequence);
                Tensor loss = logits.CrossEntropyWithLogits(targets);
                finalLoss = loss.item();
                if (device == TensorDevice.Cuda)
                    loss.BackwardAndRelease();
                else
                    loss.Backward();
            }
            currentPhase[1] = Stopwatch.GetElapsedTime(phaseStart).TotalMilliseconds;
            phase[1] += currentPhase[1];
            phaseStart = Stopwatch.GetTimestamp();
            nn.utils.clip_grad_norm_(model.parameters(), max_norm: 1f);
            currentPhase[2] = Stopwatch.GetElapsedTime(phaseStart).TotalMilliseconds;
            phase[2] += currentPhase[2];
            phaseStart = Stopwatch.GetTimestamp();
            neko.Step();
            currentPhase[3] = Stopwatch.GetElapsedTime(phaseStart).TotalMilliseconds;
            phase[3] += currentPhase[3];
            phaseStart = Stopwatch.GetTimestamp();
            adam.Step();
            currentPhase[4] = Stopwatch.GetElapsedTime(phaseStart).TotalMilliseconds;
            phase[4] += currentPhase[4];
            times[step] = Stopwatch.GetElapsedTime(started).TotalMilliseconds;
            Console.WriteLine(
                $"{name,-4} step {step + 1,2}: {times[step],10:F2} ms, "
                + $"loss={finalLoss:F6}, graph={currentPhase[1]:F2}, "
                + $"clip={currentPhase[2]:F2}, neko={currentPhase[3]:F2}, "
                + $"adam={currentPhase[4]:F2}, GC="
                + $"{GC.CollectionCount(0) - gen0Before}/"
                + $"{GC.CollectionCount(1) - gen1Before}/"
                + $"{GC.CollectionCount(2) - gen2Before}");
        }

        Console.WriteLine(
            $"{name,-4} phases mean: zero={phase[0] / steps:F2} ms, "
            + $"graph={phase[1] / steps:F2} ms, "
            + $"clip={phase[2] / steps:F2} ms, "
            + $"neko={phase[3] / steps:F2} ms, "
            + $"adam={phase[4] / steps:F2} ms");

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
