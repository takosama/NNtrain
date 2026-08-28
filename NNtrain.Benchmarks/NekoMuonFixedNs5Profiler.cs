using System.Diagnostics;
using NNtrain.Runtime.Execution;

namespace NNtrain.Benchmarks;

internal static class NekoMuonFixedNs5Profiler
{
    internal static void Run(
        int parameterCount,
        int rows,
        int columns,
        int warmup,
        int iterations)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(parameterCount);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(columns);
        ArgumentOutOfRangeException.ThrowIfNegative(warmup);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(iterations);
        if (!Tensor.IsCudaAvailable())
            throw new InvalidOperationException("CUDA is not available.");

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousDevices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = [0];
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            using IDisposable precision =
                TensorExecutionContext.PushPrecisionPolicy(
                    PrecisionPolicy.Mix8_32);
            ScenarioResult scalar = RunScenario(
                disableBatching: true,
                parameterCount,
                rows,
                columns,
                warmup,
                iterations);
            ScenarioResult grouped = RunScenario(
                disableBatching: false,
                parameterCount,
                rows,
                columns,
                warmup,
                iterations);

            Console.WriteLine("CUDA NekoMuon fixed NS5 benchmark");
            Console.WriteLine(
                $"device=0, parameters={parameterCount}, " +
                $"shape=[{rows},{columns}], warmup={warmup}, " +
                $"iterations={iterations}, mix8_32 block=128");
            Print("scalar", scalar);
            Print("grouped", grouped);
            Console.WriteLine(
                $"speedup={scalar.MedianMilliseconds / grouped.MedianMilliseconds:F2}x, " +
                $"GEMM launch reduction=" +
                $"{scalar.Telemetry.GemmLaunchCount}->" +
                $"{grouped.Telemetry.GemmLaunchCount} " +
                $"({(1d - (double)grouped.Telemetry.GemmLaunchCount
                    / scalar.Telemetry.GemmLaunchCount) * 100d:F1}%)");
        }
        finally
        {
            Tensor.CudaDeviceIndices = previousDevices;
            Tensor.ExecutionDevice = previousDevice;
        }
    }

    private static ScenarioResult RunScenario(
        bool disableBatching,
        int parameterCount,
        int rows,
        int columns,
        int warmup,
        int iterations)
    {
        int length = checked(rows * columns);
        float[] initial = Values(length, 7, 0.18f);
        float[] gradient = Values(length, 29, 0.035f);
        Parameter[] parameters = Enumerable.Range(0, parameterCount)
            .Select(index => CreateParameter(
                initial,
                rows,
                columns,
                $"matrix.{index}"))
            .ToArray();
        var optimizer = new NekoMuon(
            parameters,
            new NekoMuonOptions
            {
                LearningRate = 0.002f,
                BetaFast = 0.8f,
                BetaSlow = 0.95f,
                Rho = 0.7f,
                Epsilon = 1e-6f,
                MaxNewtonSchulzSteps = 5,
                NewtonSchulzInterval = 1,
                NewtonSchulzDepthMode =
                    NekoMuonNewtonSchulzDepthMode.Fixed,
                NewtonSchulzDepth = 5f,
                WeightDecay = 0.01f,
            },
            CudaDispatchPolicy.Defaults with
            {
                DisableBatchedNekoMuon = disableBatching,
                NekoMuonBatchSize = 8,
            });
        try
        {
            optimizer.prepare();
            for (int index = 0; index < warmup; index++)
                Step(parameters, optimizer, gradient);

            NekoMuonFixedNs5TelemetrySnapshot before =
                NekoMuonFixedNs5Telemetry.Snapshot;
            var milliseconds = new double[iterations];
            for (int index = 0; index < iterations; index++)
            {
                foreach (Parameter parameter in parameters)
                    parameter.T.SetCudaGradient(gradient, 0);
                var timer = Stopwatch.StartNew();
                optimizer.step();
                timer.Stop();
                milliseconds[index] = timer.Elapsed.TotalMilliseconds;
                optimizer.zero_grad();
            }
            Array.Sort(milliseconds);
            NekoMuonFixedNs5TelemetrySnapshot telemetry =
                NekoMuonFixedNs5Telemetry.Snapshot - before;
            return new ScenarioResult(
                milliseconds.Average(),
                Percentile(milliseconds, 0.5),
                Percentile(milliseconds, 0.95),
                optimizer.CudaBatchCapacity,
                optimizer.ConfiguredCudaScratchBytesPerDevice,
                telemetry);
        }
        finally
        {
            optimizer.DisposeCudaResources();
            foreach (Parameter parameter in parameters)
                parameter.T.InvalidateCudaBuffers();
        }
    }

    private static void Step(
        IEnumerable<Parameter> parameters,
        NekoMuon optimizer,
        float[] gradient)
    {
        foreach (Parameter parameter in parameters)
            parameter.T.SetCudaGradient(gradient, 0);
        optimizer.step();
        optimizer.zero_grad();
    }

    private static Parameter CreateParameter(
        float[] values,
        int rows,
        int columns,
        string name)
    {
        var parameter = new Parameter(
            values,
            [rows, columns],
            name,
            WeightDecayPolicy.Apply);
        parameter.T.ConvertStorageInPlace(
            TensorDType.Bfp8,
            Bfp8QuantizationDescriptor.Block(128),
            preserveFloat32Master: true);
        _ = parameter.T.EnsureCudaBfp8Buffer(0);
        parameter.T.to(new TorchDevice(TensorDevice.Cuda, 0));
        return parameter;
    }

    private static float[] Values(int length, int offset, float scale)
        => Enumerable.Range(0, length)
            .Select(index => index == length - 1
                ? scale * 2.75f
                : MathF.Sin((index + offset) * 0.173f) * scale)
            .ToArray();

    private static double Percentile(double[] sorted, double percentile)
    {
        int index = Math.Clamp(
            (int)Math.Ceiling(percentile * sorted.Length) - 1,
            0,
            sorted.Length - 1);
        return sorted[index];
    }

    private static void Print(string name, ScenarioResult result)
        => Console.WriteLine(
            $"{name}: mean={result.MeanMilliseconds:F3} ms, " +
            $"p50={result.MedianMilliseconds:F3} ms, " +
            $"p95={result.P95Milliseconds:F3} ms, " +
            $"batch={result.BatchCapacity}, " +
            $"scratch={result.ScratchBytes / 1024d / 1024d:F2} MiB, " +
            $"GEMM={result.Telemetry.GemmLaunchCount}, " +
            $"NS kernels={result.Telemetry.KernelLaunchCount}");

    private sealed record ScenarioResult(
        double MeanMilliseconds,
        double MedianMilliseconds,
        double P95Milliseconds,
        int BatchCapacity,
        long ScratchBytes,
        NekoMuonFixedNs5TelemetrySnapshot Telemetry);
}
