using System.Diagnostics;
using NNtrain.Cuda.Interop;

namespace NNtrain;

public sealed partial class NekoMuon
{
    /// <summary>
    /// Returns scalar optimizer diagnostics without copying the large moment
    /// arrays. This is safe to query after <see cref="step"/> for training
    /// telemetry and makes adaptive Newton-Schulz depth observable.
    /// </summary>
    public NekoMuonDiagnostics GetDiagnostics()
    {
        NekoMuonParameterState[] states = _state.ParameterStates;
        if (states.Length == 0)
        {
            return new NekoMuonDiagnostics(
                _state.Step,
                0f,
                0f,
                0f,
                0f,
                _state.Options.MaxNewtonSchulzSteps);
        }

        if (_cudaStateAuthorityDevice is int primaryDevice
            && _cudaConfidenceBatches.TryGetValue(
                primaryDevice,
                out CudaOptimizerKernels.NekoMuonConfidenceBatch?
                    confidenceBatch))
        {
            bool runDeviceNewtonSchulz =
                _state.Step % _state.Options.NewtonSchulzInterval == 0;
            (float deviceMinimum, float deviceMean, float deviceMaximum,
                float deviceMeanDepth) =
                confidenceBatch.Read(
                    _state.Options.MaxNewtonSchulzSteps,
                    _state.Options.NewtonSchulzDepthMode,
                    _state.Options.NewtonSchulzDepth,
                    runDeviceNewtonSchulz,
                    ForceFullNewtonSchulz);
            return new NekoMuonDiagnostics(
                _state.Step,
                deviceMinimum,
                deviceMean,
                deviceMaximum,
                deviceMeanDepth,
                _state.Options.MaxNewtonSchulzSteps);
        }

        float minimum = float.PositiveInfinity;
        float maximum = float.NegativeInfinity;
        double sum = 0d;
        double depthSum = 0d;
        bool runNewtonSchulz =
            _state.Step % _state.Options.NewtonSchulzInterval == 0;
        foreach (NekoMuonParameterState state in states)
        {
            minimum = MathF.Min(minimum, state.Confidence);
            maximum = MathF.Max(maximum, state.Confidence);
            sum += state.Confidence;
            depthSum += ForceFullNewtonSchulz && runNewtonSchulz
                ? _state.Options.MaxNewtonSchulzSteps
                : ResolveNewtonSchulzDepth(
                    _state.Options,
                    state.Confidence,
                    runNewtonSchulz);
        }
        float mean = (float)(sum / states.Length);
        return new NekoMuonDiagnostics(
            _state.Step,
            minimum,
            mean,
            maximum,
            (float)(depthSum / states.Length),
            _state.Options.MaxNewtonSchulzSteps);
    }

    public bool TryGetMix8QuantizationDiagnostics(
        out Mix8QuantizationDiagnostics diagnostics)
    {
        CudaMix8QuantizationDiagnostics? accumulator =
            _cudaMix8Diagnostics;
        int deviceIndex = _cudaMix8DiagnosticsDevice;
        if (accumulator is null || deviceIndex < 0)
        {
            diagnostics = default;
            return false;
        }
        return accumulator.TryRead(deviceIndex, out diagnostics);
    }

    private static void AddProfileTicks(
        long[]? profileTicks,
        int index,
        long start)
    {
        if (profileTicks is null)
            return;
        Interlocked.Add(
            ref profileTicks[index],
            Stopwatch.GetTimestamp() - start);
    }

    private static double TicksToMilliseconds(long ticks)
        => ticks * 1000d / Stopwatch.Frequency;

}

public readonly record struct NekoMuonDiagnostics(
    int Step,
    float MinimumConfidence,
    float MeanConfidence,
    float MaximumConfidence,
    float MeanNewtonSchulzDepth,
    int MaximumNewtonSchulzDepth);

public readonly record struct NekoMuonStepProfile(
    double UpdateMomentsMilliseconds,
    double ConfidenceMilliseconds,
    double InitializeMilliseconds,
    double NewtonSchulzMilliseconds,
    double TransposeMilliseconds,
    double ApplyUpdateMilliseconds,
    double FirstGramMilliseconds,
    double GramSquaredMilliseconds,
    double PolynomialMilliseconds)
{
    public double TotalCpuMilliseconds
        => UpdateMomentsMilliseconds
            + ConfidenceMilliseconds
            + InitializeMilliseconds
            + NewtonSchulzMilliseconds
            + TransposeMilliseconds
            + ApplyUpdateMilliseconds;
}
