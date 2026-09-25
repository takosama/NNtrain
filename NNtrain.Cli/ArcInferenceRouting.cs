using System.Diagnostics;
using NNtrain.Arc;
using NNtrain.Runtime.Execution;

namespace NNtrain;

internal sealed record ArcInferenceSettings(int[] DeviceIndices, string Mode)
{
    internal bool UsesTwoDevices => DeviceIndices.Length == 2
        && !string.Equals(Mode, ArcInferenceRouting.SingleMode, StringComparison.OrdinalIgnoreCase);
}

internal static class ArcInferenceRouting
{
    internal const string AutoMode = "auto";
    internal const string SingleMode = "single";
    internal const string TensorParallelMode = "tensorParallel";
    private const int TimedRuns = 3;
    private const double RequiredSpeedup = 0.05d;

    internal static void ValidateOptions(int[]? indices, string? mode)
    {
        if (indices is { Length: < 1 or > 2 }
            || indices is not null
                && (indices.Any(index => index < 0)
                    || indices.Distinct().Count() != indices.Length))
        {
            throw new ArgumentException(
                "inferenceDeviceIndices must contain one or two unique, non-negative Arc indices.",
                nameof(indices));
        }
        if (mode is not null
            && !string.Equals(mode, AutoMode, StringComparison.OrdinalIgnoreCase)
            && !string.Equals(mode, SingleMode, StringComparison.OrdinalIgnoreCase)
            && !string.Equals(mode, TensorParallelMode, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException(
                "arcInferenceMode must be 'auto', 'single', or 'tensorParallel'.",
                nameof(mode));
        }
    }

    internal static ArcInferenceSettings Resolve(
        WikiTrainingConfiguration modelConfiguration,
        GenerationConfiguration? generation = null)
    {
        ArgumentNullException.ThrowIfNull(modelConfiguration);
        ValidateOptions(modelConfiguration.InferenceDeviceIndices, modelConfiguration.ArcInferenceMode);
        if (generation is not null)
            ValidateOptions(generation.InferenceDeviceIndices, generation.ArcInferenceMode);
        int[] indices = (generation?.InferenceDeviceIndices
            ?? modelConfiguration.InferenceDeviceIndices
            ?? modelConfiguration.DeviceIndices
            ?? [modelConfiguration.DeviceIndex]).ToArray();
        string mode = generation?.ArcInferenceMode
            ?? modelConfiguration.ArcInferenceMode;
        ValidateOptions(indices, mode);
        if (string.Equals(mode, TensorParallelMode, StringComparison.OrdinalIgnoreCase)
            && indices.Length != 2)
        {
            throw new ArgumentException(
                "arcInferenceMode 'tensorParallel' requires two inferenceDeviceIndices.");
        }
        return new(indices, mode.ToLowerInvariant());
    }

    internal static IDisposable BeginExecution(
        ArcInferenceSettings settings,
        TensorPrecisionMode precision)
        => settings.UsesTwoDevices
            ? Tensor.BeginArcInferenceExecution(settings.DeviceIndices, precision)
            : Tensor.BeginArcExecution(settings.DeviceIndices[0], precision);

    internal static void ConfigureModel(
        LanguageModel model,
        BpeTokenizer tokenizer,
        string prompt,
        int maxNewTokens,
        ArcInferenceSettings settings,
        TextWriter output)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(output);
        if (model is not GptRinWikiJp transformer)
        {
            if (settings.UsesTwoDevices)
                throw new NotSupportedException("Two-device Arc inference requires a Transformer checkpoint.");
            output.WriteLine($"Arc inference = single GPU [{settings.DeviceIndices[0]}]");
            return;
        }

        ConfigureModelTokens(transformer, tokenizer.Encode(prompt, addBos: true),
            maxNewTokens, settings, output);
    }

    internal static void ConfigureModelTokens(
        GptRinWikiJp transformer,
        int[] promptTokens,
        int maxNewTokens,
        ArcInferenceSettings settings,
        TextWriter output)
    {
        ArgumentNullException.ThrowIfNull(transformer);
        ArgumentNullException.ThrowIfNull(promptTokens);
        ArgumentNullException.ThrowIfNull(settings);
        ArgumentNullException.ThrowIfNull(output);
        ArgumentOutOfRangeException.ThrowIfNegative(maxNewTokens);
        if (promptTokens.Length == 0
            || promptTokens.Any(token => (uint)token >= (uint)transformer.VocabularySize))
            throw new ArgumentException("The prompt must contain valid model token IDs.", nameof(promptTokens));
        ValidateOptions(settings.DeviceIndices, settings.Mode);
        if (string.Equals(settings.Mode, TensorParallelMode, StringComparison.OrdinalIgnoreCase)
            && settings.DeviceIndices.Length != 2)
            throw new ArgumentException("arcInferenceMode 'tensorParallel' requires two inferenceDeviceIndices.", nameof(settings));

        bool useTensorParallel = false;
        if (settings.UsesTwoDevices)
        {
            bool forced = string.Equals(settings.Mode, TensorParallelMode, StringComparison.OrdinalIgnoreCase);
            if (!forced && maxNewTokens == 0)
            {
                output.WriteLine("Arc inference auto: no new tokens requested; using single GPU without calibration");
            }
            else if (!transformer.CanUseArcTensorParallel(out string reason))
            {
                if (forced)
                    throw new NotSupportedException(reason);
                output.WriteLine(
                    $"Arc inference auto: tensorParallel unavailable ({reason}); using single GPU");
            }
            else if (forced)
            {
                useTensorParallel = true;
            }
            else
            {
                // Keep each complete route together. Its first warmup builds
                // shards and resident matrices outside the three timed runs.
                var (singlePlan, single) = CalibrateRoute(transformer, promptTokens, maxNewTokens, false);
                var (parallelPlan, parallel) = CalibrateRoute(transformer, promptTokens, maxNewTokens, true);
                useTensorParallel = ShouldUseTensorParallel(single.EstimatedTotalMs, parallel.EstimatedTotalMs);
                output.WriteLine($"Arc inference auto: prompt {promptTokens.Length}, {maxNewTokens} new tokens; "
                    + $"warmup + {TimedRuns} runs per route; require {RequiredSpeedup:P0} speedup");
                WriteCalibration(output, "single", singlePlan, single);
                WriteCalibration(output, "tensorParallel", parallelPlan, parallel);
            }
        }
        transformer.ArcTensorParallelEnabled = useTensorParallel;
        output.WriteLine(useTensorParallel
            ? $"Arc inference = tensorParallel GPUs [{string.Join(',', settings.DeviceIndices)}]"
            : $"Arc inference = single GPU [{settings.DeviceIndices[0]}]");
    }

    internal static bool ShouldUseTensorParallel(double singleMilliseconds, double parallelMilliseconds)
    {
        if (!double.IsFinite(singleMilliseconds) || singleMilliseconds <= 0
            || !double.IsFinite(parallelMilliseconds) || parallelMilliseconds <= 0)
            throw new ArgumentOutOfRangeException(nameof(singleMilliseconds));
        return parallelMilliseconds <= singleMilliseconds * (1d - RequiredSpeedup);
    }

    internal static void WritePeakAllocations(TextWriter output)
    {
        ExecutionSession? session = ExecutionSession.Current;
        if (session?.Options.Device != ExecutionDeviceKind.Arc) return;
        ArcExecutionLane[] lanes = session.Lanes.OfType<ArcExecutionLane>()
            .OrderBy(lane => lane.DeviceIndex).ToArray();
        output.WriteLine("Arc backend peak allocated = " + string.Join(", ",
            lanes.Select(lane => $"arc:{lane.DeviceIndex} {lane.PeakAllocatedBytes:N0} bytes")));
    }

    /// <summary>
    /// A cached call prefills once, decodes until the absolute position reaches
    /// the context limit, then recomputes complete sliding windows. Keeping the
    /// count arithmetic separate makes boundary estimates testable without GPU.
    /// </summary>
    internal static CalibrationPlan CreateCalibrationPlan(
        int promptLength, int maxNewTokens, int contextLength, bool kvCacheEnabled)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(promptLength);
        ArgumentOutOfRangeException.ThrowIfNegative(maxNewTokens);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(contextLength);
        bool cached = kvCacheEnabled && contextLength <= 8192 && promptLength <= contextLength;
        int decodeTokens = cached && maxNewTokens > 0
            ? Math.Min(maxNewTokens - 1, contextLength - promptLength) : 0;
        int samples = Math.Min(8, decodeTokens);
        int middlePrompt = cached
            ? promptLength + Math.Max(0, (decodeTokens - samples) / 2)
            : (int)Math.Max(promptLength, Math.Min(contextLength, (long)promptLength + maxNewTokens / 2L));
        return new CalibrationPlan(maxNewTokens, contextLength, cached, decodeTokens,
            cached && maxNewTokens > 0 ? maxNewTokens - 1 - decodeTokens : 0,
            middlePrompt, samples == 0 ? 0 : samples + 1);
    }

    internal static double EstimateGenerationMilliseconds(CalibrationPlan plan,
        double prefillMs, double decodeTokenMs, double fullWindowTokenMs)
    {
        ArgumentNullException.ThrowIfNull(plan);
        if (!double.IsFinite(prefillMs) || prefillMs < 0
            || !double.IsFinite(decodeTokenMs) || decodeTokenMs < 0
            || !double.IsFinite(fullWindowTokenMs) || fullWindowTokenMs < 0)
            throw new ArgumentOutOfRangeException(nameof(prefillMs), "Calibration timings must be finite and non-negative.");
        if (plan.MaxNewTokens == 0) return 0d;
        return plan.UseKvCache
            ? prefillMs + plan.CachedDecodeTokens * decodeTokenMs + plan.FullWindowTokens * fullWindowTokenMs
            : plan.MaxNewTokens * decodeTokenMs;
    }

    private static (CalibrationPlan Plan, CalibrationResult Result) CalibrateRoute(GptRinWikiJp model,
        int[] promptTokens, int maxNewTokens, bool tensorParallel)
    {
        model.ArcTensorParallelEnabled = tensorParallel;
        // Shard widths may reject caching only on the tensor-parallel route.
        // Derive the plan after switching so calibration matches generation.
        CalibrationPlan plan = CreateCalibrationPlan(promptTokens.Length, maxNewTokens,
            model.ContextLength, model.CanUseArcKvCache());
        int[] representative = ResizePrompt(promptTokens, plan.RepresentativePromptLength);
        int[]? fullWindow = plan.FullWindowTokens > 0
            // One extra input token forces the same full-window fallback used
            // after the original call fills its cache; a context-length input
            // alone would incorrectly include construction of a fresh cache.
            ? ResizePrompt(promptTokens, checked(plan.ContextLength + 1)) : null;
        CalibrationResult Sample()
        {
            double prefill = 0, decode = 0, full = 0;
            if (!plan.UseKvCache)
                decode = TimeOneToken(model, representative);
            else
            {
                prefill = TimeOneToken(model, promptTokens);
                if (plan.DecodeSampleNewTokens > 0)
                {
                    double first = 0, last = 0;
                    int callbacks = 0;
                    long started = Stopwatch.GetTimestamp();
                    _ = model.generate_token_ids(representative, plan.DecodeSampleNewTokens,
                        0f, 1, null, new Random(0), _ => {
                            last = Stopwatch.GetElapsedTime(started).TotalMilliseconds;
                            if (callbacks++ == 0) first = last;
                        });
                    if (callbacks != plan.DecodeSampleNewTokens)
                        throw new InvalidOperationException("Arc calibration requires one streaming callback per generated token.");
                    decode = (last - first) / (callbacks - 1);
                }
                if (fullWindow is not null) full = TimeOneToken(model, fullWindow);
            }
            return new CalibrationResult(prefill, decode, full,
                EstimateGenerationMilliseconds(plan, prefill, decode, full));
        }
        _ = Sample();
        CalibrationResult[] samples = Enumerable.Range(0, TimedRuns).Select(_ => Sample()).ToArray();
        return (plan, new CalibrationResult(Median(samples.Select(sample => sample.PrefillMs).ToArray()),
            Median(samples.Select(sample => sample.DecodeTokenMs).ToArray()),
            Median(samples.Select(sample => sample.FullWindowTokenMs).ToArray()),
            Median(samples.Select(sample => sample.EstimatedTotalMs).ToArray())));
    }

    private static void WriteCalibration(TextWriter output, string route,
        CalibrationPlan plan, CalibrationResult measured)
    {
        string phases = plan.UseKvCache
            ? $"prefill {measured.PrefillMs:F2} ms, cached decode {measured.DecodeTokenMs:F2} ms/token "
                + $"({plan.CachedDecodeTokens} tokens)"
                + (plan.FullWindowTokens > 0 ? $", full-window {measured.FullWindowTokenMs:F2} ms/token ({plan.FullWindowTokens} tokens)" : "")
            : $"full-prefix {measured.DecodeTokenMs:F2} ms/token "
                + $"(representative context {Math.Min(plan.ContextLength, plan.RepresentativePromptLength)})";
        output.WriteLine($"Arc inference auto: {route}: {phases}, estimated generation {measured.EstimatedTotalMs:F1} ms");
    }

    private static int[] ResizePrompt(int[] promptTokens, int targetLength)
    {
        if (targetLength <= promptTokens.Length)
            return promptTokens.TakeLast(targetLength).ToArray();
        int[] tokens = new int[targetLength];
        promptTokens.CopyTo(tokens, 0);
        Array.Fill(tokens, promptTokens[^1], promptTokens.Length, targetLength - promptTokens.Length);
        return tokens;
    }

    private static double TimeOneToken(GptRinWikiJp model, int[] tokens)
    {
        long start = Stopwatch.GetTimestamp();
        _ = model.generate_token_ids(tokens, 1, 0f, 1, null, new Random(0));
        return Stopwatch.GetElapsedTime(start).TotalMilliseconds;
    }

    private static double Median(double[] values)
    {
        Array.Sort(values);
        return values[values.Length / 2];
    }

    internal sealed record CalibrationPlan(int MaxNewTokens, int ContextLength, bool UseKvCache,
        int CachedDecodeTokens, int FullWindowTokens, int RepresentativePromptLength, int DecodeSampleNewTokens);

    private sealed record CalibrationResult(double PrefillMs, double DecodeTokenMs,
        double FullWindowTokenMs, double EstimatedTotalMs);
}
