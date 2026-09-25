using System.Diagnostics;
using System.Reflection;
using System.Runtime.ExceptionServices;
using System.Security.Cryptography;
using System.Text.Json;
using NNtrain.Arc;
using NNtrain.Runtime.Execution;

namespace NNtrain.Benchmarks;

/// <summary>
/// Compare the same teacher-forced prefixes before sampling can cause two
/// autoregressive continuations to diverge. Both routes project one last-token
/// hidden row through the real language-model head.
/// </summary>
internal static class ArcGenerationLogitProbe
{
    private static readonly MethodInfo FullHidden = typeof(GptRinWikiJp).GetMethod("ForwardHidden",
        BindingFlags.Instance | BindingFlags.NonPublic, null,
        [typeof(int[]), typeof(int), typeof(int)], null)
        ?? throw new MissingMethodException("GptRinWikiJp.ForwardHidden");
    private static readonly MethodInfo CachedHidden = typeof(GptRinWikiJp).GetMethod("ForwardHiddenArcCached",
        BindingFlags.Instance | BindingFlags.NonPublic, null,
        [typeof(int[]), typeof(int), typeof(ArcAttentionKvCache[])], null)
        ?? throw new MissingMethodException("GptRinWikiJp.ForwardHiddenArcCached");
    private static readonly FieldInfo Head = typeof(GptRinWikiJp).GetField("_languageModelHead",
        BindingFlags.Instance | BindingFlags.NonPublic)
        ?? throw new MissingFieldException("GptRinWikiJp._languageModelHead");
    private static readonly MethodInfo RawHeadLinear = typeof(Tensor).GetMethod("ArcLinear",
        BindingFlags.Instance | BindingFlags.NonPublic, null,
        [typeof(Tensor), typeof(Tensor), typeof(bool), typeof(TensorDType?)], null)
        ?? throw new MissingMethodException("Tensor.ArcLinear");

    internal static void Run(string[] args)
    {
        if (args.Length is < 3 or > 4 || args.Length == 4 && args[3] != "--include-logits")
            throw new ArgumentException("Expected training config, baseline generation JSON, NEW report.json, "
                + "and optional --include-logits.");
        string configPath = Path.GetFullPath(args[0]);
        string baselinePath = Path.GetFullPath(args[1]);
        string outputPath = Path.GetFullPath(args[2]);
        if (File.Exists(outputPath) || string.Equals(outputPath, configPath, StringComparison.OrdinalIgnoreCase)
            || string.Equals(outputPath, baselinePath, StringComparison.OrdinalIgnoreCase))
            throw new IOException("Logit probe output must be a new artifact.");
        bool includeLogits = args.Length == 4;
        using JsonDocument baselineDocument = JsonDocument.Parse(File.ReadAllText(baselinePath));
        JsonElement baseline = baselineDocument.RootElement;
        if (baseline.GetProperty("SchemaVersion").GetInt32() != 1
            || baseline.GetProperty("ModelSource").GetString() != "fixed-seed synthetic weights")
            throw new NotSupportedException("This diagnostic requires a fixed-seed synthetic generation-probe baseline.");
        if (baseline.GetProperty("ConfigurationSha256").GetString() != HashFile(configPath))
            throw new InvalidDataException("The supplied configuration differs from the baseline configuration.");
        if (baseline.GetProperty("Result").GetProperty("SelectedMode").GetString() != "single")
            throw new InvalidDataException("The source baseline must use single-GPU generation.");
        ArcExecutionOptions baselineOptions = baseline.GetProperty("ArcOptions").Deserialize<ArcExecutionOptions>()
            ?? throw new InvalidDataException("Missing baseline Arc options.");
        if (baselineOptions.InferenceKvCache || baselineOptions.InferenceGemv || baselineOptions.InferenceFusedGemv
            || baselineOptions.InferencePackedEmbedding || baselineOptions.InferenceSmallRowNorm)
            throw new InvalidDataException("The source baseline must disable all five generation optimization flags.");
        baselineOptions = baselineOptions with { DetailedProfiling = false };
        ArcExecutionOptions candidateOptions = baselineOptions with
        {
            InferenceKvCache = true,
            InferenceGemv = true,
            InferenceFusedGemv = true,
            InferencePackedEmbedding = true,
            InferenceSmallRowNorm = true,
        };
        WikiTrainingConfiguration original = WikiTrainingConfiguration.Load(configPath);
        JsonElement shape = baseline.GetProperty("Shape");
        if (!string.Equals(original.ModelArchitecture, "transformer", StringComparison.OrdinalIgnoreCase)
            || baseline.GetProperty("Seed").GetInt32() != original.Seed
            || shape.GetProperty("BatchSize").GetInt32() != 1
            || shape.GetProperty("ContextLength").GetInt32() != original.ContextLength
            || shape.GetProperty("ModelWidth").GetInt32() != original.ModelWidth
            || shape.GetProperty("Heads").GetInt32() != original.Heads
            || shape.GetProperty("HiddenSize").GetInt32() != original.HiddenSize
            || shape.GetProperty("VocabularySize").GetInt32() != original.VocabularySize
            || shape.GetProperty("TieWordEmbeddings").GetBoolean() != original.TieWordEmbeddings
            || baseline.GetProperty("Precision").GetProperty("Mode").GetString()
                != TensorPrecisionModeNames.Format(original.GetPrecisionMode())
            || baseline.GetProperty("Precision").GetProperty("Bfp8BlockSize").GetInt32() != original.Bfp8BlockSize)
            throw new InvalidDataException("The diagnostic model/seed/precision does not match the source baseline.");
        int layers = shape.GetProperty("Layers").GetInt32();
        if (layers <= 0 || layers > original.Layers)
            throw new InvalidDataException("The baseline layer override is outside the configured model.");
        WikiTrainingConfiguration config = original with { Device = "arc", Layers = layers, ArcInferenceMode = "single" };
        int[] prompt = baseline.GetProperty("PromptTokenIds").EnumerateArray().Select(token => token.GetInt32()).ToArray();
        int[] teacher = baseline.GetProperty("Result").GetProperty("Samples")[0]
            .GetProperty("GeneratedTokenIds").EnumerateArray().Select(token => token.GetInt32()).ToArray();
        int generatedTokens = baseline.GetProperty("MaxNewTokens").GetInt32();
        if (prompt.Length == 0 || generatedTokens <= 0 || generatedTokens > 4096
            || teacher.Length != prompt.Length + generatedTokens
            || !teacher.Take(prompt.Length).SequenceEqual(prompt)
            || teacher.Any(token => (uint)token >= (uint)config.VocabularySize))
            throw new InvalidDataException("The baseline prompt or generated token IDs are invalid.");
        if ((long)prompt.Length + generatedTokens - 1 > config.ContextLength)
            throw new NotSupportedException("This bounded logit diagnostic requires teacher-forced prefixes to fit within the context window.");
        int[] steps = new[] { 0, 1, 2, 7, 31, 99, 199 }.Where(step => step < generatedTokens).ToArray();
        int primary = ArcInferenceRouting.Resolve(config).DeviceIndices[0];
        double temperature = baseline.GetProperty("Sampling").GetProperty("Temperature").GetDouble();
        var binaryHashes = new[] { typeof(Tensor).Assembly, typeof(ArcExecutionLane).Assembly,
                typeof(ArcGenerationLogitProbe).Assembly, typeof(WikiTrainingConfiguration).Assembly,
                typeof(ExecutionSession).Assembly }
            .Distinct().ToDictionary(assembly => assembly.GetName().Name!, assembly => HashFile(assembly.Location));
        bool oldSimd = Tensor.SimdEnabled;
        int oldWorkers = Tensor.MaxDegreeOfParallelism;
        RouteResult reference, candidate;
        try
        {
            Tensor.SimdEnabled = config.UseSimd;
            Tensor.MaxDegreeOfParallelism = config.MaxDegreeOfParallelism;
            reference = Capture(config, primary, baselineOptions, teacher, prompt.Length, generatedTokens, steps, cached: false);
            candidate = Capture(config, primary, candidateOptions, teacher, prompt.Length, generatedTokens, steps, cached: true);
        }
        finally
        {
            Tensor.SimdEnabled = oldSimd;
            Tensor.MaxDegreeOfParallelism = oldWorkers;
        }
        object[] comparisons = steps.Select(step => Compare(step, prompt.Length + step,
            teacher[prompt.Length + step], reference.Logits[step], candidate.Logits[step], temperature, includeLogits,
            "published")).ToArray();
        object[] beforePublication = steps.Select(step => Compare(step, prompt.Length + step,
            teacher[prompt.Length + step], reference.BeforePublicationLogits[step], candidate.BeforePublicationLogits[step],
            temperature, includeLogits, "before final head rounding")).ToArray();
        var report = new
        {
            SchemaVersion = 1,
            TimestampUtc = DateTimeOffset.UtcNow,
            ConfigurationPath = configPath,
            ConfigurationSha256 = HashFile(configPath),
            BaselineGenerationPath = baselinePath,
            BaselineGenerationSha256 = HashFile(baselinePath),
            BaselineBinarySha256 = baseline.GetProperty("BinarySha256").Clone(),
            BinarySha256 = binaryHashes,
            ModelSource = "fixed-seed synthetic weights",
            config.Seed,
            Shape = shape.Clone(),
            Precision = baseline.GetProperty("Precision").Clone(),
            TeacherForcingSource = "first measured baseline generation",
            PromptTokenIds = prompt,
            TeacherForcedTokenIds = teacher,
            CheckedStepsZeroBased = steps,
            SourceSampling = baseline.GetProperty("Sampling").Clone(),
            BaselineOptions = baselineOptions,
            CandidateOptions = candidateOptions,
            ReferenceRoute = new { reference.DeviceName, reference.DriverVersion, reference.ElapsedMs, reference.ParameterCount,
                reference.KernelLaunches, reference.PeakBackendBytes, reference.AllocatedBeforeCacheRelease, reference.AllocatedAfterCacheRelease },
            CandidateRoute = new { candidate.DeviceName, candidate.DriverVersion, candidate.ElapsedMs, candidate.ParameterCount,
                candidate.KernelLaunches, candidate.PeakBackendBytes, candidate.AllocatedBeforeCacheRelease, candidate.AllocatedAfterCacheRelease },
            Comparisons = comparisons,
            BeforeFinalHeadPublicationComparisons = beforePublication,
            Notes = "No sampling occurs in this diagnostic. Every candidate cache step consumes the exact corresponding "
                + "baseline token, so autoregressive feedback cannot amplify between-route input differences. Reference "
                + "computes a full prefix at each selected step; candidate updates every intervening token's caches. "
                + "Both project only hidden.SelectLastSequenceToken() through the same model head. New-token step zero "
                + "is prompt prefill. A second head projection forces FP32 output to inspect values before the final "
                + "BF16/BFP8 storage publication; all preceding hidden activations keep the production precision. "
                + "Both fresh models use the baseline seed/configuration. The original baseline "
                + "and current binary hashes are recorded separately because adding this probe requires rebuilding "
                + "the benchmark assembly. Top40 probability total variation compares token-ID distributions, while "
                + "Top40 rank positions reveal ordering changes that can map a fixed RNG draw to another token. "
                + "Relative RMS divides error RMS by reference logit RMS. Differences establish local numerical "
                + "agreement, not trained-model quality. Route elapsed times have different work and are not a speed comparison.",
        };
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
        using (var output = new FileStream(outputPath, FileMode.CreateNew, FileAccess.Write, FileShare.Read))
            JsonSerializer.Serialize(output, report, new JsonSerializerOptions { WriteIndented = true });
        Console.WriteLine($"Teacher-forced Arc logit comparison saved: {outputPath}");
    }

    private static RouteResult Capture(WikiTrainingConfiguration config, int device, ArcExecutionOptions options,
        int[] teacher, int promptLength, int generatedTokens, int[] steps, bool cached)
    {
        using IDisposable execution = Tensor.BeginArcExecution(device, config.GetPrecisionMode(), options);
        var model = (GptRinWikiJp)WikiLanguageModelCommand.CreateModel(config, config.VocabularySize);
        model.to(new TorchDevice(TensorDevice.Arc, device));
        model.eval();
        model.ArcTensorParallelEnabled = false;
        if (cached && !model.CanUseArcKvCache())
            throw new NotSupportedException("The selected model is not eligible for Arc KV caching.");
        var head = (Linear)(Head.GetValue(model) ?? throw new InvalidOperationException("Model head is missing."));
        ArcExecutionLane lane = Tensor.ArcLane;
        var owned = new List<ArcAttentionKvCache>();
        var logits = new Dictionary<int, float[]>();
        var beforePublicationLogits = new Dictionary<int, float[]>();
        void CaptureHead(int step, Tensor hidden)
        {
            Tensor last = hidden.SelectLastSequenceToken();
            logits.Add(step, head.ForwardBatch(last).Data.ToArray());
            Tensor raw = InvokeTensor(RawHeadLinear, last, [head.W.T, head.B.T, false, (TensorDType?)TensorDType.Float32]);
            beforePublicationLogits.Add(step, raw.Data.ToArray());
        }
        long start = Stopwatch.GetTimestamp();
        long launches = lane.KernelLaunchCount;
        long allocatedBeforeRelease;
        using IDisposable noGrad = AutogradContext.NoGrad();
        try
        {
            if (cached)
            {
                int capacity = promptLength + generatedTokens - 1;
                for (int layer = 0; layer < config.Layers; ++layer)
                    owned.Add(new ArcAttentionKvCache(config.ModelWidth, config.Heads, capacity));
                ArcAttentionKvCache[] caches = owned.ToArray();
                var selected = steps.ToHashSet();
                for (int step = 0; step < generatedTokens; ++step)
                {
                    using IDisposable? frame = Tensor.BeginArcInferenceFrame();
                    int position = step == 0 ? 0 : promptLength + step - 1;
                    int[] tokens = step == 0 ? teacher[..promptLength] : [teacher[position]];
                    Tensor hidden = InvokeTensor(CachedHidden, model, [tokens, position, caches]);
                    if (selected.Contains(step))
                        CaptureHead(step, hidden);
                }
            }
            else
            {
                foreach (int step in steps)
                {
                    using IDisposable? frame = Tensor.BeginArcInferenceFrame();
                    int length = promptLength + step;
                    Tensor hidden = InvokeTensor(FullHidden, model, [teacher[..length], 1, length]);
                    CaptureHead(step, hidden);
                }
            }
            lane.Synchronize();
            allocatedBeforeRelease = lane.AllocatedBytes;
        }
        finally
        {
            foreach (ArcAttentionKvCache cache in owned) cache.Dispose();
        }
        lane.Synchronize();
        foreach (float[] values in logits.Values.Concat(beforePublicationLogits.Values))
            if (values.Length != config.VocabularySize || values.Any(value => !float.IsFinite(value)))
                throw new ArithmeticException("The diagnostic produced non-finite or invalid-sized logits.");
        Console.WriteLine($"Teacher forcing {(cached ? "cached candidate" : "full-prefix reference")}: "
            + $"{logits.Count} logit rows, {lane.KernelLaunchCount - launches:N0} kernels");
        return new RouteResult(logits, beforePublicationLogits, lane.Device.Name, lane.Device.DriverVersion,
            Stopwatch.GetElapsedTime(start).TotalMilliseconds,
            model.parameters().Sum(parameter => (long)parameter.T.numel()),
            lane.KernelLaunchCount - launches, lane.PeakAllocatedBytes, allocatedBeforeRelease, lane.AllocatedBytes);
    }

    private static Tensor InvokeTensor(MethodInfo method, object target, object[] arguments)
    {
        try { return (Tensor)(method.Invoke(target, arguments) ?? throw new InvalidOperationException("Tensor method returned null.")); }
        catch (TargetInvocationException exception) when (exception.InnerException is not null)
        {
            ExceptionDispatchInfo.Capture(exception.InnerException).Throw();
            throw;
        }
    }

    private static object Compare(int step, int prefixLength, int baselineNextToken,
        float[] reference, float[] candidate, double temperature, bool includeLogits, string stage)
    {
        double sumSquaredError = 0, sumSquaredReference = 0, sumAbsoluteError = 0, maxAbsoluteError = 0;
        int maximumErrorToken = 0;
        for (int token = 0; token < reference.Length; ++token)
        {
            double error = (double)candidate[token] - reference[token];
            double absolute = Math.Abs(error);
            sumSquaredError += error * error;
            sumSquaredReference += (double)reference[token] * reference[token];
            sumAbsoluteError += absolute;
            if (absolute > maxAbsoluteError) { maxAbsoluteError = absolute; maximumErrorToken = token; }
        }
        int[] referenceOrder = Enumerable.Range(0, reference.Length).OrderByDescending(token => reference[token]).ThenBy(token => token).ToArray();
        int[] candidateOrder = Enumerable.Range(0, candidate.Length).OrderByDescending(token => candidate[token]).ThenBy(token => token).ToArray();
        int count = Math.Min(40, reference.Length);
        int[] referenceTop = referenceOrder[..count], candidateTop = candidateOrder[..count];
        int overlap = referenceTop.Intersect(candidateTop).Count();
        double? totalVariation = null;
        if (temperature > 0 && double.IsFinite(temperature))
        {
            Dictionary<int, double> Distribution(float[] values, int[] indices)
            {
                var probabilities = indices.ToDictionary(token => token,
                    token => Math.Exp(((double)values[token] - values[indices[0]]) / temperature));
                double sum = probabilities.Values.Sum();
                return probabilities.ToDictionary(pair => pair.Key, pair => pair.Value / sum);
            }
            Dictionary<int, double> left = Distribution(reference, referenceTop), right = Distribution(candidate, candidateTop);
            totalVariation = left.Keys.Union(right.Keys).Sum(token => Math.Abs(left.GetValueOrDefault(token) - right.GetValueOrDefault(token))) * .5;
        }
        double rms = Math.Sqrt(sumSquaredError / reference.Length);
        double? relativeRms = sumSquaredReference > 0 ? Math.Sqrt(sumSquaredError / sumSquaredReference)
            : sumSquaredError == 0 ? 0 : null;
        Console.WriteLine($"  {stage} step={step}, prefix={prefixLength}: maxAbs={maxAbsoluteError:G5}, RMS={rms:G5}, "
            + $"relativeRMS={relativeRms:G5}, argmax={referenceOrder[0] == candidateOrder[0]}, top40={overlap}/{count}");
        return new
        {
            StepZeroBased = step,
            PrefixLength = prefixLength,
            BaselineSampledNextToken = baselineNextToken,
            MaximumAbsoluteError = maxAbsoluteError,
            MaximumErrorToken = maximumErrorToken,
            ReferenceAtMaximumError = reference[maximumErrorToken],
            CandidateAtMaximumError = candidate[maximumErrorToken],
            MeanAbsoluteError = sumAbsoluteError / reference.Length,
            RmsError = rms,
            ReferenceRms = Math.Sqrt(sumSquaredReference / reference.Length),
            RelativeRmsError = relativeRms,
            ReferenceArgmax = referenceOrder[0],
            CandidateArgmax = candidateOrder[0],
            ArgmaxMatches = referenceOrder[0] == candidateOrder[0],
            Top40SetOverlapCount = overlap,
            Top40SetOverlapFraction = (double)overlap / count,
            Top40RankPositionsWithDifferentTokens = referenceTop.Zip(candidateTop).Count(pair => pair.First != pair.Second),
            Top40DistributionTotalVariation = totalVariation,
            Top40DistributionTemperature = temperature > 0 ? temperature : (double?)null,
            BaselineNextTokenReferenceRank = Array.IndexOf(referenceOrder, baselineNextToken) + 1,
            BaselineNextTokenCandidateRank = Array.IndexOf(candidateOrder, baselineNextToken) + 1,
            ReferenceTop40TokenIds = referenceTop,
            CandidateTop40TokenIds = candidateTop,
            ReferenceTop40Logits = referenceTop.Select(token => reference[token]).ToArray(),
            CandidateTop40Logits = candidateTop.Select(token => candidate[token]).ToArray(),
            ReferenceLogits = includeLogits ? reference : null,
            CandidateLogits = includeLogits ? candidate : null,
        };
    }

    private static string HashFile(string path)
    {
        using FileStream input = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(input));
    }

    private sealed record RouteResult(Dictionary<int, float[]> Logits, Dictionary<int, float[]> BeforePublicationLogits,
        string DeviceName, string? DriverVersion,
        double ElapsedMs, long ParameterCount, long KernelLaunches, long PeakBackendBytes,
        long AllocatedBeforeCacheRelease, long AllocatedAfterCacheRelease);
}
