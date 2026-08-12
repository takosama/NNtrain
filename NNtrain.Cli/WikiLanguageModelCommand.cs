using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace NNtrain;

internal static class WikiLanguageModelCommand
{
    private const int CheckpointFormatVersion = 2;

    private static readonly JsonSerializerOptions CheckpointJsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
    };

    internal static int Run(
        string configurationPath,
        string? generatePrompt,
        TextWriter output,
        TextWriter error,
        bool openLossGraph = false)
    {
        try
        {
            WikiTrainingConfiguration config =
                WikiTrainingConfiguration.Load(configurationPath);
            output.WriteLine(
                $"configuration = {Path.GetFullPath(configurationPath)}");
            Tensor.SimdEnabled = config.UseSimd;
            Tensor.MaxDegreeOfParallelism =
                config.MaxDegreeOfParallelism;
            output.WriteLine(
                $"simd = {(config.UseSimd ? "enabled" : "disabled")}, " +
                $"Vector256 hardware = " +
                $"{(Tensor.IsSimdHardwareAccelerated ? "available" : "unavailable")}");
            output.WriteLine(
                $"thread parallelism = Parallel.For, workers = " +
                $"{Tensor.EffectiveMaxDegreeOfParallelism}" +
                (config.MaxDegreeOfParallelism == 0 ? " (automatic)" : ""));
            if (generatePrompt is not null)
                return GenerateOnly(config, generatePrompt, output);

            return Train(
                config,
                Path.ChangeExtension(
                    Path.GetFullPath(configurationPath),
                    ".loss.html"),
                output,
                error,
                openLossGraph);
        }
        catch (Exception exception) when (
            exception is not OutOfMemoryException
            and not StackOverflowException
            and not OperationCanceledException)
        {
            error.WriteLine($"Error: {exception.Message}");
            return 2;
        }
    }

    private static int Train(
        WikiTrainingConfiguration config,
        string lossGraphPath,
        TextWriter output,
        TextWriter error,
        bool openLossGraph)
    {
        WriteEffectiveTrainingConfiguration(config, output);
        if (!Directory.Exists(config.DataPath))
        {
            throw new DirectoryNotFoundException(
                $"Wikipedia data directory was not found at " +
                $"'{config.DataPath}'.");
        }

        BpeTokenizer? tokenizer = null;
        if (File.Exists(config.TokenizerPath))
        {
            BpeTokenizer existing = BpeTokenizer.Load(config.TokenizerPath);
            if (existing.VocabularySize == config.VocabularySize)
            {
                tokenizer = existing;
                output.WriteLine($"tokenizer = loaded {config.TokenizerPath}");
            }
            else
            {
                output.WriteLine(
                    $"tokenizer vocabulary {existing.VocabularySize} does " +
                    $"not match configured {config.VocabularySize}; " +
                    "retraining tokenizer");
            }
        }
        if (tokenizer is null)
        {
            output.WriteLine(
                $"training BPE tokenizer: target vocabulary " +
                $"{config.VocabularySize}, up to " +
                $"{config.TokenizerTrainingDocuments} documents / " +
                $"{config.TokenizerTrainingBytes} bytes");
            var timer = Stopwatch.StartNew();
            tokenizer = BpeTokenizer.Train(
                ReadDocuments(
                    config.DataPath,
                    config.TextColumn,
                    config.TokenizerTrainingDocuments),
                config.VocabularySize,
                config.TokenizerTrainingBytes);
            tokenizer.Save(config.TokenizerPath);
            timer.Stop();
            output.WriteLine(
                $"tokenizer = saved {config.TokenizerPath}, " +
                $"vocabulary {tokenizer.VocabularySize}, " +
                $"{timer.Elapsed.TotalSeconds:F2} sec");
        }

        if (config.MaxTrainingTokens == 0)
        {
            return TrainAllData(
                config,
                tokenizer,
                lossGraphPath,
                output,
                error,
                openLossGraph);
        }

        output.WriteLine("loading and tokenizing Wikipedia documents...");
        TrainingCorpus corpus = LoadTrainingCorpus(config, tokenizer, output);
        int[] tokens = corpus.Tokens;
        int sequenceCount = DivideRoundUp(
            tokens.Length - 1,
            config.ContextLength);
        if (sequenceCount < 2)
        {
            throw new InvalidDataException(
                "The selected Wikipedia data does not contain two complete " +
                "training sequences.");
        }

        int validationSequences = config.ValidationFraction == 0f
            ? 0
            : Math.Max(
                1,
                (int)MathF.Floor(sequenceCount * config.ValidationFraction));
        validationSequences = Math.Min(validationSequences, sequenceCount - 1);
        int trainingSequences = sequenceCount - validationSequences;
        output.WriteLine(
            $"tokens = {tokens.Length:N0}, sequences = " +
            $"{trainingSequences:N0} train + " +
            $"{validationSequences:N0} validation, context " +
            $"{config.ContextLength}");
        output.WriteLine(
            $"dataset continuation = every " +
            $"{config.DatasetSampleEverySteps:N0} steps, sample pool " +
            $"{corpus.SampleDocuments.Length}");

        var model = CreateModel(config, tokenizer.VocabularySize);
        IOptimizer optimizer = CreateOptimizer(model, config);
        output.WriteLine(
            $"model = {model.GetType().Name}, parameters " +
            $"{model.Parameters().Sum(parameter => (long)parameter.T.Numel):N0}, " +
            $"width {config.ModelWidth}, heads {config.Heads}, " +
            $"hidden {config.HiddenSize}, layers {config.Layers}, " +
            $"context {config.ContextLength}, batch {config.BatchSize}");
        WriteOptimizerSummary(model, config, output);

        LossGraph? lossGraph = null;
        if (config.ShowLossGraph)
        {
            lossGraph = new LossGraph(lossGraphPath, config.Epochs);
            lossGraph.Write();
            output.WriteLine(
                $"loss graph = {lossGraph.Path}, every " +
                $"{config.GraphUpdateSteps} step(s) or epoch end");
            if (openLossGraph)
                lossGraph.TryOpen(error);
        }

        int[] order = Enumerable.Range(0, trainingSequences).ToArray();
        var random = new Random(config.Seed);
        var sampleRandom = new Random(config.Seed ^ 0x6A09E667);
        int globalStep = 0;
        int batchesPerEpoch = DivideRoundUp(
            trainingSequences,
            config.BatchSize);
        long totalTrainingSteps = checked(
            (long)config.Epochs * batchesPerEpoch);
        ModuleState? bestState = null;
        float bestLoss = float.PositiveInfinity;
        int bestEpoch = 0;

        for (int epoch = 1; epoch <= config.Epochs; epoch++)
        {
            Shuffle(order, random);
            model.Train();
            float totalLoss = 0f;
            int completedTargets = 0;
            float graphWindowLoss = 0f;
            int graphWindowTargets = 0;
            int batchTotal = batchesPerEpoch;
            var timer = Stopwatch.StartNew();

            for (int batch = 0; batch < batchTotal; batch++)
            {
                int count = Math.Min(
                    config.BatchSize,
                    trainingSequences - batch * config.BatchSize);
                LanguageBatch values = CreateBatch(
                    tokens,
                    order,
                    batch * config.BatchSize,
                    count,
                    config.ContextLength);
                optimizer.ZeroGrad();
                Tensor logits = model.Forward(
                    values.Input,
                    count,
                    config.ContextLength);
                Tensor loss = logits.CrossEntropyWithLogits(values.Target);
                loss.Backward();
                SetScheduledLearningRates(
                    optimizer,
                    config,
                    (globalStep + 1d) / totalTrainingSteps);
                optimizer.Step();
                globalStep++;

                int validTargets = values.ValidTargetCount;
                totalLoss += loss.Data[0] * validTargets;
                completedTargets += validTargets;
                graphWindowLoss += loss.Data[0] * validTargets;
                graphWindowTargets += validTargets;
                bool epochEnd = batch + 1 == batchTotal;
                if (lossGraph is not null
                    && (batch + 1) % config.GraphUpdateSteps == 0
                    && !epochEnd)
                {
                    float epochPosition = epoch - 1f
                        + (float)(batch + 1) / batchTotal;
                    lossGraph.AddPoint(
                        epochPosition,
                        graphWindowLoss / graphWindowTargets);
                    lossGraph.Write();
                    graphWindowLoss = 0f;
                    graphWindowTargets = 0;
                }
                if (corpus.SampleDocuments.Length > 0
                    && globalStep % config.DatasetSampleEverySteps == 0)
                {
                    DatasetContinuation sample = CreateDatasetContinuation(
                        model,
                        tokenizer,
                        corpus.SampleDocuments,
                        config,
                        sampleRandom);
                    WriteDatasetContinuation(globalStep, sample, output);
                }
                if ((batch + 1) % config.LogEveryBatches == 0
                    || epochEnd)
                {
                    output.WriteLine(
                        $"epoch {epoch}, batch {batch + 1}/{batchTotal}, " +
                        $"loss = {loss.Data[0]:F6}");
                }
            }

            float trainingLoss = totalLoss / completedTargets;
            float validationLoss = validationSequences == 0
                ? trainingLoss
                : Evaluate(
                    model,
                    tokens,
                    trainingSequences,
                    validationSequences,
                    config);
            timer.Stop();
            output.WriteLine(
                $"epoch {epoch}, train loss = {trainingLoss:F6}, " +
                $"validation loss = {validationLoss:F6}, " +
                $"time = {timer.Elapsed.TotalSeconds:F2} sec");
            if (lossGraph is not null)
            {
                float graphTrainingLoss = graphWindowTargets == 0
                    ? trainingLoss
                    : graphWindowLoss / graphWindowTargets;
                lossGraph.AddPoint(
                    epoch,
                    graphTrainingLoss,
                    validationLoss);
                lossGraph.Write();
            }

            if (bestState is null || validationLoss < bestLoss)
            {
                bestLoss = validationLoss;
                bestEpoch = epoch;
                bestState = model.CaptureState();
                SaveCheckpoint(
                    config.CheckpointPath,
                    new WikiModelCheckpoint(
                        CheckpointFormatVersion,
                        epoch,
                        validationLoss,
                        tokenizer.VocabularySize,
                        config.ContextLength,
                        config.ModelWidth,
                        config.Heads,
                        config.HiddenSize,
                        config.Layers,
                        config.Dropout,
                        config.InitializationScale,
                        bestState,
                        config.ModelArchitecture,
                        config.HyenaFilterWidth,
                        config.ForgetMemoryKeyWidth,
                        config.ForgetMemoryValueWidth,
                        config.ForgetMemoryRetentionMinimum,
                        config.ForgetMemoryRetentionMaximum));
            }
        }

        if (bestState is null)
            throw new InvalidOperationException("Training did not produce a model state.");
        model.RestoreState(bestState);
        output.WriteLine(
            $"best model = epoch {bestEpoch}, validation loss " +
            $"{bestLoss:F6}");
        output.WriteLine($"checkpoint = {config.CheckpointPath}");
        WriteFinalDatasetContinuation(
            model,
            tokenizer,
            corpus.SampleDocuments,
            config,
            sampleRandom,
            globalStep,
            output);
        return 0;
    }

    private static int GenerateOnly(
        WikiTrainingConfiguration config,
        string prompt,
        TextWriter output)
    {
        if (!File.Exists(config.TokenizerPath))
        {
            throw new FileNotFoundException(
                "BPE tokenizer file was not found.",
                config.TokenizerPath);
        }
        if (!File.Exists(config.CheckpointPath))
        {
            throw new FileNotFoundException(
                "Wiki model checkpoint was not found.",
                config.CheckpointPath);
        }

        BpeTokenizer tokenizer = BpeTokenizer.Load(config.TokenizerPath);
        WikiModelCheckpoint checkpoint = LoadCheckpoint(config.CheckpointPath);
        if (checkpoint.VocabularySize != tokenizer.VocabularySize)
        {
            throw new InvalidDataException(
                "Checkpoint and tokenizer vocabulary sizes do not match.");
        }

        IWikiLanguageModel model = CreateModel(checkpoint, config.Seed);
        model.RestoreState(checkpoint.Model);
        output.WriteLine(
            $"checkpoint = epoch {checkpoint.Epoch}, validation loss " +
            $"{checkpoint.ValidationLoss:F6}");
        output.WriteLine(
            $"checkpoint model = {GetCheckpointArchitecture(checkpoint)}, " +
            $"vocabulary {checkpoint.VocabularySize}, " +
            $"width {checkpoint.ModelWidth}, heads {checkpoint.Heads}, " +
            $"hidden {checkpoint.HiddenSize}, layers {checkpoint.Layers}, " +
            $"context {checkpoint.ContextLength}");
        if (!CheckpointArchitectureMatchesConfiguration(checkpoint, config))
        {
            output.WriteLine(
                "note: --generate uses the architecture stored in the " +
                "checkpoint. JSON model settings take effect when a new " +
                "training run starts and its checkpoint is saved.");
        }
        WriteGeneration(model, tokenizer, prompt, config, output);
        return 0;
    }

    private static int TrainAllData(
        WikiTrainingConfiguration config,
        BpeTokenizer tokenizer,
        string lossGraphPath,
        TextWriter output,
        TextWriter error,
        bool openLossGraph)
    {
        long availableDocuments = WikiParquetCorpus.CountRowsAsync(
            config.DataPath).GetAwaiter().GetResult();
        long documentsPerEpoch = config.MaxTrainingDocuments == 0
            ? availableDocuments
            : Math.Min(availableDocuments, config.MaxTrainingDocuments);
        output.WriteLine(
            $"streaming corpus = {documentsPerEpoch:N0} documents/epoch, " +
            $"up to {config.MaxDocumentTokens:N0} tokens/document");

        var model = CreateModel(config, tokenizer.VocabularySize);
        IOptimizer optimizer = CreateOptimizer(model, config);
        output.WriteLine(
            $"model = {model.GetType().Name}, parameters " +
            $"{model.Parameters().Sum(parameter => (long)parameter.T.Numel):N0}, " +
            $"width {config.ModelWidth}, heads {config.Heads}, " +
            $"hidden {config.HiddenSize}, layers {config.Layers}, " +
            $"context {config.ContextLength}, batch {config.BatchSize}");
        WriteOptimizerSummary(model, config, output);

        LossGraph? lossGraph = null;
        if (config.ShowLossGraph)
        {
            lossGraph = new LossGraph(lossGraphPath, config.Epochs);
            lossGraph.Write();
            output.WriteLine(
                $"loss graph = {lossGraph.Path}, every " +
                $"{config.GraphUpdateSteps} step(s) or epoch end");
            if (openLossGraph)
                lossGraph.TryOpen(error);
        }

        var sampleDocuments = new List<string>(config.DatasetSamplePoolSize);
        var reservoirRandom = new Random(
            config.Seed ^ unchecked((int)0xBB67AE85));
        var generationRandom = new Random(config.Seed ^ 0x6A09E667);
        int eligibleSampleDocuments = 0;
        long globalStep = 0;
        ModuleState? bestState = null;
        float bestLoss = float.PositiveInfinity;
        int bestEpoch = 0;

        for (int epoch = 1; epoch <= config.Epochs; epoch++)
        {
            model.Train();
            var buffer = new List<int>(
                config.BatchSize * config.ContextLength
                + config.MaxDocumentTokens
                + 2);
            double totalLoss = 0d;
            long completedTargets = 0;
            double graphWindowLoss = 0d;
            long graphWindowTargets = 0;
            long documentsProcessed = 0;
            var timer = Stopwatch.StartNew();

            void TrainBatch(int batchSize, int sequenceLength)
            {
                LanguageBatch values = CreateStreamingBatch(
                    buffer,
                    batchSize,
                    sequenceLength);
                optimizer.ZeroGrad();
                Tensor logits = model.Forward(
                    values.Input,
                    batchSize,
                    sequenceLength);
                Tensor loss = logits.CrossEntropyWithLogits(values.Target);
                loss.Backward();
                double documentProgress = documentsPerEpoch == 0
                    ? 0d
                    : Math.Min(
                        1d,
                        (double)documentsProcessed / documentsPerEpoch);
                double overallProgress =
                    (epoch - 1d + documentProgress) / config.Epochs;
                SetScheduledLearningRates(
                    optimizer,
                    config,
                    overallProgress);
                optimizer.Step();
                globalStep++;

                long targets = values.ValidTargetCount;
                totalLoss += loss.Data[0] * targets;
                completedTargets += targets;
                graphWindowLoss += loss.Data[0] * targets;
                graphWindowTargets += targets;
                if (lossGraph is not null
                    && globalStep % config.GraphUpdateSteps == 0)
                {
                    float progress = documentsPerEpoch == 0
                        ? 0f
                        : Math.Min(
                            1f,
                            (float)documentsProcessed / documentsPerEpoch);
                    lossGraph.AddPoint(
                        epoch - 1f + progress,
                        (float)(graphWindowLoss / graphWindowTargets));
                    lossGraph.Write();
                    graphWindowLoss = 0d;
                    graphWindowTargets = 0;
                }
                if (sampleDocuments.Count > 0
                    && globalStep % config.DatasetSampleEverySteps == 0)
                {
                    DatasetContinuation sample = CreateDatasetContinuation(
                        model,
                        tokenizer,
                        sampleDocuments,
                        config,
                        generationRandom);
                    WriteDatasetContinuation(globalStep, sample, output);
                }
                if (globalStep % config.LogEveryBatches == 0)
                {
                    output.WriteLine(
                        $"epoch {epoch}, step {globalStep:N0}, " +
                        $"documents {documentsProcessed:N0}/" +
                        $"{documentsPerEpoch:N0}, loss = {loss.Data[0]:F6}");
                }
            }

            int? maximumDocuments = config.MaxTrainingDocuments == 0
                ? null
                : config.MaxTrainingDocuments;
            foreach (string document in ReadDocuments(
                config.DataPath,
                config.TextColumn,
                maximumDocuments))
            {
                documentsProcessed++;
                if (epoch == 1 && TryGetDocumentSplit(document, out _))
                {
                    AddReservoirSample(
                        document,
                        sampleDocuments,
                        ref eligibleSampleDocuments,
                        config.DatasetSamplePoolSize,
                        reservoirRandom);
                }

                buffer.Add(BpeTokenizer.BosTokenId);
                int[] documentTokens = tokenizer.Encode(document);
                int tokenCount = Math.Min(
                    documentTokens.Length,
                    config.MaxDocumentTokens);
                for (int index = 0; index < tokenCount; index++)
                    buffer.Add(documentTokens[index]);
                buffer.Add(BpeTokenizer.EosTokenId);

                while ((buffer.Count - 1) / config.ContextLength
                    >= config.BatchSize)
                {
                    TrainBatch(config.BatchSize, config.ContextLength);
                }
            }

            while (buffer.Count > 1)
            {
                int remainingTargets = buffer.Count - 1;
                int remainingSequences = DivideRoundUp(
                    remainingTargets,
                    config.ContextLength);
                int batchSize = Math.Min(
                    config.BatchSize,
                    remainingSequences);
                TrainBatch(batchSize, config.ContextLength);
            }

            if (completedTargets == 0)
            {
                throw new InvalidDataException(
                    "Wikipedia corpus did not produce trainable token pairs.");
            }
            float trainingLoss = (float)(totalLoss / completedTargets);
            timer.Stop();
            output.WriteLine(
                $"epoch {epoch}, train loss = {trainingLoss:F6}, " +
                $"documents = {documentsProcessed:N0}, targets = " +
                $"{completedTargets:N0}, time = " +
                $"{timer.Elapsed.TotalSeconds:F2} sec");
            if (lossGraph is not null)
            {
                float graphLoss = graphWindowTargets == 0
                    ? trainingLoss
                    : (float)(graphWindowLoss / graphWindowTargets);
                lossGraph.AddPoint(epoch, graphLoss, trainingLoss);
                lossGraph.Write();
            }

            if (bestState is null || trainingLoss < bestLoss)
            {
                bestLoss = trainingLoss;
                bestEpoch = epoch;
                bestState = model.CaptureState();
                SaveCheckpoint(
                    config.CheckpointPath,
                    new WikiModelCheckpoint(
                        CheckpointFormatVersion,
                        epoch,
                        trainingLoss,
                        tokenizer.VocabularySize,
                        config.ContextLength,
                        config.ModelWidth,
                        config.Heads,
                        config.HiddenSize,
                        config.Layers,
                        config.Dropout,
                        config.InitializationScale,
                        bestState,
                        config.ModelArchitecture,
                        config.HyenaFilterWidth,
                        config.ForgetMemoryKeyWidth,
                        config.ForgetMemoryValueWidth,
                        config.ForgetMemoryRetentionMinimum,
                        config.ForgetMemoryRetentionMaximum));
            }
        }

        if (bestState is null)
            throw new InvalidOperationException("Training did not produce a model state.");
        model.RestoreState(bestState);
        output.WriteLine(
            $"best model = epoch {bestEpoch}, train loss {bestLoss:F6}");
        output.WriteLine($"checkpoint = {config.CheckpointPath}");
        WriteFinalDatasetContinuation(
            model,
            tokenizer,
            sampleDocuments,
            config,
            generationRandom,
            globalStep,
            output);
        return 0;
    }

    private static TrainingCorpus LoadTrainingCorpus(
        WikiTrainingConfiguration config,
        BpeTokenizer tokenizer,
        TextWriter output)
    {
        var tokens = new List<int>(config.MaxTrainingTokens);
        var sampleDocuments = new List<string>(config.DatasetSamplePoolSize);
        var sampleRandom = new Random(
            config.Seed ^ unchecked((int)0xBB67AE85));
        int eligibleSampleDocuments = 0;
        int documentCount = 0;
        foreach (string document in ReadDocuments(
            config.DataPath,
            config.TextColumn,
            config.MaxTrainingDocuments == 0
                ? null
                : config.MaxTrainingDocuments))
        {
            if (tokens.Count >= config.MaxTrainingTokens)
                break;
            if (TryGetDocumentSplit(document, out _))
            {
                eligibleSampleDocuments++;
                if (sampleDocuments.Count < config.DatasetSamplePoolSize)
                {
                    sampleDocuments.Add(document);
                }
                else
                {
                    int replacement = sampleRandom.Next(
                        eligibleSampleDocuments);
                    if (replacement < sampleDocuments.Count)
                        sampleDocuments[replacement] = document;
                }
            }

            tokens.Add(BpeTokenizer.BosTokenId);
            int[] documentTokens = tokenizer.Encode(document);
            foreach (int token in documentTokens)
            {
                if (tokens.Count >= config.MaxTrainingTokens - 1)
                    break;
                tokens.Add(token);
            }
            if (tokens.Count < config.MaxTrainingTokens)
                tokens.Add(BpeTokenizer.EosTokenId);
            documentCount++;
            if (documentCount % 100 == 0)
            {
                output.Write(
                    $"\rdocuments {documentCount:N0}, tokens " +
                    $"{tokens.Count:N0}");
            }
        }
        if (documentCount >= 100)
            output.WriteLine();
        return new TrainingCorpus(
            tokens.ToArray(),
            sampleDocuments.ToArray());
    }

    private static float Evaluate(
        IWikiLanguageModel model,
        int[] tokens,
        int validationStartSequence,
        int validationSequences,
        WikiTrainingConfiguration config)
    {
        model.Eval();
        float totalLoss = 0f;
        int completedTargets = 0;
        using (AutogradContext.NoGrad())
        {
            for (int start = 0; start < validationSequences; start += config.BatchSize)
            {
                int count = Math.Min(config.BatchSize, validationSequences - start);
                int[] order = Enumerable.Range(
                    validationStartSequence + start,
                    count).ToArray();
                LanguageBatch values = CreateBatch(
                    tokens,
                    order,
                    0,
                    count,
                    config.ContextLength);
                Tensor logits = model.Forward(
                    values.Input,
                    count,
                    config.ContextLength);
                Tensor loss = logits.CrossEntropyWithLogits(values.Target);
                totalLoss += loss.Data[0] * values.ValidTargetCount;
                completedTargets += values.ValidTargetCount;
            }
        }
        return totalLoss / completedTargets;
    }

    internal static LanguageBatch CreateBatch(
        IReadOnlyList<int> tokens,
        IReadOnlyList<int> sequenceOrder,
        int orderStart,
        int count,
        int contextLength)
    {
        var input = new int[checked(count * contextLength)];
        var target = new int[input.Length];
        Array.Fill(target, Tensor.DefaultCrossEntropyIgnoreIndex);
        int validTargetCount = 0;
        if (tokens is int[] tokenArray)
        {
            void CopySequence(int item)
            {
                int sequence = sequenceOrder[orderStart + item];
                int tokenStart = checked(sequence * contextLength);
                int validLength = Math.Min(
                    contextLength,
                    tokenArray.Length - tokenStart - 1);
                if (validLength <= 0)
                    return;
                Array.Copy(
                    tokenArray,
                    tokenStart,
                    input,
                    item * contextLength,
                    validLength);
                Array.Copy(
                    tokenArray,
                    tokenStart + 1,
                    target,
                    item * contextLength,
                    validLength);
                Interlocked.Add(ref validTargetCount, validLength);
            }

            for (int item = 0; item < count; item++)
                CopySequence(item);
            return new LanguageBatch(input, target, validTargetCount);
        }

        for (int item = 0; item < count; item++)
        {
            int sequence = sequenceOrder[orderStart + item];
            int tokenStart = checked(sequence * contextLength);
            int validLength = Math.Min(
                contextLength,
                tokens.Count - tokenStart - 1);
            for (int position = 0; position < validLength; position++)
            {
                input[item * contextLength + position] =
                    tokens[tokenStart + position];
                target[item * contextLength + position] =
                    tokens[tokenStart + position + 1];
            }
            validTargetCount += validLength;
        }
        return new LanguageBatch(input, target, validTargetCount);
    }

    internal static LanguageBatch CreateStreamingBatch(
        List<int> buffer,
        int batchSize,
        int sequenceLength)
    {
        int targetCount = checked(batchSize * sequenceLength);
        if (buffer.Count < 2)
        {
            throw new ArgumentException(
                "Streaming token buffer does not contain a token pair.",
                nameof(buffer));
        }

        var input = new int[targetCount];
        var target = new int[targetCount];
        Array.Fill(target, Tensor.DefaultCrossEntropyIgnoreIndex);
        int validTargetCount = Math.Min(targetCount, buffer.Count - 1);
        Span<int> values = CollectionsMarshal.AsSpan(buffer);
        values[..validTargetCount].CopyTo(input);
        values.Slice(1, validTargetCount).CopyTo(target);
        buffer.RemoveRange(0, validTargetCount);
        return new LanguageBatch(input, target, validTargetCount);
    }

    private static IEnumerable<string> ReadDocuments(
        string path,
        string textColumn,
        int? maxDocuments)
    {
        IAsyncEnumerator<string> enumerator = WikiParquetCorpus
            .ReadTextsAsync(path, textColumn, maxDocuments)
            .GetAsyncEnumerator();
        try
        {
            while (enumerator.MoveNextAsync().AsTask().GetAwaiter().GetResult())
                yield return enumerator.Current;
        }
        finally
        {
            enumerator.DisposeAsync().AsTask().GetAwaiter().GetResult();
        }
    }

    private static void AddReservoirSample(
        string document,
        List<string> samples,
        ref int eligibleDocuments,
        int capacity,
        Random random)
    {
        eligibleDocuments++;
        if (samples.Count < capacity)
        {
            samples.Add(document);
            return;
        }

        int replacement = random.Next(eligibleDocuments);
        if (replacement < samples.Count)
            samples[replacement] = document;
    }

    internal static IWikiLanguageModel CreateModel(
        WikiTrainingConfiguration config,
        int vocabularySize)
    {
        if (config.IsForgetMemoryV2Architecture())
        {
            return new FrogetMemoryV2Gpt(
                vocabularySize,
                config.ContextLength,
                config.ModelWidth,
                config.HiddenSize,
                config.Layers,
                config.ForgetMemoryKeyWidth,
                config.ForgetMemoryValueWidth,
                config.ForgetMemoryRetentionMinimum,
                config.ForgetMemoryRetentionMaximum,
                new Random(config.Seed),
                config.InitializationScale,
                config.Dropout);
        }

        if (config.IsArchitecture(
            WikiTrainingConfiguration.ForgetScanArchitecture))
        {
            return new ForgetScanGpt(
                vocabularySize,
                config.ContextLength,
                config.ModelWidth,
                config.HiddenSize,
                config.Layers,
                new Random(config.Seed),
                config.InitializationScale,
                config.Dropout);
        }

        if (config.IsArchitecture(WikiTrainingConfiguration.HyenaArchitecture))
        {
            return new HyenaGpt(
                vocabularySize,
                config.ContextLength,
                config.ModelWidth,
                config.HiddenSize,
                config.Layers,
                new Random(config.Seed),
                config.InitializationScale,
                config.Dropout,
                config.HyenaFilterWidth,
                config.GetHyenaConvolutionAlgorithm());
        }

        return new GptRinWikiJp(
            vocabularySize,
            config.ContextLength,
            config.ModelWidth,
            config.Heads,
            config.HiddenSize,
            config.Layers,
            new Random(config.Seed),
            config.InitializationScale,
            config.Dropout);
    }

    private static void WriteEffectiveTrainingConfiguration(
        WikiTrainingConfiguration config,
        TextWriter output)
    {
        string architectureDetails = config.IsForgetMemoryV2Architecture()
            ? $", matrix delta memory key {config.ForgetMemoryKeyWidth}, " +
                $"value {config.ForgetMemoryValueWidth}, retention " +
                $"{config.ForgetMemoryRetentionMinimum:G}-" +
                $"{config.ForgetMemoryRetentionMaximum:G}"
            : config.IsArchitecture(WikiTrainingConfiguration.HyenaArchitecture)
                ? $", Hyena filter width {config.HyenaFilterWidth}, " +
                    $"convolution {config.HyenaConvolutionAlgorithm}"
                : config.IsArchitecture(
                    WikiTrainingConfiguration.ForgetScanArchitecture)
                    ? ", associative forget scan"
                    : string.Empty;
        output.WriteLine(
            $"effective training = epochs {config.Epochs}, " +
            $"batch {config.BatchSize}, context {config.ContextLength}, " +
            $"max document tokens {config.MaxDocumentTokens}");
        output.WriteLine(
            $"effective model = {config.ModelArchitecture}, " +
            $"vocabulary {config.VocabularySize}, " +
            $"width {config.ModelWidth}, heads {config.Heads}, " +
            $"hidden {config.HiddenSize}, layers {config.Layers}, " +
            $"dropout {config.Dropout:G}" + architectureDetails);
        output.WriteLine(
            $"special tokens = {BpeTokenizer.PadToken}:" +
            $"{BpeTokenizer.PadTokenId}, {BpeTokenizer.BosToken}:" +
            $"{BpeTokenizer.BosTokenId}, {BpeTokenizer.EosToken}:" +
            $"{BpeTokenizer.EosTokenId}; padded targets use ignoreIndex " +
            $"{Tensor.DefaultCrossEntropyIgnoreIndex}");
    }

    private static string GetCheckpointArchitecture(
        WikiModelCheckpoint checkpoint)
        => string.IsNullOrWhiteSpace(checkpoint.ModelArchitecture)
            ? WikiTrainingConfiguration.TransformerArchitecture
            : checkpoint.ModelArchitecture;

    private static bool CheckpointArchitectureMatchesConfiguration(
        WikiModelCheckpoint checkpoint,
        WikiTrainingConfiguration config)
        => checkpoint.VocabularySize == config.VocabularySize
            && checkpoint.ContextLength == config.ContextLength
            && checkpoint.ModelWidth == config.ModelWidth
            && checkpoint.Heads == config.Heads
            && checkpoint.HiddenSize == config.HiddenSize
            && checkpoint.Layers == config.Layers
            && (config.IsForgetMemoryV2Architecture()
                ? IsCheckpointForgetMemoryV2(checkpoint)
                : string.Equals(
                    GetCheckpointArchitecture(checkpoint),
                    config.ModelArchitecture,
                    StringComparison.OrdinalIgnoreCase))
            && (!string.Equals(
                    GetCheckpointArchitecture(checkpoint),
                    WikiTrainingConfiguration.HyenaArchitecture,
                    StringComparison.OrdinalIgnoreCase)
                || checkpoint.HyenaFilterWidth == config.HyenaFilterWidth)
            && (!IsCheckpointForgetMemoryV2(checkpoint)
                || (checkpoint.ForgetMemoryKeyWidth
                        == config.ForgetMemoryKeyWidth
                    && checkpoint.ForgetMemoryValueWidth
                        == config.ForgetMemoryValueWidth
                    && checkpoint.ForgetMemoryRetentionMinimum
                        == config.ForgetMemoryRetentionMinimum
                    && checkpoint.ForgetMemoryRetentionMaximum
                        == config.ForgetMemoryRetentionMaximum));

    private static bool IsCheckpointForgetMemoryV2(
        WikiModelCheckpoint checkpoint)
        => string.Equals(
                GetCheckpointArchitecture(checkpoint),
                WikiTrainingConfiguration.ForgetMemoryV2Architecture,
                StringComparison.OrdinalIgnoreCase)
            || string.Equals(
                GetCheckpointArchitecture(checkpoint),
                WikiTrainingConfiguration.FrogetMemoryV2ArchitectureAlias,
                StringComparison.OrdinalIgnoreCase);

    internal static IOptimizer CreateOptimizer(
        IWikiLanguageModel model,
        WikiTrainingConfiguration config)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(config);

        if (config.IsOptimizer(WikiTrainingConfiguration.NekoMuonOptimizer))
        {
            var nekoMuon = new NekoMuon(
                model.HiddenWeightParameters,
                new NekoMuonOptions
                {
                    LearningRate = config.LearningRate,
                    NewtonSchulzInterval =
                        config.NekoMuonNewtonSchulzInterval,
                    WeightDecay = config.WeightDecay,
                });
            var auxiliaryAdamW = new AdamW(
                model.AuxiliaryParameters,
                new AdamWOptions
                {
                    LearningRate = config.AuxiliaryLearningRate,
                    Beta1 = 0.9f,
                    Beta2 = 0.95f,
                    Epsilon = 1e-8f,
                    WeightDecay = config.WeightDecay,
                    UseBFloat16FirstMoment =
                        config.AdamWUseBFloat16FirstMoment,
                    UseBFloat16SecondMoment =
                        config.AdamWUseBFloat16SecondMoment,
                });
            return new CompositeOptimizer(nekoMuon, auxiliaryAdamW);
        }

        return new AdamW(
            model.Parameters(),
            new AdamWOptions
            {
                LearningRate = config.LearningRate,
                WeightDecay = config.WeightDecay,
                UseBFloat16FirstMoment =
                    config.AdamWUseBFloat16FirstMoment,
                UseBFloat16SecondMoment =
                    config.AdamWUseBFloat16SecondMoment,
            });
    }

    internal static float CalculateLearningRateFactor(
        double overallProgress,
        float warmupPercent)
    {
        const float MinimumFactor = 1e-6f;
        if (!double.IsFinite(overallProgress)
            || overallProgress < 0d
            || overallProgress > 1d)
        {
            throw new ArgumentOutOfRangeException(nameof(overallProgress));
        }
        if (!float.IsFinite(warmupPercent)
            || warmupPercent < 0f
            || warmupPercent >= 100f)
        {
            throw new ArgumentOutOfRangeException(nameof(warmupPercent));
        }

        double warmupFraction = warmupPercent / 100d;
        if (warmupFraction > 0d && overallProgress <= warmupFraction)
        {
            return MathF.Max(
                MinimumFactor,
                (float)(overallProgress / warmupFraction));
        }

        double decayProgress = warmupFraction == 1d
            ? 1d
            : (overallProgress - warmupFraction)
                / (1d - warmupFraction);
        decayProgress = Math.Clamp(decayProgress, 0d, 1d);
        float cosine = 0.5f
            * (1f + MathF.Cos(MathF.PI * (float)decayProgress));
        return MathF.Max(MinimumFactor, cosine);
    }

    internal static float SetScheduledLearningRates(
        IOptimizer optimizer,
        WikiTrainingConfiguration config,
        double overallProgress)
    {
        ArgumentNullException.ThrowIfNull(optimizer);
        ArgumentNullException.ThrowIfNull(config);
        float factor = CalculateLearningRateFactor(
            overallProgress,
            config.WarmupPercent);

        if (config.IsOptimizer(WikiTrainingConfiguration.NekoMuonOptimizer))
        {
            if (optimizer is not CompositeOptimizer composite
                || composite.Optimizers.Count != 2
                || composite.Optimizers[0]
                    is not ILearningRateAdjustable primary
                || composite.Optimizers[1]
                    is not ILearningRateAdjustable auxiliary)
            {
                throw new InvalidOperationException(
                    "NekoMuon scheduling requires adjustable primary and " +
                    "auxiliary optimizers.");
            }

            primary.SetLearningRate(
                MathF.Max(float.Epsilon, config.LearningRate * factor));
            auxiliary.SetLearningRate(
                MathF.Max(
                    float.Epsilon,
                    config.AuxiliaryLearningRate * factor));
            return factor;
        }

        if (optimizer is not ILearningRateAdjustable adjustable)
        {
            throw new InvalidOperationException(
                "Learning-rate scheduling requires an adjustable optimizer.");
        }
        adjustable.SetLearningRate(
            MathF.Max(float.Epsilon, config.LearningRate * factor));
        return factor;
    }

    private static void WriteOptimizerSummary(
        IWikiLanguageModel model,
        WikiTrainingConfiguration config,
        TextWriter output)
    {
        if (config.IsOptimizer(WikiTrainingConfiguration.NekoMuonOptimizer))
        {
            output.WriteLine(
                $"optimizer = NekoMuon " +
                $"({model.HiddenWeightParameters.Count} matrix parameters, " +
                $"lr {config.LearningRate:G}, Newton-Schulz every " +
                $"{config.NekoMuonNewtonSchulzInterval} steps) + AdamW " +
                $"({model.AuxiliaryParameters.Count} auxiliary parameters, " +
                $"lr {config.AuxiliaryLearningRate:G}, moments " +
                $"{GetAdamWMomentStorage(config)})");
        }
        else
        {
            output.WriteLine(
                $"optimizer = AdamW ({model.Parameters().Count()} " +
                $"parameters, lr {config.LearningRate:G}, moments " +
                $"{GetAdamWMomentStorage(config)})");
        }
        output.WriteLine(
            $"learning-rate schedule = linear warmup " +
            $"{config.WarmupPercent:G}% of total training, then cosine " +
            "decay");
    }

    private static string GetAdamWMomentStorage(
        WikiTrainingConfiguration config)
        => $"{(config.AdamWUseBFloat16FirstMoment ? "bf16" : "f32")}/" +
            $"{(config.AdamWUseBFloat16SecondMoment ? "bf16" : "f32")}";

    private static IWikiLanguageModel CreateModel(
        WikiModelCheckpoint checkpoint,
        int seed)
    {
        if (IsCheckpointForgetMemoryV2(checkpoint))
        {
            return new FrogetMemoryV2Gpt(
                checkpoint.VocabularySize,
                checkpoint.ContextLength,
                checkpoint.ModelWidth,
                checkpoint.HiddenSize,
                checkpoint.Layers,
                checkpoint.ForgetMemoryKeyWidth,
                checkpoint.ForgetMemoryValueWidth,
                checkpoint.ForgetMemoryRetentionMinimum,
                checkpoint.ForgetMemoryRetentionMaximum,
                new Random(seed),
                checkpoint.InitializationScale,
                checkpoint.Dropout);
        }

        if (string.Equals(
            GetCheckpointArchitecture(checkpoint),
            WikiTrainingConfiguration.ForgetScanArchitecture,
            StringComparison.OrdinalIgnoreCase))
        {
            return new ForgetScanGpt(
                checkpoint.VocabularySize,
                checkpoint.ContextLength,
                checkpoint.ModelWidth,
                checkpoint.HiddenSize,
                checkpoint.Layers,
                new Random(seed),
                checkpoint.InitializationScale,
                checkpoint.Dropout);
        }

        if (string.Equals(
            GetCheckpointArchitecture(checkpoint),
            WikiTrainingConfiguration.HyenaArchitecture,
            StringComparison.OrdinalIgnoreCase))
        {
            return new HyenaGpt(
                checkpoint.VocabularySize,
                checkpoint.ContextLength,
                checkpoint.ModelWidth,
                checkpoint.HiddenSize,
                checkpoint.Layers,
                new Random(seed),
                checkpoint.InitializationScale,
                checkpoint.Dropout,
                checkpoint.HyenaFilterWidth);
        }

        return new GptRinWikiJp(
            checkpoint.VocabularySize,
            checkpoint.ContextLength,
            checkpoint.ModelWidth,
            checkpoint.Heads,
            checkpoint.HiddenSize,
            checkpoint.Layers,
            new Random(seed),
            checkpoint.InitializationScale,
            checkpoint.Dropout);
    }

    internal static DatasetContinuation CreateDatasetContinuation(
        IWikiLanguageModel model,
        BpeTokenizer tokenizer,
        IReadOnlyList<string> documents,
        WikiTrainingConfiguration config,
        Random random)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(documents);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(random);
        if (documents.Count == 0)
        {
            throw new ArgumentException(
                "At least one dataset sample document is required.",
                nameof(documents));
        }

        string document = documents[random.Next(documents.Count)];
        if (!TryGetDocumentSplit(document, out int split))
        {
            throw new ArgumentException(
                "Dataset sample document must have non-empty first and " +
                "second halves.",
                nameof(documents));
        }

        string prompt = document[..split];
        string expected = document[split..];
        int[] promptIds = tokenizer.Encode(prompt, addBos: true);
        int[] generatedIds = model.GenerateTokenIds(
            promptIds,
            config.MaxNewTokens,
            config.Temperature,
            config.TopK,
            BpeTokenizer.EosTokenId,
            random);
        string generated = tokenizer.Decode(
            generatedIds.Skip(promptIds.Length));

        const int displayCharacters = 400;
        return new DatasetContinuation(
            document.Length,
            split,
            TakeTail(prompt, displayCharacters),
            TakeHead(expected, displayCharacters),
            generated);
    }

    private static void WriteDatasetContinuation(
        long step,
        DatasetContinuation sample,
        TextWriter output)
    {
        output.WriteLine();
        output.WriteLine(
            $"dataset continuation sample at step {step:N0} " +
            $"(split {sample.SplitIndex:N0}/{sample.DocumentLength:N0} chars):");
        output.WriteLine("[prompt: first half, final context]");
        output.WriteLine(sample.PromptTail);
        output.WriteLine("[dataset continuation: second half excerpt]");
        output.WriteLine(sample.ExpectedContinuation);
        output.WriteLine("[model continuation]");
        output.WriteLine(sample.GeneratedContinuation);
        output.WriteLine();
    }

    private static void WriteFinalDatasetContinuation(
        IWikiLanguageModel model,
        BpeTokenizer tokenizer,
        IReadOnlyList<string> documents,
        WikiTrainingConfiguration config,
        Random random,
        long step,
        TextWriter output)
    {
        if (documents.Count == 0)
        {
            output.WriteLine(
                "dataset continuation unavailable: no splittable sample " +
                "document was retained");
            return;
        }

        DatasetContinuation sample = CreateDatasetContinuation(
            model,
            tokenizer,
            documents,
            config,
            random);
        WriteDatasetContinuation(step, sample, output);
    }

    private static bool TryGetDocumentSplit(
        string document,
        out int split)
    {
        split = document.Length / 2;
        if (split > 0
            && split < document.Length
            && char.IsLowSurrogate(document[split]))
        {
            split--;
        }
        return split > 0 && split < document.Length;
    }

    private static string TakeHead(string text, int count)
    {
        if (text.Length <= count)
            return text;
        int length = count;
        if (length > 0 && char.IsHighSurrogate(text[length - 1]))
            length--;
        return text[..length] + "…";
    }

    private static string TakeTail(string text, int count)
    {
        if (text.Length <= count)
            return text;
        int start = text.Length - count;
        if (start < text.Length && char.IsLowSurrogate(text[start]))
            start++;
        return "…" + text[start..];
    }

    private static void WriteGeneration(
        IWikiLanguageModel model,
        BpeTokenizer tokenizer,
        string prompt,
        WikiTrainingConfiguration config,
        TextWriter output)
    {
        string generated = model.Generate(
            prompt,
            tokenizer,
            config.MaxNewTokens,
            config.Temperature,
            config.TopK,
            new Random(config.Seed ^ 0x27D4EB2D));
        output.WriteLine("generated text:");
        output.WriteLine(generated);
    }

    private static void SaveCheckpoint(
        string path,
        WikiModelCheckpoint checkpoint)
    {
        string? directory = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(directory))
            Directory.CreateDirectory(directory);
        string temporaryPath = path + ".tmp";
        File.WriteAllText(
            temporaryPath,
            JsonSerializer.Serialize(checkpoint, CheckpointJsonOptions));
        File.Move(temporaryPath, path, overwrite: true);
    }

    private static WikiModelCheckpoint LoadCheckpoint(string path)
    {
        WikiModelCheckpoint checkpoint =
            JsonSerializer.Deserialize<WikiModelCheckpoint>(
                File.ReadAllText(path),
                CheckpointJsonOptions)
            ?? throw new InvalidDataException(
                "Wiki model checkpoint cannot be JSON null.");
        if (checkpoint.FormatVersion is < 1 or > CheckpointFormatVersion
            || checkpoint.Model is null)
        {
            throw new InvalidDataException(
                "Wiki model checkpoint has an unsupported format.");
        }
        return checkpoint;
    }

    private static int DivideRoundUp(int value, int divisor)
        => value / divisor + (value % divisor == 0 ? 0 : 1);

    private static void Shuffle(int[] values, Random random)
    {
        for (int index = values.Length - 1; index > 0; index--)
        {
            int other = random.Next(index + 1);
            (values[index], values[other]) = (values[other], values[index]);
        }
    }

    internal readonly record struct LanguageBatch(
        int[] Input,
        int[] Target,
        int ValidTargetCount);

    private readonly record struct TrainingCorpus(
        int[] Tokens,
        string[] SampleDocuments);

    internal readonly record struct DatasetContinuation(
        int DocumentLength,
        int SplitIndex,
        string PromptTail,
        string ExpectedContinuation,
        string GeneratedContinuation);

    private sealed record WikiModelCheckpoint(
        int FormatVersion,
        int Epoch,
        float ValidationLoss,
        int VocabularySize,
        int ContextLength,
        int ModelWidth,
        int Heads,
        int HiddenSize,
        int Layers,
        float Dropout,
        float InitializationScale,
        ModuleState Model,
        string? ModelArchitecture = null,
        int HyenaFilterWidth = 64,
        int ForgetMemoryKeyWidth = 16,
        int ForgetMemoryValueWidth = 16,
        float ForgetMemoryRetentionMinimum = 0.5f,
        float ForgetMemoryRetentionMaximum = 0.99f);
}
