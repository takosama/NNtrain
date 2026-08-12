using System.Diagnostics;

namespace NNtrain;

internal static partial class WikiLanguageModelCommand
{
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
            torch.manual_seed(config.Seed);
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
            BpeTokenizer existing = tokenizers.load_bpe(config.TokenizerPath);
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
            tokenizer = tokenizers.train_bpe(
                ReadDocuments(
                    config.DataPath,
                    config.TextColumn,
                    config.TokenizerTrainingDocuments),
                config.VocabularySize,
                config.TokenizerTrainingBytes);
            tokenizer.save(config.TokenizerPath);
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
        WarmupCosineProgressLRScheduler scheduler =
            lr_scheduler.WarmupCosineProgressLR(
                optimizer,
                config.WarmupPercent);
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
            model.train();
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
                optimizer.zero_grad();
                Tensor logits = model.Forward(
                    values.Input,
                    count,
                    config.ContextLength);
                Tensor loss = logits.CrossEntropyWithLogits(values.Target);
                loss.backward();
                scheduler.step((globalStep + 1d) / totalTrainingSteps);
                optimizer.step();
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

        BpeTokenizer tokenizer = tokenizers.load_bpe(config.TokenizerPath);
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
        WarmupCosineProgressLRScheduler scheduler =
            lr_scheduler.WarmupCosineProgressLR(
                optimizer,
                config.WarmupPercent);
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
            model.train();
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
                optimizer.zero_grad();
                Tensor logits = model.Forward(
                    values.Input,
                    batchSize,
                    sequenceLength);
                Tensor loss = logits.CrossEntropyWithLogits(values.Target);
                loss.backward();
                double documentProgress = documentsPerEpoch == 0
                    ? 0d
                    : Math.Min(
                        1d,
                        (double)documentsProcessed / documentsPerEpoch);
                double overallProgress =
                    (epoch - 1d + documentProgress) / config.Epochs;
                scheduler.step(overallProgress);
                optimizer.step();
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
        model.eval();
        float totalLoss = 0f;
        int completedTargets = 0;
        using (torch.no_grad())
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

    private readonly record struct TrainingCorpus(
        int[] Tokens,
        string[] SampleDocuments);

}
