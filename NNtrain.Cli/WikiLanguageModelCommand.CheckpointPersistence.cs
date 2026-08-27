using NNtrain.Training.Persistence;

namespace NNtrain;

internal static partial class WikiLanguageModelCommand
{
    /// <summary>
    /// Compatibility boundary around the existing Wiki v1-v8 readers and
    /// v8 artifact layout. It changes publication ownership, not bytes,
    /// artifact names, slots, or optimizer-leaf order.
    /// </summary>
    private static class WikiCheckpointCompatibilityAdapter
    {
        private static readonly CheckpointVersionRegistry<WikiModelCheckpoint>
            Readers = CreateReaders();
        private static readonly CheckpointVersionRegistry<WikiWriteRequest>
            Writers = CreateWriters();

        internal static WikiModelCheckpoint Load(string path)
        {
            try
            {
                return Readers.Read(path);
            }
            catch (NotSupportedException exception)
            {
                throw new InvalidDataException(
                    "Wiki model checkpoint has an unsupported format.",
                    exception);
            }
        }

        internal static WikiCheckpointMetadata LoadMetadata(string path)
        {
            WikiCheckpointMetadata metadata =
                torch.load<WikiCheckpointMetadata>(path);
            RequireReadable(metadata.FormatVersion);
            return metadata;
        }

        internal static WikiResumeCheckpointData LoadResume(string path)
        {
            WikiResumeCheckpointData resume =
                torch.load<WikiResumeCheckpointData>(path);
            RequireReadable(resume.FormatVersion);
            return resume;
        }

        internal static void Save(
            string path,
            WikiModelCheckpoint checkpoint,
            IOptimizer optimizer,
            Module? currentModelSource,
            ICheckpointFaultInjector? faultInjector)
        {
            ValidateCheckpoint(checkpoint, requireArtifactMetadata: false);
            ArgumentNullException.ThrowIfNull(optimizer);
            string fullPath = Path.GetFullPath(path);

            WikiCheckpointMetadata? previous = File.Exists(fullPath)
                ? torch.load<WikiCheckpointMetadata>(fullPath)
                : null;
            int artifactSlot = previous is
                { FormatVersion: >= 7, ArtifactSlot: 0 }
                    ? 1
                    : 0;
            int previousBestSlot = previous is { FormatVersion: >= 7 }
                ? GetBestArtifactSlot(
                    previous.ArtifactSlot,
                    previous.BestArtifactSlot)
                : -1;
            bool bestMetadataChanged = previous is null
                || previous.Epoch != checkpoint.Epoch
                || BitConverter.SingleToInt32Bits(previous.ValidationLoss)
                    != BitConverter.SingleToInt32Bits(
                        checkpoint.ValidationLoss);
            bool writeBestArtifact = previousBestSlot < 0
                || !File.Exists(GetBestModelArtifactPath(
                    fullPath,
                    previousBestSlot))
                || bestMetadataChanged;
            int bestArtifactSlot = writeBestArtifact
                ? previousBestSlot is 0 ? 1 : 0
                : previousBestSlot;
            Module? bestModelSource = writeBestArtifact
                && checkpoint.Model.Parameters.Length == 0
                    ? currentModelSource
                    : null;
            if (writeBestArtifact
                && checkpoint.Model.Parameters.Length == 0
                && bestModelSource is null)
            {
                throw new InvalidDataException(
                    "The best-model artifact is missing or its metadata changed, " +
                    "but the in-memory best state has already been released. " +
                    "Refusing to replace it with the current model.");
            }

            ModuleState? currentState = currentModelSource is null
                ? checkpoint.CurrentModel ?? checkpoint.Model
                : null;
            TensorPrecisionMode precisionMode =
                GetCheckpointPrecisionMode(checkpoint);
            bool bfp8Artifact = precisionMode is TensorPrecisionMode.Bfp8
                or TensorPrecisionMode.Mix8_32;
            bool float32CurrentArtifact = bfp8Artifact
                || precisionMode == TensorPrecisionMode.Mix16_32;
            ModuleState? currentArtifact = currentState is not null
                && float32CurrentArtifact
                    ? RelabelStateDType(currentState, TensorDType.Float32)
                    : currentState;
            ModuleState? bestArtifact = writeBestArtifact
                ? checkpoint.Model
                : null;
            if (bfp8Artifact && bestArtifact is not null)
            {
                bestArtifact = RelabelStateDType(
                    bestArtifact,
                    TensorDType.Float32);
            }
            IReadOnlyList<IOptimizer> leaves =
                OptimizerStateStream.GetLeafOptimizers(optimizer);
            var optimizerTypes = new string[leaves.Count];
            for (int index = 0; index < leaves.Count; index++)
            {
                optimizerTypes[index] =
                    OptimizerStateStream.GetStateType(leaves[index]);
            }

            var manifest = checkpoint with
            {
                Model = EmptyModuleState(),
                CurrentModel = null,
                Optimizer = null,
                ArtifactSlot = artifactSlot,
                BestArtifactSlot = bestArtifactSlot,
                OptimizerStateTypes = optimizerTypes,
            };
            var request = new WikiWriteRequest(
                fullPath,
                currentArtifact,
                currentModelSource,
                float32CurrentArtifact
                    ? TensorDType.Float32
                    : null,
                bestArtifact,
                bestModelSource,
                bfp8Artifact ? TensorDType.Float32 : null,
                artifactSlot,
                bestArtifactSlot,
                leaves,
                manifest);
            var repository = new CheckpointRepository(
                fullPath,
                faultInjector);
            Writers.Write(
                repository,
                checkpoint.FormatVersion,
                request);

            // Preserve the previous peak-memory behavior after the exact
            // current snapshot and optimizer states have reached disk.
            checkpoint = manifest;
            request = null!;
            currentState = null;
            currentArtifact = null!;
            GC.Collect(
                GC.MaxGeneration,
                GCCollectionMode.Aggressive,
                blocking: true,
                compacting: true);
        }

        private static CheckpointVersionRegistry<WikiModelCheckpoint>
            CreateReaders()
        {
            var registry = new CheckpointVersionRegistry<WikiModelCheckpoint>(
                DetectFormatVersion);
            for (int version = 1;
                version <= CheckpointFormatVersion;
                version++)
            {
                registry.RegisterReader(
                    new DelegateCheckpointVersionReader<WikiModelCheckpoint>(
                        version,
                        torch.load<WikiModelCheckpoint>));
            }
            return registry;
        }

        private static CheckpointVersionRegistry<WikiWriteRequest>
            CreateWriters()
        {
            var registry = new CheckpointVersionRegistry<WikiWriteRequest>();
            registry.RegisterWriter(
                new DelegateCheckpointVersionWriter<WikiWriteRequest>(
                    CheckpointFormatVersion,
                    WriteCurrentFormat));
            return registry;
        }

        private static int DetectFormatVersion(string path)
            => torch.load<WikiCheckpointVersionHeader>(path).FormatVersion;

        private static void RequireReadable(int formatVersion)
        {
            if (!Readers.CanRead(formatVersion))
            {
                throw new InvalidDataException(
                    "Wiki model checkpoint has an unsupported format.");
            }
        }

        private static void WriteCurrentFormat(
            CheckpointWriteTransaction transaction,
            WikiWriteRequest request)
        {
            transaction.PublishArtifact(
                new CheckpointArtifactWriteRequest(
                    GetCurrentModelArtifactPath(
                        request.FullPath,
                        request.ArtifactSlot),
                    stagingPath =>
                    {
                        if (request.CurrentModelSource is not null)
                        {
                            SafeTensorFile.SaveModel(
                                request.CurrentModelSource,
                                stagingPath,
                                artifactDTypeOverride:
                                    request.CurrentArtifactDTypeOverride);
                        }
                        else
                        {
                            safetensors.torch.save_file(
                                request.CurrentModel
                                    ?? throw new InvalidOperationException(
                                        "Current checkpoint payload is missing."),
                                stagingPath);
                        }
                    },
                    PreservePreviousOnRollback: false));
            if (request.BestModel is not null
                || request.BestModelSource is not null)
            {
                transaction.PublishArtifact(
                    GetBestModelArtifactPath(
                        request.FullPath,
                        request.BestArtifactSlot),
                    stagingPath =>
                    {
                        if (request.BestModelSource is not null)
                        {
                            SafeTensorFile.SaveModel(
                                request.BestModelSource,
                                stagingPath,
                                artifactDTypeOverride:
                                    request.BestArtifactDTypeOverride);
                        }
                        else
                        {
                            safetensors.torch.save_file(
                                request.BestModel!,
                                stagingPath);
                        }
                    },
                    preservePreviousOnRollback: false);
            }

            for (int index = 0;
                index < request.OptimizerLeaves.Count;
                index++)
            {
                IOptimizer leaf = request.OptimizerLeaves[index];
                transaction.PublishArtifact(
                    GetOptimizerBinaryArtifactPath(
                        request.FullPath,
                        request.ArtifactSlot,
                        index),
                    stagingPath => SaveOptimizerBinaryArtifact(
                        stagingPath,
                        leaf),
                    preservePreviousOnRollback: false);
                GC.Collect(
                    GC.MaxGeneration,
                    GCCollectionMode.Aggressive,
                    blocking: true,
                    compacting: true);
            }

            transaction.CommitManifest(
                stagingPath => torch.save(request.Manifest, stagingPath));
        }

        private static void SaveOptimizerBinaryArtifact(
            string path,
            IOptimizer optimizer)
        {
            using var stream = new FileStream(
                path,
                FileMode.Create,
                FileAccess.Write,
                FileShare.None,
                bufferSize: 4 * 1024 * 1024,
                FileOptions.SequentialScan);
            OptimizerStateStream.SaveStateBinary(optimizer, stream);
            stream.Flush(flushToDisk: true);
        }

        private sealed record WikiCheckpointVersionHeader(int FormatVersion);

        private sealed record WikiWriteRequest(
            string FullPath,
            ModuleState? CurrentModel,
            Module? CurrentModelSource,
            TensorDType? CurrentArtifactDTypeOverride,
            ModuleState? BestModel,
            Module? BestModelSource,
            TensorDType? BestArtifactDTypeOverride,
            int ArtifactSlot,
            int BestArtifactSlot,
            IReadOnlyList<IOptimizer> OptimizerLeaves,
            WikiModelCheckpoint Manifest);
    }
}
