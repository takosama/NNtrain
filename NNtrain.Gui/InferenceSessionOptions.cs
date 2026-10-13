namespace NNtrain.Gui;

/// <summary>Immutable session settings and their per-device Qwen execution policy.</summary>
internal sealed class InferenceSessionOptions
{
    internal int InferencePrefillChunkTokens { get; init; }
    internal bool UseOptimizedVisionKernels { get; init; }
    internal bool UseXmxVisionLinear { get; init; }
    internal bool UseXmxVisionAttention { get; init; }
    internal bool UseFlashVisionAttention { get; init; }
    internal bool InferenceBatchMixedAttention { get; init; }
    internal bool CollectKernelTimings { get; init; }
    internal int InferenceProjectionRows { get; init; }
    internal bool InferenceXmxPrefill { get; init; }
    internal bool InferenceXmxPackedPrefill { get; init; }
    internal bool InferenceXmxFactoredPrefill { get; init; }
    internal bool InferenceResidentIq2Panels { get; init; }
    internal bool InferenceXmxGgufBslmPrefill { get; init; }
    internal bool InferenceSubgroupRecurrentRms { get; init; }
    internal int InferenceBufferPoolMiB { get; init; }
    internal int InferenceDeferredReleaseMiB { get; init; }
    internal bool InferenceBatchRecurrent { get; init; }
    internal bool InferenceSplitOutputHead { get; init; }
    internal bool PrepareVisionOnLoad { get; init; }

    internal Qwen35ExecutionOptions CreateQwenExecutionOptions(int deviceCount, bool hasAdapter)
        => new()
        {
            InferencePrefillChunkTokens = InferencePrefillChunkTokens,
            InferenceBatchMixedAttention = InferenceBatchMixedAttention,
            InferenceBatchTextAttention = true,
            // Use both GPUs for the Q5_K vocabulary projection when
            // the model's format, adapters and memory budget allow it.
            InferenceSplitOutputHead = InferenceSplitOutputHead && deviceCount > 1,
            InferencePackedIq2Decode = true,
            CollectKernelTimings = CollectKernelTimings,
            InferenceProjectionRows = InferenceProjectionRows,
            InferenceXmxPrefill = InferenceXmxPrefill,
            InferenceXmxPackedPrefill = InferenceXmxPackedPrefill,
            InferenceXmxFactoredPrefill = InferenceXmxFactoredPrefill,
            InferenceXmxGgufBslmPrefill = InferenceXmxGgufBslmPrefill,
            // Resident IQ2 panels exchange extra capacity for speed.
            // Keep single-device text models within their old budget.
            InferenceResidentIq2Panels = InferenceResidentIq2Panels && deviceCount > 1,
            InferenceSubgroupRecurrentRms = InferenceSubgroupRecurrentRms,
            InferenceBufferPoolMiB = InferenceBufferPoolMiB,
            InferenceDeferredReleaseMiB = InferenceDeferredReleaseMiB,
            InferenceBatchRecurrent = InferenceBatchRecurrent,
            ComputeModelFingerprintOnLoad = hasAdapter
        };
}
