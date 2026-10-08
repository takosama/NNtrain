using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

public sealed partial class Qwen35QuantizedModel
{
    private SplitOutputHead? _splitOutputHead;

    private void ReleaseSplitOutputHeadForReservation(ArcExecutionLane lane, long bytes)
    {
        // Match the lane's reservation contract before releasing optional state.
        // A malformed request must leave both the existing reservation and the
        // prepared output head unchanged.
        ArgumentOutOfRangeException.ThrowIfNegative(bytes);
        long normalBudget = checked(lane.EffectivePhysicalBufferBudgetBytes + lane.ExternalMemoryReservationBytes);
        if (bytes > normalBudget)
            throw new ArgumentOutOfRangeException(nameof(bytes), "External memory exceeds the lane's device budget.");
        if (_splitOutputHead is not { } split || !ReferenceEquals(split.Peer, lane)) return;

        long requestedBudget = normalBudget - bytes;
        long live = lane.AllocatedBytes;
        // Release only when doing so makes this request feasible. An excessive
        // request which would still exceed the base model's live bytes follows
        // the lane's original rejection path, without discarding the fast path.
        if (requestedBudget - live >= WorkspaceReserveBytes || requestedBudget < live - split.DeviceBytes) return;
        split.Dispose();
        _splitOutputHead = null;
        // SetExternalMemoryReservation subsequently reclaims these now-idle or
        // retired handles before publishing the new budget. Do not recreate the
        // optional half-head if a later call reduces the reservation.
    }

    // An adapter can be attached after load. Retain the complete original head
    // and select its existing path whenever the output projection is modified.
    internal bool SplitOutputHeadAvailable => !_disposed && !_loraFaulted
        && _splitOutputHead is { } split && !_options.LoraTraining
        && !_lora.ContainsKey("output.weight")
        && _prism?.ForwardWeights.Contains("output.weight") != true
        && WorkspaceReserveBytes <= split.Peer.EffectivePhysicalBufferBudgetBytes - split.Peer.AllocatedBytes;

    internal static bool CanSplitOutputHead(uint storageType, int inputWidth, int outputWidth,
        int deviceCount, bool training, bool subgroup, bool outputLora, bool prism)
        => storageType == Qwen35Gguf.Q5KType && inputWidth > 0 && inputWidth % 256 == 0
            && outputWidth >= 2 && deviceCount >= 2 && !training && subgroup && !outputLora && !prism;

    internal static bool CanAllocateSplitOutputHead(int inputWidth, int peerRows,
        ulong maximumAllocationBytes, long availableBytes)
    {
        if (inputWidth <= 0 || inputWidth % 256 != 0 || peerRows <= 0 || availableBytes < 0) return false;
        try
        {
            long weightBytes = checked((long)(inputWidth / 256) * 176 * peerRows);
            long inputBytes = checked((long)inputWidth * sizeof(float));
            long outputBytes = checked((long)peerRows * sizeof(float));
            // GGUF prefix reads and managed staging arrays use an Int32 byte count.
            if (weightBytes > int.MaxValue || inputBytes > int.MaxValue || outputBytes > int.MaxValue)
                return false;
            long largest = Math.Max(weightBytes, Math.Max(inputBytes, outputBytes));
            long total = checked(weightBytes + inputBytes + 2 * outputBytes + WorkspaceReserveBytes);
            return (ulong)largest <= maximumAllocationBytes && total <= availableBytes;
        }
        catch (OverflowException) { return false; }
    }

    private void InitializeSplitOutputHead(GgufReader gguf, Action<string>? progress)
    {
        if (!_options.InferenceSplitOutputHead || !_matrices.TryGetValue("output.weight", out Matrix? head)
            || !CanSplitOutputHead(head.StorageType, head._inputWidth, head.OutputWidth, _lanes.Count,
                _options.LoraTraining, head.SupportsSplitOutputHead, _lora.ContainsKey("output.weight"),
                _prism?.ForwardWeights.Contains("output.weight") == true)) return;

        int peerRows = head.OutputWidth / 2;
        ArcExecutionLane? peer = _lanes.FirstOrDefault(lane => !ReferenceEquals(lane, head.Lane)
            && lane.Options.XmxMatrices && lane.Device.SupportsXmx && lane.Device.MinimumSubgroupSize == 16
            && lane.Device.Extensions.Split(' ').Contains("cl_intel_subgroups")
            && !lane.Options.Qwen35TrainingKernels
            && CanAllocateSplitOutputHead(head._inputWidth, peerRows, lane.Device.MaximumAllocationBytes,
                lane.EffectivePhysicalBufferBudgetBytes - lane.AllocatedBytes));
        if (peer is null) return;

        int weightBytes = checked(head._inputWidth / 256 * 176 * peerRows);
        ArcBuffer? weight = null, input = null, bias = null, output = null;
        try
        {
            // Duplicate only the lower vocabulary half; the owner retains its
            // full weight for adapter, precision-mode and memory fallback.
            weight = peer.UploadRaw(gguf.ReadTensorBytes(gguf.GetTensor("output.weight"), weightBytes));
            input = peer.Allocate(head._inputWidth);
            bias = peer.Allocate(peerRows);
            output = peer.Allocate(peerRows);
            peer.Run("q35a_zero", peerRows, 0, bias, peerRows);
            _splitOutputHead = new SplitOutputHead(peer, weight, input, bias, output,
                head._inputWidth, peerRows, weightBytes);
            weight = input = bias = output = null;
            progress?.Invoke($"Output Q5_K head: {peerRows:N0} vocabulary rows on Arc {peer.DeviceIndex}, "
                + $"{head.OutputWidth - peerRows:N0} on Arc {head.Lane.DeviceIndex}; "
                + $"{weightBytes / 1048576.0:F1} MiB additional resident weights.");
        }
        finally { output?.Dispose(); bias?.Dispose(); input?.Dispose(); weight?.Dispose(); }
    }

    private ArcBuffer? TrySplitOutputHead(ArcBuffer normalizedInput)
    {
        if (!SplitOutputHeadAvailable) return null;
        SplitOutputHead split = _splitOutputHead!;
        Matrix head = _matrices["output.weight"];
        ArcExecutionLane owner = head.Lane;
        long outputBytes = checked((long)head.OutputWidth * sizeof(float));
        if ((ulong)outputBytes > owner.Device.MaximumAllocationBytes
            || outputBytes + WorkspaceReserveBytes > owner.EffectivePhysicalBufferBudgetBytes - owner.AllocatedBytes)
            return null;
        ArcBuffer logits = owner.Allocate(head.OutputWidth);
        try
        {
            // Read the small normalized hidden row before enqueueing the owner
            // projection. Waiting for this read flushes both commands, so the
            // owner's upper half can execute during the peer upload/projection.
            using PendingRead inputRead = owner.ReadAsync(normalizedInput, split.InputStaging);
            RunOutputHeadRange(owner, normalizedInput, head.EncodedOutputHead, _zeroBias[owner], logits,
                head._inputWidth, head.OutputWidth - split.PeerRows,
                split.PeerRows, split.PeerRows, split.PeerRows);
            inputRead.Wait();
            split.Peer.Write(split.Input, split.InputStaging);
            RunOutputHeadRange(split.Peer, split.Input, split.Weight, split.Bias, split.Output,
                head._inputWidth, split.PeerRows, 0, 0, 0);
            using PendingRead outputRead = split.Peer.ReadAsync(split.Output, split.OutputStaging);
            outputRead.Wait();
            // The owner's in-order queue joins its upper half before this write.
            // Keep one complete resident logits vector for existing GPU argmax
            // and the full-vocabulary GUI sampler, including odd-size splits.
            owner.WriteFloatRange(logits, 0, split.OutputStaging);
            return logits;
        }
        catch { logits.Dispose(); throw; }
    }

    private static void RunOutputHeadRange(ArcExecutionLane lane, ArcBuffer input, ArcBuffer weight,
        ArcBuffer bias, ArcBuffer output, int inputWidth, int outputCount,
        int weightFirst, int biasFirst, int outputFirst)
    {
        int group = lane.Options.Qwen35ProjectionWorkgroupSize;
        long workItems = ((long)outputCount + group / 16 - 1) / (group / 16) * group;
        lane.Run("q35l_q5_k_sg16_range", workItems, group, input, weight, bias, output,
            inputWidth, outputCount, weightFirst, biasFirst, outputFirst);
    }

    private sealed class SplitOutputHead : IDisposable
    {
        internal readonly ArcExecutionLane Peer;
        internal readonly ArcBuffer Weight, Input, Bias, Output;
        internal readonly int PeerRows, WeightBytes;
        internal readonly float[] InputStaging, OutputStaging;
        internal long DeviceBytes => Weight.ByteLength + Input.ByteLength + Bias.ByteLength + Output.ByteLength;

        internal SplitOutputHead(ArcExecutionLane peer, ArcBuffer weight, ArcBuffer input,
            ArcBuffer bias, ArcBuffer output, int inputWidth, int peerRows, int weightBytes)
        {
            Peer = peer; Weight = weight; Input = input; Bias = bias; Output = output;
            PeerRows = peerRows; WeightBytes = weightBytes;
            InputStaging = new float[inputWidth]; OutputStaging = new float[peerRows];
        }

        public void Dispose()
        {
            Output.Dispose(); Bias.Dispose(); Input.Dispose(); Weight.Dispose();
        }
    }
}
