using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

public sealed partial class Qwen35QuantizedModel
{
    // Non-owning references resolved once after the validated tensor directory
    // is loaded. LoRA attachment remains dynamic, keyed by each matrix's name.
    private sealed class LayerBindings
    {
        internal readonly LayerState State;
        internal readonly bool Recurrent;
        internal readonly ArcBuffer AttentionNorm, PostAttentionNorm;
        internal readonly Matrix FfnGate, FfnUp, FfnDown, AttentionOutput;
        internal readonly Matrix? Qkv, RecurrentGate, Alpha, Beta, Query, Key, Value;
        internal readonly ArcBuffer? Convolution, DtBias, A, RecurrentNorm, QueryNorm, KeyNorm;

        internal LayerBindings(Qwen35QuantizedModel model, int layer, LayerState state)
        {
            State = state;
            Recurrent = model.Descriptor.IsRecurrent(layer);
            string prefix = $"blk.{layer}.";
            Matrix GetMatrix(string suffix) => model._matrices[prefix + suffix];
            ArcBuffer GetDense(string suffix) => model._dense[prefix + suffix];
            AttentionNorm = GetDense("attn_norm.weight");
            PostAttentionNorm = GetDense("post_attention_norm.weight");
            FfnGate = GetMatrix("ffn_gate.weight");
            FfnUp = GetMatrix("ffn_up.weight");
            FfnDown = GetMatrix("ffn_down.weight");
            if (Recurrent)
            {
                Qkv = GetMatrix("attn_qkv.weight");
                RecurrentGate = GetMatrix("attn_gate.weight");
                Alpha = GetMatrix("ssm_alpha.weight");
                Beta = GetMatrix("ssm_beta.weight");
                AttentionOutput = GetMatrix("ssm_out.weight");
                Convolution = GetDense("ssm_conv1d.weight");
                DtBias = GetDense("ssm_dt.bias");
                A = GetDense("ssm_a");
                RecurrentNorm = GetDense("ssm_norm.weight");
            }
            else
            {
                Query = GetMatrix("attn_q.weight");
                Key = GetMatrix("attn_k.weight");
                Value = GetMatrix("attn_v.weight");
                AttentionOutput = GetMatrix("attn_output.weight");
                QueryNorm = GetDense("attn_q_norm.weight");
                KeyNorm = GetDense("attn_k_norm.weight");
            }
        }
    }

    private sealed class LayerState(ArcExecutionLane lane, Qwen35GgufDescriptor d, bool recurrent) : IDisposable
    {
        internal ArcExecutionLane Lane { get; } = lane;
        internal ArcBuffer? Convolution, Recurrent, Keys, Values;
        private ArcBuffer? _promptConvolution, _promptRecurrent;
        private int _capacity;
        internal long StorageBytes => (Convolution?.ByteLength ?? 0) + (Recurrent?.ByteLength ?? 0)
            + (Keys?.ByteLength ?? 0) + (Values?.ByteLength ?? 0)
            + (_promptConvolution?.ByteLength ?? 0) + (_promptRecurrent?.ByteLength ?? 0);
        internal long AdditionalPromptSnapshotBytes => recurrent
            ? (_promptConvolution is null ? Convolution?.ByteLength ?? 0 : 0)
                + (_promptRecurrent is null ? Recurrent?.ByteLength ?? 0 : 0) : 0;
        internal bool HasPromptState => !recurrent || (_promptConvolution is not null && _promptRecurrent is not null);
        internal void Initialize()
        {
            if (recurrent)
            {
                Convolution = Lane.Allocate(checked((2 * d.LinearKeyHeads + d.LinearValueHeads) * d.LinearHeadWidth * (d.ConvKernel - 1)));
                Recurrent = Lane.Allocate(checked(d.LinearValueHeads * d.LinearHeadWidth * d.LinearHeadWidth));
                Reset();
            }
            else EnsureCapacity(Math.Min(16, d.ContextLength));
        }
        internal void Reset()
        {
            ClearPromptState();
            if (Convolution is not null) Zero(Convolution);
            if (Recurrent is not null) Zero(Recurrent);
            // Cached rows beyond the current position are never read. Resetting
            // the model position is enough; capacity remains reusable on device.
        }
        internal void CapturePromptState()
        {
            if (!recurrent) return;
            ArcBuffer convolutionBuffer = Convolution!;
            ArcBuffer recurrentBuffer = Recurrent!;
            try
            {
                _promptConvolution ??= Lane.AllocateBytes(checked((int)convolutionBuffer.ByteLength));
                _promptRecurrent ??= Lane.AllocateBytes(checked((int)recurrentBuffer.ByteLength));
                Lane.CopyBytes(convolutionBuffer, _promptConvolution, 0, 0, checked((int)convolutionBuffer.ByteLength));
                Lane.CopyBytes(recurrentBuffer, _promptRecurrent, 0, 0, checked((int)recurrentBuffer.ByteLength));
            }
            catch { ClearPromptState(); throw; }
        }
        internal void RestorePromptState()
        {
            if (!recurrent) return;
            if (!HasPromptState) throw new InvalidOperationException("Qwen3.5 prompt state is unavailable.");
            Lane.CopyBytes(_promptConvolution!, Convolution!, 0, 0, checked((int)Convolution!.ByteLength));
            Lane.CopyBytes(_promptRecurrent!, Recurrent!, 0, 0, checked((int)Recurrent!.ByteLength));
        }
        internal void ClearPromptState()
        {
            _promptConvolution?.Dispose();
            _promptRecurrent?.Dispose();
            _promptConvolution = _promptRecurrent = null;
        }
        private void Zero(ArcBuffer buffer)
        {
            int length = checked((int)(buffer.ByteLength / 4));
            Lane.Run("q35a_zero", length, 0, buffer, length);
        }
        internal void EnsureCapacity(int required)
        {
            if (recurrent || required <= _capacity) return;
            int capacity = (int)Math.Min(d.ContextLength, Math.Max(required, Math.Max(16L, 2L * _capacity)));
            long bytes = checked((long)capacity * d.KvHeadCount * d.HeadWidth * 4);
            if (bytes > int.MaxValue || (ulong)bytes > Lane.Device.MaximumAllocationBytes)
                throw new NotSupportedException("Qwen3.5 KV cache exceeds the device allocation limit.");
            if (Lane.AllocatedBytes + 2 * bytes + WorkspaceReserveBytes > Lane.EffectivePhysicalBufferBudgetBytes)
                throw new NotSupportedException("Qwen3.5 KV cache growth exceeds the GPU memory budget; use a shorter context or more devices.");
            ArcBuffer? keys = null, values = null;
            try
            {
                keys = Lane.AllocateBytes((int)bytes); values = Lane.AllocateBytes((int)bytes);
                if (Keys is not null)
                {
                    int oldBytes = checked((int)Keys.ByteLength);
                    Lane.CopyBytes(Keys, keys, 0, 0, oldBytes);
                    Lane.CopyBytes(Values!, values, 0, 0, oldBytes);
                    Lane.Synchronize();
                }
                Keys?.Dispose(); Values?.Dispose();
                Keys = keys; Values = values; keys = values = null; _capacity = capacity;
            }
            finally { keys?.Dispose(); values?.Dispose(); }
        }
        public void Dispose()
        {
            ClearPromptState();
            Convolution?.Dispose(); Recurrent?.Dispose(); Keys?.Dispose(); Values?.Dispose();
        }
    }
}
