using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

public sealed partial class Qwen35QuantizedModel
{
    internal static int AlignedTransposeChunkRows(int rows, long bytesPerRow,
        long scratchBudgetBytes, int rowTile)
    {
        if (rows < 1 || bytesPerRow < 1 || rowTile < 1)
            throw new ArgumentOutOfRangeException(nameof(rows));
        int capacity = checked((int)Math.Max(1L, Math.Min(rows, scratchBudgetBytes / bytesPerRow)));
        // A short tail launches another complete quantized-weight traversal.
        // Align the main chunks to the kernel's row tile when the budget permits.
        return capacity >= rowTile ? capacity / rowTile * rowTile : capacity;
    }

    private sealed class Matrix : IDisposable
    {
        internal readonly ArcExecutionLane Lane;
        internal readonly string Name;
        internal readonly int StorageBytes, OutputWidth;
        private readonly ArcBuffer _encoded;
        private readonly ArcBuffer? _residentGrid, _residentScales, _residentBlockD;
        private readonly uint _type;
        private readonly string _quantization;
        internal readonly int _inputWidth;
        private readonly Qwen35QuantizedKernel _kernel;
        private readonly int _transposeRows;
        private readonly int _transposeOctetRows;
        private readonly int _forwardRows;
        private readonly int _q5ForwardRows;
        private readonly bool _trainingIq2Bf16XmxForward;
        private readonly bool _trainingIq2Fp16XmxForward;
        private readonly bool _trainingIq2RollingTranspose;
        private readonly string _projectionKernel, _fusedProjectionKernel;
        private readonly string _rowProjectionKernelPrefix;
        private readonly bool _inferenceXmxPrefill;
        private readonly bool _inferenceXmxPackedPrefill;
        private readonly bool _inferenceXmxFactoredPrefill;
        private readonly bool _inferenceXmxGgufBslmPrefill;
        private readonly bool _iq2TiledForward, _iq2TiledBackward;
        private readonly bool PairedProjection;
        private readonly bool _iq2DecodePair4;
        private readonly bool _iq2DecodeLevel3;
        private readonly bool _packedIq2;
        private readonly bool PairedLoraProjection;
        internal bool SupportsFusedLora => _kernel == Qwen35QuantizedKernel.Subgroup
            && _type is not (Qwen35Gguf.PQ20Type or Qwen35Gguf.PTQ10Type or Qwen2Gguf.BF16Type);
        internal bool SupportsSharedPrefillProjection => !Lane.Options.Qwen35TrainingKernels
            && (_forwardRows > 1 || _inferenceXmxPrefill || _iq2TiledForward) && _kernel == Qwen35QuantizedKernel.Subgroup
            && _type is Qwen2Gguf.Q4KType or Qwen35Gguf.IQ2SType or Qwen35Gguf.IQ3SType;
        internal bool IsIq2S => _type == Qwen35Gguf.IQ2SType;
        internal uint StorageType => _type;
        internal bool SupportsSplitOutputHead => _type == Qwen35Gguf.Q5KType
            && _kernel == Qwen35QuantizedKernel.Subgroup && !Lane.Options.Qwen35TrainingKernels;
        internal ArcBuffer EncodedOutputHead => _encoded;
        internal Matrix(ArcExecutionLane lane, GgufTensorInfo info, byte[] payload, Qwen35QuantizedKernel kernel, int transposeRows, int forwardRows, int q5ForwardRows, bool trainingIq2Bf16XmxForward, bool trainingIq2Fp16XmxForward, bool trainingIq2RollingTranspose, int transposeOctetRows, bool inferenceXmxPrefill = false, bool inferenceXmxPackedPrefill = false, bool inferenceXmxFactoredPrefill = false, bool inferenceResidentIq2Panels = false, bool inferenceXmxGgufBslmPrefill = false, bool iq2TiledForward = false, bool iq2TiledBackward = false, bool inferenceIq2DecodePair4 = false, bool inferenceIq2DecodeLevel3 = false, bool inferencePackedIq2Decode = false)
        {
            _transposeRows = transposeRows;
            _forwardRows = forwardRows;
            _rowProjectionKernelPrefix = lane.Options.Qwen35TrainingKernels ? "q35t_linear_" : "q35l_prefill_linear_";
            _q5ForwardRows = q5ForwardRows;
            _trainingIq2Bf16XmxForward = trainingIq2Bf16XmxForward;
            _trainingIq2Fp16XmxForward = trainingIq2Fp16XmxForward;
            _trainingIq2RollingTranspose = trainingIq2RollingTranspose;
            Lane = lane; Name = info.Name; StorageBytes = payload.Length; _type = info.Type;
            _quantization = _type switch
            {
                Qwen2Gguf.Q4KType => "q4_k", Qwen35Gguf.Q5KType => "q5_k",
                Qwen2Gguf.Q6KType => "q6_k", Qwen35Gguf.IQ2SType => "iq2_s",
                Qwen35Gguf.IQ3SType => "iq3_s",
                Qwen35Gguf.PQ20Type => "pq2_0", Qwen35Gguf.PTQ10Type => "ptq1_0",
                Qwen2Gguf.BF16Type => "bf16",
                _ => throw new NotSupportedException($"Unsupported matrix storage type {_type}.")
            };
            _inputWidth = checked((int)info.Shape[0]); OutputWidth = checked((int)info.Shape[1]);
            bool subgroup = lane.Options.XmxMatrices && lane.Device.SupportsXmx
                && lane.Device.MinimumSubgroupSize == 16
                && lane.Device.Extensions.Split(' ').Contains("cl_intel_subgroups");
            _transposeOctetRows = subgroup ? transposeOctetRows : 0;
            _kernel = kernel == Qwen35QuantizedKernel.Auto
                // PQ2's cooperative reduction is faster on the measured Arc
                // workload; PTQ1 benefits from its dedicated SG16 block loop.
                ? (subgroup && _type != Qwen35Gguf.PQ20Type
                    ? Qwen35QuantizedKernel.Subgroup : Qwen35QuantizedKernel.Cooperative) : kernel;
            _inferenceXmxPrefill = (inferenceXmxPrefill || inferenceResidentIq2Panels
                || (inferenceXmxGgufBslmPrefill && _type == Qwen35Gguf.IQ2SType)) && !lane.Options.Qwen35TrainingKernels
                && _kernel == Qwen35QuantizedKernel.Subgroup
                && lane.Device.Extensions.Split(' ').Contains("cl_khr_fp16")
                && _type is Qwen2Gguf.Q4KType or Qwen35Gguf.IQ2SType or Qwen35Gguf.IQ3SType
                && HasFiniteF16DecodedWeightRange(payload, _type);
            _inferenceXmxPackedPrefill = inferenceXmxPackedPrefill && _inferenceXmxPrefill;
            _inferenceXmxFactoredPrefill = inferenceXmxFactoredPrefill && _inferenceXmxPrefill
                && _type is Qwen35Gguf.IQ2SType or Qwen35Gguf.IQ3SType;
            _inferenceXmxGgufBslmPrefill = inferenceXmxGgufBslmPrefill && _inferenceXmxPrefill
                && _type == Qwen35Gguf.IQ2SType;
            _iq2TiledForward = iq2TiledForward && IsIq2S && lane.SupportsIq2TiledProjection
                && _kernel == Qwen35QuantizedKernel.Subgroup && !trainingIq2Bf16XmxForward && !trainingIq2Fp16XmxForward;
            _iq2TiledBackward = iq2TiledBackward && IsIq2S && lane.SupportsIq2TiledProjection
                && _kernel == Qwen35QuantizedKernel.Subgroup;
            if (_kernel == Qwen35QuantizedKernel.Subgroup && !subgroup)
                throw new NotSupportedException("The Qwen3.5 subgroup kernel requires Intel SG16 support.");
            PairedProjection = (lane.Options.Qwen35PairedProjection
                || (lane.Options.Qwen35PairedProjectionTypes & (_type switch {
                    Qwen35Gguf.IQ2SType => 1, Qwen2Gguf.Q4KType => 2, Qwen35Gguf.IQ3SType => 4, _ => 0 })) != 0)
                && _type is Qwen2Gguf.Q4KType or Qwen35Gguf.IQ2SType or Qwen35Gguf.IQ3SType;
            _projectionKernel = $"q35l_{_quantization}_sg16" + (PairedProjection ? "_pair" : "");
            _iq2DecodePair4 = inferenceIq2DecodePair4 && !lane.Options.Qwen35TrainingKernels
                && _kernel == Qwen35QuantizedKernel.Subgroup && _type == Qwen35Gguf.IQ2SType;
            _iq2DecodeLevel3 = inferenceIq2DecodeLevel3 && !lane.Options.Qwen35TrainingKernels
                && _kernel == Qwen35QuantizedKernel.Subgroup && _type == Qwen35Gguf.IQ2SType;
            PairedLoraProjection = PairedProjection && lane.Options.Qwen35PairedLoraProjection;
            _fusedProjectionKernel = $"q35l_{_quantization}_sg16" + (PairedLoraProjection ? "_pair" : "") + "_lora";
            if (inferenceResidentIq2Panels && _inferenceXmxPrefill && CanUseResidentIq2Panels(info))
            {
                int columns = checked((OutputWidth + 15) / 16 * 16);
                int gridBytes = checked(_inputWidth / 32 * columns * 12);
                int scaleBytes = checked(_inputWidth / 32 * columns);
                int coefficientBytes = checked(_inputWidth / 256 * columns * sizeof(ushort));
                try
                {
                    _residentGrid = lane.AllocateBytes(gridBytes);
                    _residentScales = lane.AllocateBytes(scaleBytes);
                    _residentBlockD = lane.AllocateBytes(coefficientBytes);
                    using ArcBuffer encoded = lane.UploadRaw(payload);
                    lane.Run("q35l_prefill_xmx_pack_iq2_s_grid3", (long)_inputWidth / 32 * OutputWidth, 0,
                        encoded, _residentGrid, _residentScales, _residentBlockD, _inputWidth, OutputWidth);
                    // Publish the panels before releasing their temporary GGUF input.
                    lane.Synchronize();
                    StorageBytes = checked(gridBytes + scaleBytes + coefficientBytes);
                    _encoded = null!;
                }
                catch
                {
                    _residentGrid?.Dispose(); _residentScales?.Dispose(); _residentBlockD?.Dispose();
                    throw;
                }
            }
            else
            {
                _packedIq2 = inferencePackedIq2Decode && CanUseResidentIq2Panels(info)
                    && _kernel == Qwen35QuantizedKernel.Subgroup && SupportsIq2PackedDecode(lane)
                    && !trainingIq2Bf16XmxForward && !trainingIq2Fp16XmxForward
                    && (!_inferenceXmxPrefill || _inferenceXmxGgufBslmPrefill)
                    && !_iq2DecodePair4 && !_iq2DecodeLevel3;
                // Repack on the host before the blocking upload: one resident
                // allocation, identical bytes per tensor, no GPU copy peak.
                _encoded = lane.UploadRaw(_packedIq2 ? PackIq2DecodeBytes(payload, _inputWidth, OutputWidth) : payload);
                if (_packedIq2)
                {
                    // Resolve routes once at load time; token generation does
                    // not repeat device checks, string splits or shape checks.
                    _projectionKernel = "q35l_iq2_s_packed82_sg16" + (PairedProjection ? "_pair" : "");
                    _fusedProjectionKernel = "q35l_iq2_s_packed82_sg16" + (PairedLoraProjection ? "_pair" : "") + "_lora";
                }
            }
        }
        internal ArcBuffer Forward(ArcBuffer input, ArcBuffer zeroBias, int rows = 1)
        {
            if (input.ByteLength < 4L * rows * _inputWidth) throw new ArgumentException("Qwen3.5 projection width mismatch.");
            ArcBuffer output = Lane.Allocate(checked(rows * OutputWidth));
            try
            {
                if (_residentGrid is not null)
                {
                    if (rows >= 16)
                    {
                        int elements = checked((rows + 7) / 8 * 8 * _inputWidth);
                        using ArcBuffer high = Lane.AllocateBytes(checked(elements * sizeof(ushort)));
                        using ArcBuffer low = Lane.AllocateBytes(checked(elements * sizeof(ushort)));
                        using ArcBuffer rangeStatus = Lane.Allocate(1);
                        Lane.Run("q35a_zero", 1, 0, rangeStatus, 1);
                        Lane.Run("q35l_prefill_xmx_pack_input_f16x2", elements, 0, input, high, low, rangeStatus, rows, _inputWidth);
                        Lane.Run2D("q35l_prefill_xmx_iq2_s_grid3_bslm", ((long)OutputWidth + 63) / 64 * 16,
                            ((long)rows + 127) / 128 * 16, 16, 16, high, low, _residentGrid, _residentScales!, _residentBlockD!,
                            zeroBias, output, rows, _inputWidth, OutputWidth, input, rangeStatus);
                    }
                    else
                    {
                        int group = PairedProjection ? 32 : Lane.Options.Qwen35ProjectionWorkgroupSize;
                        long global = PairedProjection
                            ? ((long)rows * ((OutputWidth + 1) / 2) + 1) / 2 * 32
                            : ((long)rows * OutputWidth + group / 16 - 1) / (group / 16) * group;
                        Lane.Run(PairedProjection ? "q35l_iq2_s_grid3_sg16_pair" : "q35l_iq2_s_grid3_sg16_fast",
                            global, group, input, _residentGrid, _residentScales!, _residentBlockD!,
                            zeroBias, output, rows, _inputWidth, OutputWidth);
                    }
                }
                // Training uses block-factored integer panels. Inference keeps
                // the original K16 coefficient order to preserve model logits.
                else if (Lane.Options.Qwen35TrainingKernels && _iq2TiledForward && !_inferenceXmxPrefill && rows >= 128
                    && Lane.AllocatedBytes + Lane.Iq2TiledForwardWorkspaceBytes(rows, _inputWidth, OutputWidth)
                        + 16L * 1024 * 1024 < Lane.EffectivePhysicalBufferBudgetBytes)
                {
                    Lane.Iq2TiledForward(input, _encoded, zeroBias, output, rows, _inputWidth, OutputWidth);
                }
                else if (_inferenceXmxPrefill && rows >= 16)
                {
                    if (_inferenceXmxGgufBslmPrefill)
                    {
                        // Row-major A removes the long stride in each XMX
                        // reduction while retaining the old K16 arithmetic.
                        bool rowMajor = _iq2TiledForward && rows >= 512;
                        int elements = checked((rows + 7) / 8 * 8 * _inputWidth);
                        using ArcBuffer high = Lane.AllocateBytes(checked(elements * sizeof(ushort)));
                        using ArcBuffer low = Lane.AllocateBytes(checked(elements * sizeof(ushort)));
                        using ArcBuffer rangeStatus = Lane.Allocate(1);
                        Lane.Run("q35a_zero", 1, 0, rangeStatus, 1);
                        Lane.Run(rowMajor ? "q35l_prefill_xmx_pack_input_f16x2_rowmajor" : "q35l_prefill_xmx_pack_input_f16x2",
                            elements, 0, input, high, low, rangeStatus, rows, _inputWidth);
                        Lane.Run2D((rowMajor ? "q35l_prefill_xmx_iq2_s_gguf_bslm_k32r" : "q35l_prefill_xmx_iq2_s_gguf_bslm") + (_packedIq2 ? "_packed82" : ""), ((long)OutputWidth + 63) / 64 * 16,
                            ((long)rows + 127) / 128 * 16, 16, 16, high, low, _encoded, zeroBias, output,
                            rows, _inputWidth, OutputWidth, input, rangeStatus);
                    }
                    else if (_inferenceXmxFactoredPrefill && rows >= 384
                        && checked((long)_inputWidth * ((OutputWidth + 15L) / 16 * 16) * 9 / 4) <= 512L * 1024 * 1024)
                    {
                        int inputElements = checked((rows + 7) / 8 * 8 * _inputWidth);
                        int weightElements = checked((OutputWidth + 15) / 16 * 16 * _inputWidth);
                        using ArcBuffer high = Lane.AllocateBytes(checked(inputElements * sizeof(ushort)));
                        using ArcBuffer low = Lane.AllocateBytes(checked(inputElements * sizeof(ushort)));
                        using ArcBuffer grid = Lane.AllocateBytes(checked(weightElements * sizeof(ushort)));
                        using ArcBuffer scales = Lane.Allocate(weightElements / 16);
                        using ArcBuffer rangeStatus = Lane.Allocate(1);
                        Lane.Run("q35a_zero", 1, 0, rangeStatus, 1);
                        Lane.Run("q35l_prefill_xmx_pack_input_f16x2", inputElements, 0, input, high, low, rangeStatus, rows, _inputWidth);
                        Lane.Run($"q35l_prefill_xmx_pack_{_quantization}_factored", (long)_inputWidth * OutputWidth / 8, 0,
                            _encoded, grid, scales, _inputWidth, OutputWidth);
                        Lane.Run2D("q35l_prefill_xmx_packed_factored", ((long)OutputWidth + 63) / 64 * 16,
                            ((long)rows + 127) / 128 * 16, 16, 16, high, low, grid, scales, zeroBias, output,
                            rows, _inputWidth, OutputWidth, input, _encoded, rangeStatus,
                            _type == Qwen35Gguf.IQ2SType ? 0 : 1);
                    }
                    else if (_inferenceXmxPackedPrefill && rows >= 128
                        && checked((long)_inputWidth * ((OutputWidth + 15L) / 16 * 16) * 4) <= 512L * 1024 * 1024)
                    {
                        int inputElements = checked((rows + 7) / 8 * 8 * _inputWidth);
                        int weightElements = checked((OutputWidth + 15) / 16 * 16 * _inputWidth);
                        using ArcBuffer high = Lane.AllocateBytes(checked(inputElements * sizeof(ushort)));
                        using ArcBuffer low = Lane.AllocateBytes(checked(inputElements * sizeof(ushort)));
                        using ArcBuffer weightHigh = Lane.AllocateBytes(checked(weightElements * sizeof(ushort)));
                        using ArcBuffer weightLow = Lane.AllocateBytes(checked(weightElements * sizeof(ushort)));
                        using ArcBuffer rangeStatus = Lane.Allocate(1);
                        Lane.Run("q35a_zero", 1, 0, rangeStatus, 1);
                        Lane.Run("q35l_prefill_xmx_pack_input_f16x2", inputElements, 0, input, high, low, rangeStatus, rows, _inputWidth);
                        Lane.Run($"q35l_prefill_xmx_pack_{_quantization}_f16x2", (long)_inputWidth * OutputWidth / 8, 0,
                            _encoded, weightHigh, weightLow, _inputWidth, OutputWidth);
                        Lane.Run2D("q35l_prefill_xmx_packed_f16x2", ((long)OutputWidth + 63) / 64 * 16,
                            ((long)rows + 127) / 128 * 16, 16, 16, high, low, weightHigh, weightLow, zeroBias, output,
                            rows, _inputWidth, OutputWidth, input, _encoded, rangeStatus,
                            _type == Qwen35Gguf.IQ2SType ? 0 : _type == Qwen35Gguf.IQ3SType ? 1 : 2);
                    }
                    else
                    {
                        int elements = checked(rows * _inputWidth);
                        using ArcBuffer high = Lane.AllocateBytes(checked(elements * sizeof(ushort)));
                        using ArcBuffer low = Lane.AllocateBytes(checked(elements * sizeof(ushort)));
                        using ArcBuffer rangeStatus = Lane.Allocate(1);
                        Lane.Run("q35a_zero", 1, 0, rangeStatus, 1);
                        Lane.Run("q35l_prefill_xmx_input_f16x2", elements, 0, input, high, low, rangeStatus, elements);
                        Lane.Run2D($"q35l_prefill_xmx_{_quantization}_" + (_inferenceXmxFactoredPrefill ? "factored" : "f16x2"),
                            ((long)OutputWidth + 63) / 64 * 16, ((long)rows + 127) / 128 * 16,
                            16, 16, high, low, _encoded, zeroBias, output, rows, _inputWidth, OutputWidth, input, rangeStatus);
                    }
                }
                else if ((_trainingIq2Bf16XmxForward || _trainingIq2Fp16XmxForward)
                    && _kernel == Qwen35QuantizedKernel.Subgroup
                    && IsIq2S && rows > 1)
                {
                    using ArcBuffer input16 = Lane.AllocateBytes(checked(rows * _inputWidth * sizeof(ushort)));
                    Lane.Run(_trainingIq2Fp16XmxForward ? "q35t_iq2_input_f16" : "q35t_iq2_input_bf16",
                        (long)rows * _inputWidth, 0, input, input16, checked(rows * _inputWidth));
                    if (_trainingIq2Fp16XmxForward && rows >= 128)
                        Lane.Run2D("q35t_linear_iq2_s_xmx_f16_r128_n64",
                            ((long)OutputWidth + 63) / 64 * 16, ((long)rows + 127) / 128 * 16,
                            16, 16, input16, _encoded, zeroBias, output, rows, _inputWidth, OutputWidth);
                    else
                        Lane.Run2D(_trainingIq2Fp16XmxForward
                                ? "q35t_linear_iq2_s_xmx_f16" : "q35t_linear_iq2_s_xmx_bf16",
                            ((long)OutputWidth + 31) / 32 * 16, ((long)rows + 63) / 64 * 8,
                            16, 8, input16, _encoded, zeroBias, output, rows, _inputWidth, OutputWidth);
                }
                else if (_kernel == Qwen35QuantizedKernel.Subgroup
                    && _type == Qwen35Gguf.Q5KType && _q5ForwardRows > 1 && rows > 1)
                    Lane.Run($"q35t_linear_q5_k_rows{_q5ForwardRows}",
                        ((((long)rows + _q5ForwardRows - 1) / _q5ForwardRows * OutputWidth + 1) / 2) * 32, 32,
                        input, _encoded, zeroBias, output, rows, _inputWidth, OutputWidth);
                else if (_kernel == Qwen35QuantizedKernel.Subgroup && _forwardRows > 1 && rows > 1
                    && _type is Qwen2Gguf.Q4KType or Qwen35Gguf.IQ2SType or Qwen35Gguf.IQ3SType)
                    Lane.Run($"{_rowProjectionKernelPrefix}{_quantization}_rows{_forwardRows}" + (_packedIq2 ? "_packed82" : ""),
                        ((((long)rows + _forwardRows - 1) / _forwardRows * OutputWidth + 1) / 2) * 32, 32,
                        input, _encoded, zeroBias, output, rows, _inputWidth, OutputWidth);
                else if (_iq2DecodeLevel3 && rows == 1)
                    Lane.Run("q35l_iq2_s_sg16_levels3", ((long)((OutputWidth + 1) / 2) + 1) / 2 * 32,
                        32, input, _encoded, zeroBias, output, rows, _inputWidth, OutputWidth);
                else if (_iq2DecodePair4 && rows == 1)
                    Lane.Run("q35l_iq2_s_sg16_pair4", ((long)((OutputWidth + 3) / 4) + 1) / 2 * 32,
                        32, input, _encoded, zeroBias, output, rows, _inputWidth, OutputWidth);
                else if (_kernel == Qwen35QuantizedKernel.Subgroup && PairedProjection)
                    Lane.Run(_projectionKernel, ((long)rows * ((OutputWidth + 1) / 2) + 1) / 2 * 32,
                        32, input, _encoded, zeroBias, output, rows, _inputWidth, OutputWidth);
                else if (_kernel == Qwen35QuantizedKernel.Subgroup)
                    Lane.Run(_projectionKernel,
                        ((long)rows * OutputWidth + Lane.Options.Qwen35ProjectionWorkgroupSize / 16 - 1)
                            / (Lane.Options.Qwen35ProjectionWorkgroupSize / 16) * Lane.Options.Qwen35ProjectionWorkgroupSize,
                        Lane.Options.Qwen35ProjectionWorkgroupSize, input, _encoded, zeroBias, output, rows, _inputWidth, OutputWidth);
                else if (_kernel == Qwen35QuantizedKernel.Cooperative)
                    Lane.Run2D($"q35l_{_quantization}_coop64",
                        (long)OutputWidth * 64, rows, 64, 1, input, _encoded, zeroBias, output, rows, _inputWidth, OutputWidth);
                else
                    Lane.Run(_type is Qwen2Gguf.Q4KType or Qwen2Gguf.Q6KType
                        ? $"qwen_linear_{_quantization}" : $"q35l_{_quantization}_reference",
                        (long)rows * OutputWidth, 0, input, _encoded, zeroBias, output, rows, _inputWidth, OutputWidth);
                return output;
            }
            catch { output.Dispose(); throw; }
        }
        private static bool HasFiniteF16DecodedWeightRange(byte[] payload, uint type)
        {
            int blockBytes = type == Qwen35Gguf.IQ2SType ? 82 : type == Qwen35Gguf.IQ3SType ? 110 : 144;
            for (int offset = 0; offset < payload.Length; offset += blockBytes)
            {
                float d = MathF.Abs((float)BitConverter.UInt16BitsToHalf((ushort)(payload[offset] | payload[offset + 1] << 8)));
                // Bounds cover every scale, grid entry and sign in the block.
                float bound = type == Qwen35Gguf.IQ2SType ? d * 166.625f : d * 1024f;
                if (type == Qwen2Gguf.Q4KType)
                {
                    float minimum = MathF.Abs((float)BitConverter.UInt16BitsToHalf((ushort)(payload[offset + 2] | payload[offset + 3] << 8)));
                    bound = d * 945f + minimum * 63f;
                }
                if (!float.IsFinite(bound) || bound > 65504f) return false;
            }
            return true;
        }
        internal ArcBuffer ForwardFusedLora(ArcBuffer input, ArcBuffer zeroBias, ArcBuffer z, Qwen35LoraMatrix adapter, int rows = 1)
        {
            ArcBuffer output = Lane.Allocate(checked(rows * OutputWidth));
            try
            {
                if (_residentGrid is not null)
                {
                    // Resident pairing retains bitwise original base and LoRA sums.
                    int residentGroup = PairedProjection ? 32 : Lane.Options.Qwen35ProjectionWorkgroupSize;
                    long residentGlobal = PairedProjection
                        ? ((long)rows * ((OutputWidth + 1) / 2) + 1) / 2 * 32
                        : ((long)rows * OutputWidth + residentGroup / 16 - 1) / (residentGroup / 16) * residentGroup;
                    Lane.Run(PairedProjection ? "q35l_iq2_s_grid3_sg16_pair_lora" : "q35l_iq2_s_grid3_sg16_fast_lora", residentGlobal,
                        residentGroup, input, _residentGrid, _residentScales!, _residentBlockD!, zeroBias, output,
                        rows, _inputWidth, OutputWidth, z, adapter.B, adapter.Rank, adapter.Scale);
                    return output;
                }
                int group = PairedLoraProjection ? 32 : Lane.Options.Qwen35ProjectionWorkgroupSize;
                long work = PairedLoraProjection ? ((long)rows * ((OutputWidth + 1) / 2) + 1) / 2 * 32
                    : ((long)rows * OutputWidth + group / 16 - 1) / (group / 16) * group;
                Lane.Run(_fusedProjectionKernel, work,
                    group, input, _encoded, zeroBias, output, rows, _inputWidth, OutputWidth,
                    z, adapter.B, adapter.Rank, adapter.Scale);
                return output;
            }
            catch { output.Dispose(); throw; }
        }
        internal void BackwardInput(ArcBuffer dy, ArcBuffer dx, int rows, long scratchBudgetBytes)
        {
            using ArcBuffer? restored = _packedIq2 ? UnpackIq2Decode(Lane, _encoded, _inputWidth, OutputWidth) : null;
            ArcBuffer encoded = restored ?? _encoded;
            if (_iq2TiledBackward && rows >= 128 && OutputWidth % 16 == 0 && scratchBudgetBytes >= 128L * 1024 * 1024
                && Lane.AllocatedBytes + Math.Min(scratchBudgetBytes, Lane.Iq2TiledBackwardWorkspaceBytes(rows, _inputWidth, OutputWidth))
                    + 16L * 1024 * 1024 < Lane.EffectivePhysicalBufferBudgetBytes)
            {
                Lane.Iq2TiledBackward(dy, encoded, dx, rows, _inputWidth, OutputWidth, workspaceBudgetBytes: scratchBudgetBytes);
                return;
            }
            const int tile = 1024;
            int splits = (OutputWidth + tile - 1) / tile;
            if (scratchBudgetBytes < sizeof(float)) throw new ArgumentOutOfRangeException(nameof(scratchBudgetBytes));
            long bytesPerRow = checked((long)splits * _inputWidth * sizeof(float));
            if (_trainingIq2RollingTranspose && IsIq2S && _transposeOctetRows == 8
                && splits > 1 && checked((long)rows * bytesPerRow) > scratchBudgetBytes
                && scratchBudgetBytes >= (long)_inputWidth * sizeof(float))
            {
                int rollingChunkRows = AlignedTransposeChunkRows(rows,
                    checked((long)_inputWidth * sizeof(float)), scratchBudgetBytes, 8);
                for (int first = 0; first < rows; first += rollingChunkRows)
                {
                    int count = Math.Min(rollingChunkRows, rows - first);
                    int elements = checked(count * _inputWidth);
                    using ArcBuffer rolling = Lane.Allocate(elements);
                    Lane.Run("q35a_zero", elements, 0, rolling, elements);
                    for (int split = 0; split < splits; split++)
                        Lane.Run("q35t_xpose_iq2_s_vec8_rows8_rolling",
                            ((long)count + 7) / 8 * (_inputWidth / 8), 32,
                            dy, encoded, rolling, count, first, _inputWidth,
                            OutputWidth, split, tile);
                    Lane.Run("q35t_add_offset", elements, 0, rolling, dx,
                        elements, checked(first * _inputWidth));
                }
                return;
            }
            if (checked((long)rows * bytesPerRow) <= scratchBudgetBytes)
            {
                // Keep the original fast path and its accumulation order.
                using ArcBuffer fullPartial = Lane.Allocate(checked(rows * splits * _inputWidth));
                RunTranspose(dy, fullPartial, rows);
                Lane.Run("q35t_xpose_reduce", (long)rows * _inputWidth, 0,
                    fullPartial, dx, rows, _inputWidth, splits);
                return;
            }
            int rowTile = _transposeOctetRows != 0 ? _transposeOctetRows : _transposeRows;
            int chunkRows = AlignedTransposeChunkRows(rows, bytesPerRow, scratchBudgetBytes, rowTile);
            // Row tiles leave every token's quantized transpose and split
            // reduction unchanged, while bounding the vocabulary-head scratch.
            for (int row = 0; row < rows; row += chunkRows)
            {
                int count = Math.Min(chunkRows, rows - row);
                int dyElements = checked(count * OutputWidth);
                int dxElements = checked(count * _inputWidth);
                using ArcBuffer dyTile = Lane.Allocate(dyElements);
                Lane.CopyBytes(dy, dyTile, checked(row * OutputWidth * sizeof(float)), 0,
                    checked(dyElements * sizeof(float)));
                using ArcBuffer partial = Lane.Allocate(checked(count * splits * _inputWidth));
                using ArcBuffer dxTile = Lane.Allocate(dxElements);
                Lane.Run("q35a_zero", dxElements, 0, dxTile, dxElements);
                RunTranspose(dyTile, partial, count);
                Lane.Run("q35t_xpose_reduce", (long)count * _inputWidth, 0,
                    partial, dxTile, count, _inputWidth, splits);
                Lane.Run("q35t_add_offset", dxElements, 0, dxTile, dx, dxElements,
                    checked(row * _inputWidth));
            }

            void RunTranspose(ArcBuffer source, ArcBuffer partial, int count)
            {
                if (_transposeOctetRows != 0)
                    Lane.Run("q35t_xpose_" + _quantization + "_vec8_rows" + _transposeOctetRows,
                        ((long)count + _transposeOctetRows - 1) / _transposeOctetRows * splits * (_inputWidth / 8), 32,
                        source, encoded, partial, count, _inputWidth, OutputWidth, splits, tile);
                else
                {
                    string suffix = _transposeRows == 1 ? "" : "_rows" + _transposeRows;
                    Lane.Run("q35t_xpose_" + _quantization + suffix,
                        ((long)count + _transposeRows - 1) / _transposeRows * splits * _inputWidth, 128,
                        source, encoded, partial, count, _inputWidth, OutputWidth, splits, tile);
                }
            }
        }
        internal ArcBuffer Embedding(int tokenId)
        {
            ArcBuffer output = Lane.Allocate(_inputWidth);
            try
            {
                using ArcBuffer id = Lane.UploadRaw(new[] { tokenId });
                Lane.Run(_type is Qwen2Gguf.Q4KType or Qwen2Gguf.Q6KType
                    ? $"qwen_embedding_{_quantization}" : $"q35l_{_quantization}_embedding",
                    _inputWidth, 0, _encoded, id, output, _inputWidth);
                return output;
            }
            catch { output.Dispose(); throw; }
        }
        internal ArcBuffer Embedding(IReadOnlyList<int> tokens, int count)
        {
            ArcBuffer output = Lane.Allocate(checked(count * _inputWidth));
            try
            {
                using ArcBuffer ids = Lane.UploadRaw(tokens.Take(count).ToArray());
                Lane.Run("q35t_embedding_" + _quantization, (long)count * _inputWidth, 128,
                    _encoded, ids, output, count, _inputWidth);
                return output;
            }
            catch { output.Dispose(); throw; }
        }
        public void Dispose()
        {
            _encoded?.Dispose();
            _residentGrid?.Dispose(); _residentScales?.Dispose(); _residentBlockD?.Dispose();
        }
    }
}
