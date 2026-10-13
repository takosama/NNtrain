using System.Runtime.InteropServices;
using System.Text;

namespace NNtrain.Arc;

public sealed partial class ArcExecutionLane
{
    private static readonly string[] Qwen35KernelResourceSuffixes =
    [
        ".qwen.cl", ".qwen35_attention.cl", ".qwen35_delta.cl",
        ".qwen35_delta_fused.cl", ".qwen35_linear_fast.cl", ".qwen35_iq.cl", ".qwen35_iq2_packed_decode.cl", ".qwen35_iq2_signed_lookup.cl", ".qwen35_output_head.cl", ".qwen35_iq2_decode132.cl", ".qwen35_iq2_tiles.cl", ".qwen35_lora.cl", ".qwen35_prism.cl",
        ".qwen35_projection_pair.cl", ".qwen35_norm_fast.cl", ".qwen35_prefill_linear.cl", ".qwen35_prefill_xmx.cl", ".qwen35_iq2_packed_prefill.cl", ".qwen35_prefill_xmx_tiles.cl", ".qwen35_vision.cl",
        ".qwen35_vision_attention_fast.cl", ".qwen35_vision_attention_xmx.cl", ".qwen35_vision_linear_fast.cl"
    ];

    private static readonly string[] Qwen35TrainingResourceSuffixes =
        [".qwen35_train.cl", ".qwen35_train_attention.cl", ".qwen35_train_attention_packed.cl", ".qwen35_train_attention_streamed.cl", ".qwen35_train_attention_rowfused.cl", ".qwen35_train_attention_streamed_rowfused.cl", ".qwen35_train_delta.cl", ".qwen35_train_linear.cl", ".qwen35_train_embedding.cl", ".qwen35_train_transpose.cl"];

    private static readonly string[] Qwen35VisionResourceSuffixes =
        [".qwen35_vision.cl", ".qwen35_vision_attention_fast.cl", ".qwen35_vision_attention_xmx.cl", ".qwen35_vision_linear_fast.cl"];

    private nint ProgramForRequestedKernel(string name)
    {
        if (Options.AsrKernelsOnly)
            return name.StartsWith("asr_", StringComparison.Ordinal) ? _program
                : throw new NotSupportedException("This Arc lane only contains ASR kernels.");
        if (Options.Qwen35VisionKernelsOnly)
            return name.StartsWith("q35v_", StringComparison.Ordinal) ? _program
                : throw new NotSupportedException("This Arc lane only contains Qwen3.5 vision kernels.");
        if (!Options.Qwen35InferenceKernelsOnly) return ProgramForKernel(name);
        // Do not silently compile standalone training programs when a caller
        // requests a kernel outside the explicitly selected inference workload.
        if (name.StartsWith("q35s_", StringComparison.Ordinal)
            || (Options.ExperimentalOptimizationKernels && name.StartsWith("q35c_", StringComparison.Ordinal))
            || (Options.Qwen35TrainingKernels && name.StartsWith("q35t_", StringComparison.Ordinal))
            || name.StartsWith("qwen_", StringComparison.Ordinal)
            || name.StartsWith("q35a_", StringComparison.Ordinal)
            || name.StartsWith("q35d_", StringComparison.Ordinal)
            || name.StartsWith("q35l_", StringComparison.Ordinal)
            || name.StartsWith("q35v_", StringComparison.Ordinal)) return _program;
        throw new NotSupportedException("This Arc lane only contains Qwen3.5 inference kernels.");
    }

    private void InitializeProgram()
    {
        int error;
        var sourceText = new StringBuilder();
        var assembly = typeof(ArcExecutionLane).Assembly;
        int resourceCount = 0;
        string[] selectedResources = Options.AsrKernelsOnly ? [".asr_linear.cl"] : Options.Qwen35VisionKernelsOnly
            ? Qwen35VisionResourceSuffixes : Qwen35KernelResourceSuffixes;
        foreach (string resourceName in assembly.GetManifestResourceNames().Where(n => n.EndsWith(".cl", StringComparison.Ordinal)).Order(StringComparer.Ordinal))
        {
            if (!Options.AsrKernelsOnly && resourceName.EndsWith(".asr_linear.cl", StringComparison.Ordinal)) continue;
            if (Options.Qwen35InferenceKernelsOnly
                && !selectedResources.Any(suffix => resourceName.EndsWith(suffix, StringComparison.Ordinal))
                && !(Options.Qwen35TrainingKernels && Qwen35TrainingResourceSuffixes.Any(suffix => resourceName.EndsWith(suffix, StringComparison.Ordinal))))
                continue;
            // Flash has its own compiler policy; never change GEMM/codec
            // register allocation to accommodate a different kernel group.
            if (resourceName.EndsWith(".attention_flash.cl", StringComparison.Ordinal)
                || resourceName.EndsWith(".attention_xmx_products.cl", StringComparison.Ordinal)) continue;
            using Stream resource = assembly.GetManifestResourceStream(resourceName)!;
            using var reader = new StreamReader(resource);
            sourceText.AppendLine(reader.ReadToEnd());
            resourceCount++;
        }
        if (sourceText.Length == 0) throw new InvalidOperationException("Arc OpenCL kernels are missing.");
        if (Options.Qwen35InferenceKernelsOnly && resourceCount != selectedResources.Length + (Options.Qwen35TrainingKernels ? Qwen35TrainingResourceSuffixes.Length : 0))
            throw new InvalidOperationException("The dedicated Qwen3.5 OpenCL kernel resources are incomplete.");
        byte[] source = Encoding.UTF8.GetBytes(sourceText.ToString());
        string buildOptions = "-cl-std=CL1.2 -cl-fp32-correctly-rounded-divide-sqrt";
        buildOptions += $" -DQ35_PROJECTION_WG={Options.Qwen35ProjectionWorkgroupSize}";
        if (Options.Qwen35UnrollQ4) buildOptions += " -DQ35_Q4_UNROLL=1";
        if (Options.Qwen35NativeHalfScale) buildOptions += " -DQ35_NATIVE_HALF=1";
        if (Options.Qwen35VisionFlashAttention) buildOptions += " -DARC_VISION_FLASH=1";
        if (Options.Qwen35DeltaSubgroupRms) buildOptions += " -DARC_DELTA_SUBGROUP_RMS=1";
        if (Options.Qwen35TrainingKernels || !Options.Qwen35InferenceKernelsOnly) buildOptions += " -DARC_QWEN35_TRAINING=1";
        if (Options.ExperimentalOptimizationKernels) buildOptions += " -DARC_OPTIMIZATION_PROBES=1";
        if (Options.Qwen35ResidentIq2Panels) buildOptions += " -DARC_Q35_RESIDENT_IQ2=1";
        if (Options.Qwen35GgufBslmPrefill) buildOptions += " -DARC_Q35_GGUF_BSLM=1";
        if (Options.Mix8_16Int8Linear) buildOptions += " -DARC_INT8_LINEAR=1";
        if (Device.SupportsXmx && Options.XmxMatrices)
            buildOptions += $" -DARC_XMX=1 -DARC_SG={Device.MinimumSubgroupSize}";
        if (Device.Extensions.Split(' ').Contains("cl_intel_subgroup_local_block_io"))
            buildOptions += " -DARC_SLM_BLOCK_IO=1";
        string? cachePath = ProgramCachePath(source, buildOptions);
        if (!TryLoadProgramCache(cachePath, buildOptions))
        {
            GCHandle pin = GCHandle.Alloc(source, GCHandleType.Pinned);
            try { _program = OpenClNative.clCreateProgramWithSource(_context, 1, [pin.AddrOfPinnedObject()], [(nuint)source.Length], out error); }
            finally { pin.Free(); }
            OpenClNative.Check(error, "create program");
            error = OpenClNative.clBuildProgram(_program, 1, [Device.NativeDevice], buildOptions, 0, 0);
            if (error != 0)
            {
                OpenClNative.clGetProgramBuildInfo(_program, Device.NativeDevice, 0x1183, 0, null, out nuint length);
                byte[] log = new byte[checked((int)length)];
                OpenClNative.clGetProgramBuildInfo(_program, Device.NativeDevice, 0x1183, length, log, out _);
                throw new InvalidOperationException($"Arc OpenCL kernel build failed ({error}): {Encoding.UTF8.GetString(log)}");
            }
            SaveProgramCache(cachePath);
        }
    }
}
