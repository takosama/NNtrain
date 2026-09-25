using System.Runtime.InteropServices;
using System.Text;

namespace NNtrain.Arc;

public sealed partial class ArcExecutionLane
{
    private readonly Dictionary<int, nint> _flashPrograms = [];
    private nint _attentionProductsProgram;
    private nint _largeEpilogueProgram;

    // Called under the lane lock. Isolating the program avoids a global
    // register-allocation change to established GEMM/optimizer kernels.
    private nint ProgramForKernel(string name)
    {
        if (name.StartsWith("gemm_xmx_bfp8_epilogue_32x64_", StringComparison.Ordinal)
            || name.StartsWith("gemm_xmx_bfp8_epilogue_16x128_", StringComparison.Ordinal))
        {
            if (!Options.XmxMatrices || !Device.SupportsXmx || Device.MinimumSubgroupSize != 16)
                throw new NotSupportedException("Large XMX tiles require SG16 XMX.");
            if (_largeEpilogueProgram == 0)
                _largeEpilogueProgram = BuildAttentionProgram(".xmx_bfp8_epilogue.cl",
                    "-cl-std=CL1.2 -cl-fp32-correctly-rounded-divide-sqrt -DARC_XMX=1 -DARC_SG=16 -DARC_EPILOGUE_STANDALONE=1 -cl-intel-256-GRF-per-thread");
            return _largeEpilogueProgram;
        }
        if (name.StartsWith("attention_products_", StringComparison.Ordinal)
            || name.StartsWith("attention_xmx_products_", StringComparison.Ordinal))
        {
            if ((!Options.XmxAttentionProducts && !Options.Mix8_16XmxAttentionProducts)
                || !Options.XmxMatrices || !Device.SupportsXmx || Device.MinimumSubgroupSize != 16)
                throw new NotSupportedException("Attention matrix products require an enabled SG16 XMX session.");
            if (_attentionProductsProgram == 0)
                _attentionProductsProgram = BuildAttentionProgram(".attention_xmx_products.cl",
                    "-cl-std=CL1.2 -cl-fp32-correctly-rounded-divide-sqrt -DARC_XMX=1 -DARC_SG=16"
                    + $" -DARC_ATTN_DIRECT={(Options.DirectXmxAttentionProducts ? 1 : 0)}"
                    + (Device.Extensions.Split(' ').Contains("cl_intel_subgroup_local_block_io") ? " -DARC_SLM_BLOCK_IO=1" : ""));
            return _attentionProductsProgram;
        }
        if (!name.StartsWith("attention_flash_", StringComparison.Ordinal)) return _program;
        if (!Options.FlashAttention || !Options.XmxMatrices || !Device.SupportsXmx || Device.MinimumSubgroupSize != 16)
            throw new NotSupportedException("FlashAttention requires an enabled SG16 XMX session.");
        int depth = name.Contains("_d16_", StringComparison.Ordinal) ? 16
            : name.Contains("_d32_", StringComparison.Ordinal) ? 32
            : name.Contains("_d64_", StringComparison.Ordinal) ? 64 : 0;
        if (_flashPrograms.TryGetValue(depth, out nint existing)) return existing;
        string options = "-cl-std=CL1.2 -cl-fp32-correctly-rounded-divide-sqrt -DARC_XMX=1 -DARC_SG=16 -DARC_FLASH=1"
            + $" -DARC_FLASH_XMX={(Options.FlashAttentionXmxProducts ? 1 : 0)}"
            + $" -DARC_FLASH_ASYNC={(Options.FlashAttentionXmxProducts && Options.FlashAttentionAsyncCopy ? 1 : 0)}"
            + $" -DARC_FLASH_DEPTH={depth}";
        if (Options.FlashAttentionLargeRegisters && Options.FlashAttentionXmxProducts)
            options += " -cl-intel-256-GRF-per-thread";
        nint result = BuildAttentionProgram(".attention_flash.cl", options);
        _flashPrograms.Add(depth, result);
        return result;
    }

    private nint BuildAttentionProgram(string suffix, string options)
    {
        var assembly = typeof(ArcExecutionLane).Assembly;
        string resourceName = assembly.GetManifestResourceNames().Single(n => n.EndsWith(suffix, StringComparison.Ordinal));
        using Stream resource = assembly.GetManifestResourceStream(resourceName)!;
        using var reader = new StreamReader(resource);
        byte[] source = Encoding.UTF8.GetBytes(reader.ReadToEnd());
        GCHandle pin = GCHandle.Alloc(source, GCHandleType.Pinned);
        nint program;
        try
        {
            program = OpenClNative.clCreateProgramWithSource(_context, 1, [pin.AddrOfPinnedObject()], [(nuint)source.Length], out int error);
            OpenClNative.Check(error, "create FlashAttention program");
        }
        finally { pin.Free(); }
        try
        {
            int error = OpenClNative.clBuildProgram(program, 1, [Device.NativeDevice], options, 0, 0);
            if (error != 0)
            {
                OpenClNative.clGetProgramBuildInfo(program, Device.NativeDevice, 0x1183, 0, null, out nuint length);
                byte[] log = new byte[checked((int)length)];
                OpenClNative.clGetProgramBuildInfo(program, Device.NativeDevice, 0x1183, length, log, out _);
                throw new InvalidOperationException($"Arc FlashAttention build failed ({error}): {Encoding.UTF8.GetString(log)}");
            }
            return program;
        }
        catch { OpenClNative.clReleaseProgram(program); throw; }
    }
}
