using System.Runtime.InteropServices;

namespace NNtrain.Cuda.Interop;

/// <summary>
/// Versioned CUDA tensor primitives used at runtime ownership boundaries.
/// </summary>
public static partial class CudaNativeGateway
{
    /// <summary>
    /// Assigns or accumulates one Float32 scalar entirely on the selected
    /// CUDA stream. The value is a kernel argument, not an H2D buffer copy.
    /// </summary>
    public static int TensorAccumulateScalar(
        int device,
        nint destination,
        float value,
        bool accumulate)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.CudaOutputGradientSeedMinor,
            "CUDA-resident output-gradient seeding");
        return Complete(
            TensorNativeMethods.AccumulateScalar(
                destination,
                value,
                accumulate ? 1 : 0),
            CudaNativeOperation.TensorAccumulateScalar,
            device);
    }

    private static class TensorNativeMethods
    {
        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_tensor_accumulate_scalar",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int AccumulateScalar(
            nint destination,
            float value,
            int accumulate);
    }
}
