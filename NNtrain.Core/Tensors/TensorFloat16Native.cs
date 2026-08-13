using System.Runtime.InteropServices;

namespace NNtrain;

/// <summary>
/// Optional Windows x64 F16C accelerator for dense Float16 linear kernels.
/// The managed AVX2 codec remains the portable implementation; this shim is
/// loaded only when both the local native payload and the CPU support F16C.
/// </summary>
internal static partial class TensorFloat16Native
{
    private const string LibraryName = "NNtrain.F16C";
    private static readonly bool Available = ProbeAvailability();
    private static int _lastNativeFailure;

    internal static bool IsAvailable
        => Available && Volatile.Read(ref _lastNativeFailure) == 0;

    internal static void DisableAfterFailure()
        => Volatile.Write(ref _lastNativeFailure, 1);

    private static bool ProbeAvailability()
    {
        if (!OperatingSystem.IsWindows())
            return false;

        try
        {
            return NativeIsAvailable() != 0;
        }
        catch (DllNotFoundException)
        {
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
        catch (BadImageFormatException)
        {
            return false;
        }
    }

    [LibraryImport(LibraryName, EntryPoint = "nntrain_f16c_available")]
    private static partial int NativeIsAvailable();

    [LibraryImport(LibraryName, EntryPoint = "nntrain_f16_linear_forward")]
    private static unsafe partial void NativeLinearForward(
        ushort* input,
        ushort* weight,
        ushort* bias,
        ushort* output,
        int rowStart,
        int rowCount,
        int inputWidth,
        int outputWidth,
        int applyRelu);

    [LibraryImport(LibraryName, EntryPoint = "nntrain_f16_linear_backward_input")]
    private static unsafe partial void NativeLinearBackwardInput(
        float* outputGradient,
        ushort* output,
        ushort* weight,
        float* inputGradient,
        int rowStart,
        int rowCount,
        int inputWidth,
        int outputWidth,
        int applyRelu);

    [LibraryImport(LibraryName, EntryPoint = "nntrain_f16_linear_backward_weight")]
    private static unsafe partial void NativeLinearBackwardWeight(
        ushort* input,
        float* outputGradient,
        ushort* output,
        float* weightGradient,
        float* biasGradient,
        int columnStart,
        int columnCount,
        int rows,
        int inputWidth,
        int outputWidth,
        int applyRelu);


    internal static unsafe void LinearForwardRows(
        Half[] input,
        Half[] weight,
        Half[] bias,
        Half[] output,
        int rowStart,
        int rowCount,
        int inputWidth,
        int outputWidth,
        bool applyRelu)
    {
        fixed (Half* inputPointer = input)
        fixed (Half* weightPointer = weight)
        fixed (Half* biasPointer = bias)
        fixed (Half* outputPointer = output)
        {
            NativeLinearForward(
                (ushort*)inputPointer,
                (ushort*)weightPointer,
                (ushort*)biasPointer,
                (ushort*)outputPointer,
                rowStart,
                rowCount,
                inputWidth,
                outputWidth,
                applyRelu ? 1 : 0);
        }
    }

    internal static unsafe void LinearBackwardInputRows(
        float[] outputGradient,
        Half[] output,
        Half[] weight,
        float[] inputGradient,
        int rowStart,
        int rowCount,
        int inputWidth,
        int outputWidth,
        bool applyRelu)
    {
        fixed (float* outputGradientPointer = outputGradient)
        fixed (Half* outputPointer = output)
        fixed (Half* weightPointer = weight)
        fixed (float* inputGradientPointer = inputGradient)
        {
            NativeLinearBackwardInput(
                outputGradientPointer,
                (ushort*)outputPointer,
                (ushort*)weightPointer,
                inputGradientPointer,
                rowStart,
                rowCount,
                inputWidth,
                outputWidth,
                applyRelu ? 1 : 0);
        }
    }

    internal static unsafe void LinearBackwardWeightColumns(
        Half[] input,
        float[] outputGradient,
        Half[] output,
        float[] weightGradient,
        float[] biasGradient,
        int columnStart,
        int columnCount,
        int rows,
        int inputWidth,
        int outputWidth,
        bool applyRelu)
    {
        fixed (Half* inputPointer = input)
        fixed (float* outputGradientPointer = outputGradient)
        fixed (Half* outputPointer = output)
        fixed (float* weightGradientPointer = weightGradient)
        fixed (float* biasGradientPointer = biasGradient)
        {
            NativeLinearBackwardWeight(
                (ushort*)inputPointer,
                outputGradientPointer,
                (ushort*)outputPointer,
                weightGradientPointer,
                biasGradientPointer,
                columnStart,
                columnCount,
                rows,
                inputWidth,
                outputWidth,
                applyRelu ? 1 : 0);
        }
    }

}
