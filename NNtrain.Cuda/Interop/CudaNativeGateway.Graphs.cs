using System.Runtime.InteropServices;

namespace NNtrain.Cuda.Interop;

public static partial class CudaNativeGateway
{
    public static int GraphBeginCapture(int device, nint stream)
    {
        EnsureGraphAbi();
        return Complete(
            GraphNativeMethods.BeginCapture(device, stream),
            CudaNativeOperation.GraphBeginCapture,
            device);
    }

    public static int GraphEndCapture(
        int device,
        nint stream,
        out nint graph)
    {
        EnsureGraphAbi();
        return Complete(
            GraphNativeMethods.EndCapture(device, stream, out graph),
            CudaNativeOperation.GraphEndCapture,
            device);
    }

    public static int GraphInstantiate(
        int device,
        nint graph,
        out nint executable)
    {
        EnsureGraphAbi();
        return Complete(
            GraphNativeMethods.Instantiate(device, graph, out executable),
            CudaNativeOperation.GraphInstantiate,
            device);
    }

    public static int GraphLaunch(
        int device,
        nint executable,
        nint stream)
    {
        EnsureGraphAbi();
        return Complete(
            GraphNativeMethods.Launch(device, executable, stream),
            CudaNativeOperation.GraphLaunch,
            device);
    }

    public static int GraphDestroy(int device, nint graph)
    {
        EnsureGraphAbi();
        return Complete(
            GraphNativeMethods.Destroy(device, graph),
            CudaNativeOperation.GraphDestroy,
            device);
    }

    public static int GraphExecutableDestroy(
        int device,
        nint executable)
    {
        EnsureGraphAbi();
        return Complete(
            GraphNativeMethods.ExecutableDestroy(device, executable),
            CudaNativeOperation.GraphExecutableDestroy,
            device);
    }

    public static int GraphDropoutMask(
        int device,
        nint stepCounter,
        uint seed,
        float dropoutProbability,
        nint output,
        int length,
        nint stream)
    {
        EnsureGraphAbi();
        return Complete(
            GraphNativeMethods.DropoutMask(
                device,
                stepCounter,
                seed,
                dropoutProbability,
                output,
                length,
                stream),
            CudaNativeOperation.GraphRngStep,
            device);
    }

    public static int GraphCounterSet(
        int device,
        nint stepCounter,
        ulong value,
        nint stream)
    {
        EnsureGraphDropoutAbi();
        return Complete(
            GraphNativeMethods.CounterSet(
                device, stepCounter, value, stream),
            CudaNativeOperation.GraphCounterSet,
            device);
    }

    public static int GraphCounterAdvance(
        int device,
        nint stepCounter,
        nint stream)
    {
        EnsureGraphDropoutAbi();
        return Complete(
            GraphNativeMethods.CounterAdvance(device, stepCounter, stream),
            CudaNativeOperation.GraphCounterAdvance,
            device);
    }

    public static int GraphDropoutForwardFloat32(
        int device,
        nint stepCounter,
        ulong operationSeed,
        float dropoutProbability,
        nint input,
        nint output,
        int length,
        nint stream)
    {
        EnsureGraphDropoutAbi();
        return Complete(
            GraphNativeMethods.DropoutForwardFloat32(
                device,
                stepCounter,
                operationSeed,
                dropoutProbability,
                input,
                output,
                length,
                stream),
            CudaNativeOperation.GraphDropoutForward,
            device);
    }

    public static int GraphDropoutForwardBFloat16(
        int device,
        nint stepCounter,
        ulong operationSeed,
        float dropoutProbability,
        nint input,
        nint output,
        int length,
        nint stream)
    {
        EnsureGraphDropoutAbi();
        return Complete(
            GraphNativeMethods.DropoutForwardBFloat16(
                device,
                stepCounter,
                operationSeed,
                dropoutProbability,
                input,
                output,
                length,
                stream),
            CudaNativeOperation.GraphDropoutForward,
            device);
    }

    public static int GraphAddDropoutForwardFloat32(
        int device,
        nint stepCounter,
        ulong operationSeed,
        float dropoutProbability,
        nint residual,
        nint branch,
        nint output,
        int length,
        nint stream)
    {
        EnsureGraphDropoutAbi();
        return Complete(
            GraphNativeMethods.AddDropoutForwardFloat32(
                device,
                stepCounter,
                operationSeed,
                dropoutProbability,
                residual,
                branch,
                output,
                length,
                stream),
            CudaNativeOperation.GraphAddDropoutForward,
            device);
    }

    public static int GraphAddDropoutForwardBFloat16(
        int device,
        nint stepCounter,
        ulong operationSeed,
        float dropoutProbability,
        nint residual,
        nint branch,
        nint output,
        int length,
        nint stream)
    {
        EnsureGraphDropoutAbi();
        return Complete(
            GraphNativeMethods.AddDropoutForwardBFloat16(
                device,
                stepCounter,
                operationSeed,
                dropoutProbability,
                residual,
                branch,
                output,
                length,
                stream),
            CudaNativeOperation.GraphAddDropoutForward,
            device);
    }

    public static int GraphDropoutBackwardFloat32(
        int device,
        nint stepCounter,
        ulong operationSeed,
        float dropoutProbability,
        nint outputGradient,
        nint inputGradient,
        int length,
        nint stream)
    {
        EnsureGraphDropoutAbi();
        return Complete(
            GraphNativeMethods.DropoutBackwardFloat32(
                device,
                stepCounter,
                operationSeed,
                dropoutProbability,
                outputGradient,
                inputGradient,
                length,
                stream),
            CudaNativeOperation.GraphDropoutBackward,
            device);
    }

    public static int GraphAddDropoutBackwardFloat32(
        int device,
        nint stepCounter,
        ulong operationSeed,
        float dropoutProbability,
        nint outputGradient,
        nint residualGradient,
        nint branchGradient,
        int length,
        bool sameParent,
        nint stream)
    {
        EnsureGraphDropoutAbi();
        return Complete(
            GraphNativeMethods.AddDropoutBackwardFloat32(
                device,
                stepCounter,
                operationSeed,
                dropoutProbability,
                outputGradient,
                residualGradient,
                branchGradient,
                length,
                sameParent ? 1 : 0,
                stream),
            CudaNativeOperation.GraphAddDropoutBackward,
            device);
    }

    private static void EnsureGraphAbi()
        => EnsureMinimumAbiMinor(
            CudaAbiVersion.CudaGraphMinor,
            "CUDA Graph capture and replay");

    private static void EnsureGraphDropoutAbi()
        => EnsureMinimumAbiMinor(
            CudaAbiVersion.CudaGraphDropoutMinor,
            "CUDA Graph replay-stable dropout");

    private static class GraphNativeMethods
    {
        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_cuda_stream_begin_capture",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int BeginCapture(int device, nint stream);

        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_cuda_stream_end_capture",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int EndCapture(
            int device,
            nint stream,
            out nint graph);

        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_cuda_graph_instantiate",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Instantiate(
            int device,
            nint graph,
            out nint executable);

        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_cuda_graph_launch",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Launch(
            int device,
            nint executable,
            nint stream);

        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_cuda_graph_destroy",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Destroy(int device, nint graph);

        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_cuda_graph_exec_destroy",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int ExecutableDestroy(
            int device,
            nint executable);

        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_cuda_graph_dropout_mask",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int DropoutMask(
            int device,
            nint stepCounter,
            uint seed,
            float dropoutProbability,
            nint output,
            int length,
            nint stream);

        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_cuda_graph_counter_set",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int CounterSet(
            int device,
            nint stepCounter,
            ulong value,
            nint stream);

        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_cuda_graph_counter_advance",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int CounterAdvance(
            int device,
            nint stepCounter,
            nint stream);

        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_cuda_graph_dropout_forward_float",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int DropoutForwardFloat32(
            int device,
            nint stepCounter,
            ulong operationSeed,
            float dropoutProbability,
            nint input,
            nint output,
            int length,
            nint stream);

        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_cuda_graph_dropout_forward_bf16",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int DropoutForwardBFloat16(
            int device,
            nint stepCounter,
            ulong operationSeed,
            float dropoutProbability,
            nint input,
            nint output,
            int length,
            nint stream);

        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_cuda_graph_add_dropout_forward_float",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int AddDropoutForwardFloat32(
            int device,
            nint stepCounter,
            ulong operationSeed,
            float dropoutProbability,
            nint residual,
            nint branch,
            nint output,
            int length,
            nint stream);

        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_cuda_graph_add_dropout_forward_bf16",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int AddDropoutForwardBFloat16(
            int device,
            nint stepCounter,
            ulong operationSeed,
            float dropoutProbability,
            nint residual,
            nint branch,
            nint output,
            int length,
            nint stream);

        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_cuda_graph_dropout_backward_float",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int DropoutBackwardFloat32(
            int device,
            nint stepCounter,
            ulong operationSeed,
            float dropoutProbability,
            nint outputGradient,
            nint inputGradient,
            int length,
            nint stream);

        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_cuda_graph_add_dropout_backward_float",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int AddDropoutBackwardFloat32(
            int device,
            nint stepCounter,
            ulong operationSeed,
            float dropoutProbability,
            nint outputGradient,
            nint residualGradient,
            nint branchGradient,
            int length,
            int sameParent,
            nint stream);
    }
}
