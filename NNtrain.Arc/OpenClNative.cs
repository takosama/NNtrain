using System.Runtime.InteropServices;

namespace NNtrain.Arc;

internal static class OpenClNative
{
    private const string Library = "OpenCL.dll";
    [DllImport(Library)] internal static extern int clGetDeviceAndHostTimer(nint device, out ulong deviceTimestamp, out ulong hostTimestamp);
    [DllImport(Library)] internal static extern int clGetPlatformIDs(uint count, [Out] nint[]? platforms, out uint actual);
    [DllImport(Library)] internal static extern int clGetDeviceIDs(nint platform, ulong type, uint count, [Out] nint[]? devices, out uint actual);
    [DllImport(Library)] internal static extern int clGetDeviceInfo(nint device, uint name, nuint size, [Out] byte[]? value, out nuint actual);
    [DllImport(Library)] internal static extern nint clCreateContext(nint properties, uint count, nint[] devices, nint notify, nint data, out int error);
    [DllImport(Library)] internal static extern nint clCreateCommandQueue(nint context, nint device, ulong properties, out int error);
    [DllImport(Library)] internal static extern nint clCreateProgramWithSource(nint context, uint count, nint[] strings, nuint[] lengths, out int error);
    [DllImport(Library, CharSet = CharSet.Ansi)] internal static extern int clBuildProgram(nint program, uint count, nint[] devices, string options, nint notify, nint data);
    [DllImport(Library)] internal static extern int clGetProgramBuildInfo(nint program, nint device, uint name, nuint size, [Out] byte[]? value, out nuint actual);
    [DllImport(Library, CharSet = CharSet.Ansi)] internal static extern nint clCreateKernel(nint program, string name, out int error);
    [DllImport(Library)] internal static extern int clGetKernelWorkGroupInfo(nint kernel, nint device, uint name, nuint size, out ulong value, out nuint actual);
    [DllImport(Library)] internal static extern nint clCreateBuffer(nint context, ulong flags, nuint size, nint host, out int error);
    [DllImport(Library)] internal static extern int clSetKernelArg(nint kernel, uint index, nuint size, nint value);
    [DllImport(Library)] internal static extern int clEnqueueNDRangeKernel(nint queue, nint kernel, uint dimensions, nint offset, nuint[] global, nuint[]? local, uint waitCount, nint waitList, nint evt);
    [DllImport(Library)] internal static extern int clEnqueueReadBuffer(nint queue, nint buffer, uint blocking, nuint offset, nuint bytes, nint host, uint waitCount, nint waitList, nint evt);
    [DllImport(Library)] internal static extern int clEnqueueWriteBuffer(nint queue, nint buffer, uint blocking, nuint offset, nuint bytes, nint host, uint waitCount, nint waitList, nint evt);
    [DllImport(Library)] internal static extern int clEnqueueCopyBuffer(nint queue, nint source, nint target, nuint sourceOffset, nuint targetOffset, nuint bytes, uint waitCount, nint waitList, nint evt);
    [DllImport(Library)] internal static extern int clGetEventProfilingInfo(nint evt, uint name, nuint size, out ulong value, out nuint actual);
    [DllImport(Library)] internal static extern int clReleaseEvent(nint evt);
    [DllImport(Library)] internal static extern int clFinish(nint queue);
    [DllImport(Library)] internal static extern int clFlush(nint queue);
    [DllImport(Library)] internal static extern int clWaitForEvents(uint count, nint events);
    [DllImport(Library)] internal static extern int clReleaseMemObject(nint buffer);
    [DllImport(Library)] internal static extern int clReleaseKernel(nint kernel);
    [DllImport(Library)] internal static extern int clReleaseProgram(nint program);
    [DllImport(Library)] internal static extern int clReleaseCommandQueue(nint queue);
    [DllImport(Library)] internal static extern int clReleaseContext(nint context);

    internal static void Check(int error, string operation)
    {
        if (error != 0) throw new InvalidOperationException($"Arc OpenCL {operation} failed ({error}).");
    }
}
