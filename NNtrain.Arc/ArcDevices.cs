using System.Text;

namespace NNtrain.Arc;

public sealed record ArcDeviceInfo(int Index, string Name, string DriverVersion, ulong GlobalMemoryBytes, ulong MaximumAllocationBytes)
{
    internal nint NativeDevice { get; init; }
    public string Extensions { get; init; } = "";
    public int MinimumSubgroupSize { get; init; }
    public bool SupportsXmx => Extensions.Split(' ').Contains("cl_intel_subgroup_matrix_multiply_accumulate")
        && MinimumSubgroupSize is 8 or 16;
}

public static class ArcDevices
{
    public static IReadOnlyList<ArcDeviceInfo> Enumerate()
    {
        if (!OperatingSystem.IsWindows()) return [];
        try
        {
            int error = OpenClNative.clGetPlatformIDs(0, null, out uint count);
            if (error == -1001) return []; // CL_PLATFORM_NOT_FOUND_KHR
            OpenClNative.Check(error, "enumerate platforms");
            var platforms = new nint[count];
            OpenClNative.Check(OpenClNative.clGetPlatformIDs(count, platforms, out _), "read platforms");
            var result = new List<ArcDeviceInfo>();
            foreach (nint platform in platforms)
            {
                error = OpenClNative.clGetDeviceIDs(platform, 4, 0, null, out count); // GPU only
                if (error == -1) continue;
                OpenClNative.Check(error, "enumerate GPU devices");
                var devices = new nint[count];
                OpenClNative.Check(OpenClNative.clGetDeviceIDs(platform, 4, count, devices, out _), "read GPU devices");
                foreach (nint device in devices)
                {
                    string name = Text(device, 0x102b);
                    if (BitConverter.ToUInt32(Info(device, 0x1001)) != 0x8086
                        || !name.Contains("Arc", StringComparison.OrdinalIgnoreCase)) continue;
                    result.Add(new(result.Count, name, Text(device, 0x102d),
                        BitConverter.ToUInt64(Info(device, 0x101f)),
                        BitConverter.ToUInt64(Info(device, 0x1010))) {
                            NativeDevice = device, Extensions = Text(device, 0x1030), MinimumSubgroupSize = SubgroupSize(device) });
                }
            }
            return result.AsReadOnly();
        }
        catch (DllNotFoundException) { return []; }
        catch (EntryPointNotFoundException) { return []; }
    }

    internal static ArcDeviceInfo Get(int index)
    {
        IReadOnlyList<ArcDeviceInfo> devices = Enumerate();
        return index >= 0 && index < devices.Count ? devices[index]
            : throw new InvalidOperationException($"Intel Arc device {index} is unavailable. Install the Intel graphics driver with OpenCL support. CPU/CUDA fallback is not performed.");
    }

    private static byte[] Info(nint device, uint field)
    {
        OpenClNative.Check(OpenClNative.clGetDeviceInfo(device, field, 0, null, out nuint size), "device info size");
        var bytes = new byte[checked((int)size)];
        OpenClNative.Check(OpenClNative.clGetDeviceInfo(device, field, size, bytes, out _), "device info");
        return bytes;
    }
    private static int SubgroupSize(nint device)
    {
        if (OpenClNative.clGetDeviceInfo(device, 0x4108, 0, null, out nuint length) != 0 || length == 0) return 0;
        byte[] data = new byte[checked((int)length)];
        if (OpenClNative.clGetDeviceInfo(device, 0x4108, length, data, out _) != 0) return 0;
        int minimum = int.MaxValue;
        for (int i = 0; i < data.Length; i += IntPtr.Size)
            minimum = Math.Min(minimum, checked((int)(IntPtr.Size == 8 ? BitConverter.ToUInt64(data, i) : BitConverter.ToUInt32(data, i))));
        return minimum;
    }
    private static string Text(nint device, uint field) => Encoding.UTF8.GetString(Info(device, field)).TrimEnd('\0');
}
