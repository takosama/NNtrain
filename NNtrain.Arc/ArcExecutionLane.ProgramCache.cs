using System.Security.Cryptography;
using System.Text;

namespace NNtrain.Arc;

public sealed partial class ArcExecutionLane
{
    public bool ProgramBinaryCacheHit { get; private set; }
    private const int MaximumProgramBinaryBytes = 128 * 1024 * 1024;

    // Cache identity covers every compiler input and the device/driver. A cache
    // is an optional acceleration: unreadable, corrupt or rejected entries fall
    // back to building the embedded source, never to a different program.
    private string? ProgramCachePath(byte[] source, string options)
    {
        if (!Options.CacheProgramBinary) return null;
        try
        {
            string directory = string.IsNullOrWhiteSpace(Options.ProgramCacheDirectory) ? Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "NNtrain", "OpenCLCache")
                : Options.ProgramCacheDirectory;
            return Path.Combine(Path.GetFullPath(directory), ProgramCacheKey(source, options,
                $"{Device.Name}\n{Device.DriverVersion}\n{Device.Extensions}\n{Device.MinimumSubgroupSize}\n{IntPtr.Size}") + ".bin");
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException or NotSupportedException)
        { return null; }
    }

    internal static string ProgramCacheKey(byte[] source, string options, string device)
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        hash.AppendData("NNtrain.OpenCL.v1\0"u8);
        hash.AppendData(Encoding.UTF8.GetBytes(device)); hash.AppendData([0]);
        hash.AppendData(Encoding.UTF8.GetBytes(options)); hash.AppendData([0]);
        hash.AppendData(source);
        return Convert.ToHexString(hash.GetHashAndReset());
    }

    internal static byte[]? ReadProgramCache(string path)
    {
        try
        {
            using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read | FileShare.Delete);
            if (stream.Length <= 32 || stream.Length > MaximumProgramBinaryBytes + 32L) return null;
            byte[] digest = new byte[32]; stream.ReadExactly(digest);
            byte[] binary = GC.AllocateUninitializedArray<byte>(checked((int)stream.Length - 32));
            stream.ReadExactly(binary);
            return CryptographicOperations.FixedTimeEquals(digest, SHA256.HashData(binary)) ? binary : null;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { return null; }
    }

    private unsafe bool TryLoadProgramCache(string? path, string options)
    {
        if (path is null || ReadProgramCache(path) is not byte[] binary) return false;
        nint candidate = 0;
        try
        {
            fixed (byte* data = binary)
            {
                int[] statuses = new int[1];
                candidate = OpenClNative.clCreateProgramWithBinary(_context, 1, [Device.NativeDevice],
                    [(nuint)binary.Length], [(nint)data], statuses, out int error);
                if (error != 0 || statuses[0] != 0 || candidate == 0) return false;
            }
            if (OpenClNative.clBuildProgram(candidate, 1, [Device.NativeDevice], options, 0, 0) != 0) return false;
            _program = candidate; candidate = 0; ProgramBinaryCacheHit = true;
            return true;
        }
        finally { if (candidate != 0) OpenClNative.clReleaseProgram(candidate); }
    }

    private unsafe void SaveProgramCache(string? path)
    {
        if (path is null) return;
        nuint size = 0;
        if (OpenClNative.clGetProgramInfo(_program, 0x1165 /* CL_PROGRAM_BINARY_SIZES */,
            (nuint)sizeof(nuint), (nint)(&size), out _) != 0 || size == 0 || size > MaximumProgramBinaryBytes) return;
        byte[] binary = GC.AllocateUninitializedArray<byte>((int)size);
        fixed (byte* data = binary)
        {
            nint pointer = (nint)data;
            if (OpenClNative.clGetProgramInfo(_program, 0x1166 /* CL_PROGRAM_BINARIES */,
                (nuint)sizeof(nint), (nint)(&pointer), out _) != 0) return;
        }
        WriteProgramCache(path, binary);
    }

    internal static void WriteProgramCache(string path, byte[] binary)
    {
        string? temporary = null;
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            temporary = path + "." + Guid.NewGuid().ToString("N") + ".tmp";
            using (var stream = new FileStream(temporary, FileMode.CreateNew, FileAccess.Write, FileShare.None))
            {
                stream.Write(SHA256.HashData(binary)); stream.Write(binary);
            }
            File.Move(temporary, path, overwrite: true);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { }
        finally
        {
            if (temporary is not null)
                try { File.Delete(temporary); }
                catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { }
        }
    }
}
