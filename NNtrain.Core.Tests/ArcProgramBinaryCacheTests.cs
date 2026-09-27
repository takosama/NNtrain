using System.Security.Cryptography;
using System.Text;
using NNtrain.Arc;
using Xunit;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Core.Tests;

public sealed class ArcProgramBinaryCacheTests
{
    [Fact]
    public void CpuKeyIncludesSourceCompilerOptionsAndDeviceDriver()
    {
        byte[] source = Encoding.UTF8.GetBytes("__kernel void example() {}\n");
        const string options = "-cl-std=CL1.2 -DQ35_PROJECTION_WG=32";
        const string device = "Intel Arc B580\n32.0.101.1\ncl_intel_subgroups\n16\n8";
        string key = ProgramCacheKey(source, options, device);
        Assert.Matches("^[0-9A-F]{64}$", key);
        Assert.Equal(key, ProgramCacheKey((byte[])source.Clone(), options, device));
        Assert.NotEqual(key, ProgramCacheKey([.. source, (byte)' '], options, device));
        Assert.NotEqual(key, ProgramCacheKey(source, options.Replace("WG=32", "WG=64"), device));
        Assert.NotEqual(key, ProgramCacheKey(source, options, device.Replace("32.0.101.1", "32.0.101.2")));
        Assert.NotEqual(key, ProgramCacheKey(source, options, device.Replace("B580", "A770")));
        Assert.NotEqual(key, ProgramCacheKey(source, options, device.Replace("\n16\n", "\n8\n")));
        Assert.NotEqual(key, ProgramCacheKey(source, options, device + " extra_extension"));
        // Input boundaries cannot collapse distinct ordinary compiler inputs.
        Assert.NotEqual(ProgramCacheKey("c"u8.ToArray(), "ab", "device"),
            ProgramCacheKey("bc"u8.ToArray(), "a", "device"));
    }

    [Fact]
    public void CpuCacheRoundTripsAndReplacesWithoutTemporaryFiles()
    {
        using var directory = new CacheDirectory();
        string path = Path.Combine(directory.Path, "cache.bin");
        byte[] first = Enumerable.Range(0, 4097).Select(i => (byte)(i * 13)).ToArray();
        WriteProgramCache(path, first);
        Assert.Equal(first, ReadProgramCache(path));
        byte[] onDisk = File.ReadAllBytes(path);
        Assert.Equal(SHA256.HashData(first), onDisk[..32]);
        Assert.Equal(first, onDisk[32..]);
        byte[] replacement = [3, 9, 27, 81];
        WriteProgramCache(path, replacement);
        Assert.Equal(replacement, ReadProgramCache(path));
        Assert.Single(Directory.GetFiles(directory.Path));
        Assert.Empty(Directory.GetFiles(directory.Path, "*.tmp"));
    }

    [Fact]
    public void CpuMissingTruncatedChangedAndOversizedEntriesAreMisses()
    {
        using var directory = new CacheDirectory();
        string path = Path.Combine(directory.Path, "cache.bin");
        Assert.Null(ReadProgramCache(path));
        byte[] valid = [.. SHA256.HashData(new byte[] { 1, 2, 3 }), 1, 2, 3];
        foreach (int length in new[] { 0, 1, 31, 32, 34 })
        {
            File.WriteAllBytes(path, valid[..length]);
            Assert.Null(ReadProgramCache(path));
        }
        foreach (int offset in new[] { 0, 31, 32, valid.Length - 1 })
        {
            byte[] changed = (byte[])valid.Clone();
            changed[offset] ^= 1;
            File.WriteAllBytes(path, changed);
            Assert.Null(ReadProgramCache(path));
        }
        using (var oversized = new FileStream(path, FileMode.Create, FileAccess.Write))
            oversized.SetLength(128L * 1024 * 1024 + 33);
        Assert.Null(ReadProgramCache(path));
    }

    [Fact]
    public void CpuUnavailableCacheDoesNotReplaceOrThrow()
    {
        using var directory = new CacheDirectory();
        string blocked = Path.Combine(directory.Path, "blocked");
        File.WriteAllBytes(blocked, [17]);
        string path = Path.Combine(blocked, "cache.bin");
        Assert.Null(ReadProgramCache(path));
        WriteProgramCache(path, [1, 2, 3]);
        Assert.Equal(new byte[] { 17 }, File.ReadAllBytes(blocked));
    }

    [Fact]
    public void GpuSecondLoadHitsAndPreservesKernelResults()
    {
        RequireArc();
        using var directory = new CacheDirectory();
        var options = Options(directory.Path);
        int[] expected;
        using (var first = new ArcExecutionLane(0, options))
        {
            Assert.False(first.ProgramBinaryCacheHit);
            expected = RunSimpleKernels(first);
        }
        Assert.Single(Directory.GetFiles(directory.Path, "*.bin"));
        using var second = new ArcExecutionLane(0, options);
        Assert.True(second.ProgramBinaryCacheHit);
        Assert.Equal(expected, RunSimpleKernels(second));
    }

    [Fact]
    public void GpuChangedCompilerOptionsProduceANewEntry()
    {
        RequireArc();
        using var directory = new CacheDirectory();
        var options = Options(directory.Path);
        int[] expected;
        using (var first = new ArcExecutionLane(0, options))
        {
            Assert.False(first.ProgramBinaryCacheHit);
            expected = RunSimpleKernels(first);
        }
        var changed = options with { ExperimentalOptimizationKernels = true };
        using (var second = new ArcExecutionLane(0, changed))
        {
            Assert.False(second.ProgramBinaryCacheHit);
            Assert.Equal(expected, RunSimpleKernels(second));
        }
        Assert.Equal(2, Directory.GetFiles(directory.Path, "*.bin").Length);
        using var third = new ArcExecutionLane(0, changed);
        Assert.True(third.ProgramBinaryCacheHit);
        Assert.Equal(expected, RunSimpleKernels(third));
    }

    [Fact]
    public void GpuCorruptOrDriverRejectedBinariesRebuildThenHit()
    {
        RequireArc();
        using var directory = new CacheDirectory();
        var options = Options(directory.Path);
        int[] expected;
        using (var first = new ArcExecutionLane(0, options)) expected = RunSimpleKernels(first);
        string path = Assert.Single(Directory.GetFiles(directory.Path, "*.bin"));
        byte[] corrupted = File.ReadAllBytes(path);
        corrupted[^1] ^= 1;
        File.WriteAllBytes(path, corrupted);
        using (var rebuilt = new ArcExecutionLane(0, options))
        {
            Assert.False(rebuilt.ProgramBinaryCacheHit);
            Assert.Equal(expected, RunSimpleKernels(rebuilt));
        }
        // This envelope passes SHA validation but is not an OpenCL program.
        // The driver-rejection path must also release it and compile source.
        WriteProgramCache(path, "not an OpenCL program"u8.ToArray());
        Assert.NotNull(ReadProgramCache(path));
        using (var rebuilt = new ArcExecutionLane(0, options))
        {
            Assert.False(rebuilt.ProgramBinaryCacheHit);
            Assert.Equal(expected, RunSimpleKernels(rebuilt));
        }
        using var cached = new ArcExecutionLane(0, options);
        Assert.True(cached.ProgramBinaryCacheHit);
        Assert.Equal(expected, RunSimpleKernels(cached));
    }

    [Fact]
    public void GpuDisabledOrUnavailableCacheStillRunsSource()
    {
        RequireArc();
        using var directory = new CacheDirectory();
        using (var disabled = new ArcExecutionLane(0, Options(directory.Path) with { CacheProgramBinary = false }))
        {
            Assert.False(disabled.ProgramBinaryCacheHit);
            RunSimpleKernels(disabled);
        }
        Assert.Empty(Directory.GetFiles(directory.Path));
        string blocked = Path.Combine(directory.Path, "blocked");
        File.WriteAllBytes(blocked, [17]);
        using var unavailable = new ArcExecutionLane(0, Options(blocked));
        Assert.False(unavailable.ProgramBinaryCacheHit);
        RunSimpleKernels(unavailable);
        Assert.Equal(new byte[] { 17 }, File.ReadAllBytes(blocked));
    }

    private static ArcExecutionOptions Options(string directory) => new()
    {
        Qwen35InferenceKernelsOnly = true, CacheProgramBinary = true,
        ProgramCacheDirectory = directory, XmxMatrices = false, BufferPoolBytes = 0
    };

    private static void RequireArc() =>
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc GPU is required.");

    private static int[] RunSimpleKernels(ArcExecutionLane lane)
    {
        using ArcBuffer values = lane.Upload(new float[] { 1, -0f, 1.5f, -3.75f, 23.25f, 0.125f, -123.5f });
        using ArcBuffer additions = lane.Upload(new float[] { 2, 0, -0.5f, 2.25f, -22.25f, -0.125f, 17 });
        lane.Run("q35a_add_in_place", 128, 128, values, additions, 6);
        float[] actual = new float[7];
        lane.Read(values, actual);
        Assert.Equal(new float[] { 3, 0, 1, -1.5f, 1, 0, -123.5f }.Select(BitConverter.SingleToInt32Bits),
            actual.Select(BitConverter.SingleToInt32Bits));
        lane.Run("q35a_zero", 128, 128, values, 3);
        lane.Read(values, actual);
        int[] expected = new float[] { 0, 0, 0, -1.5f, 1, 0, -123.5f }.Select(BitConverter.SingleToInt32Bits).ToArray();
        Assert.Equal(expected, actual.Select(BitConverter.SingleToInt32Bits));
        return actual.Select(BitConverter.SingleToInt32Bits).ToArray();
    }

    private sealed class CacheDirectory : IDisposable
    {
        public string Path { get; } = System.IO.Path.Combine(System.IO.Path.GetTempPath(),
            "NNtrain-program-cache-tests", Guid.NewGuid().ToString("N"));
        public CacheDirectory() => Directory.CreateDirectory(Path);
        public void Dispose() => Directory.Delete(Path, recursive: true);
    }
}
