using System.Runtime.InteropServices;
using System.Text;
using NNtrain;
using NNtrain.Cuda.Execution;
using NNtrain.Cuda.Interop;
using Xunit;

public sealed class CudaNativeAbiGatewayTests
{
    private static readonly string[] RequiredExports =
    [
        "nntrain_abi_version",
        "nntrain_last_error",
        "nntrain_error_snapshot",
        "nntrain_capability_bitmap",
        "nntrain_cuda_device_count",
        "nntrain_cuda_device_name",
        "nntrain_cuda_set_device",
        "nntrain_cuda_use_external_stream",
        "nntrain_cuda_synchronize",
        "nntrain_cuda_mem_info",
        "nntrain_cuda_malloc",
        "nntrain_cuda_free",
        "nntrain_cuda_memset",
        "nntrain_cuda_copy_h2d",
        "nntrain_cuda_copy_d2h",
        "nntrain_cuda_host_alloc",
        "nntrain_cuda_host_free",
        "nntrain_cuda_stream_create",
        "nntrain_cuda_stream_destroy",
        "nntrain_cuda_stream_synchronize",
        "nntrain_cuda_event_create",
        "nntrain_cuda_event_destroy",
        "nntrain_cuda_event_record",
        "nntrain_cuda_event_query",
        "nntrain_cuda_copy_d2h_async_record",
        "nntrain_cuda_copy_h2d_async_record",
        "nntrain_cuda_event_synchronize",
        "nntrain_cuda_copy_d2d",
        "nntrain_cuda_can_access_peer",
        "nntrain_cuda_error_string",
        "nntrain_cuda_bfp8_quantize_f32",
        "nntrain_cuda_bfp8_dequantize_f32",
        "nntrain_cuda_bfp8_dequantize_bf16",
        "nntrain_cuda_bfp8_quantize_bf16",
        "nntrain_cuda_bfp8_requantize_i32",
        "nntrain_cuda_bfp8_transpose_i8",
    ];

    [Fact]
    public void PackedAbiVersionRoundTrips()
    {
        var version = new CudaAbiVersion(7, 19);

        Assert.Equal(version, CudaAbiVersion.FromPacked(version.Packed));
        Assert.Equal("7.19", version.ToString());
    }

    [Fact]
    public void NativePayloadRetainsRuntimeExportsAndAddsVersionedGatewayExports()
    {
        string path = Path.Combine(
            AppContext.BaseDirectory,
            CudaNativeGateway.LibraryName);
        Assert.True(File.Exists(path), $"Native payload not found at {path}.");

        nint library = NativeLibrary.Load(path);
        try
        {
            foreach (string export in RequiredExports)
            {
                Assert.True(
                    NativeLibrary.TryGetExport(library, export, out _),
                    $"Missing native export: {export}");
            }
        }
        finally
        {
            NativeLibrary.Free(library);
        }
    }

    [Fact]
    public void GatewayValidatesAbiAndConvertsDeviceCapabilities()
    {
        CudaNativeGateway.EnsureCompatibleAbi();

        Assert.Equal(1, CudaNativeGateway.AbiVersion.Major);
        Assert.Equal(
            0,
            CudaNativeGateway.DeviceCount(out int deviceCount));
        Assert.True(deviceCount > 0);
        Assert.Equal(
            0,
            CudaNativeGateway.KernelCapabilities(
                0,
                out CudaKernelCapabilities capabilities));
        Assert.True(capabilities.ComputeCapabilityMajor >= 8);
        Assert.True(capabilities.Supports(CudaKernelFeature.TensorCores));
        Assert.True(capabilities.Supports(CudaKernelFeature.BFloat16));
        Assert.True(capabilities.Supports(CudaKernelFeature.FlashAttention));
        Assert.True(capabilities.Supports(CudaKernelFeature.Bfp8Quantization));
        Assert.True(capabilities.Supports(CudaKernelFeature.Int8TensorCores));
        Assert.False(capabilities.Supports(CudaKernelFeature.CudaGraphs));
    }

    [Fact]
    public void FailureSnapshotIsImmutableAndCopiedIntoCompatibleException()
    {
        int status = CudaNativeGateway.DeviceName(
            0,
            new StringBuilder(),
            capacity: 0);
        Assert.NotEqual(0, status);

        NativeCudaException exception = Assert.Throws<NativeCudaException>(
            () => NativeCudaRuntime.Check(status, "ABI probe"));

        Assert.Equal(status, exception.Status);
        Assert.Equal(
            $"ABI probe failed with CUDA error {status}: " +
                CudaNativeGateway.ErrorString(status),
            exception.Message);
        Assert.True(exception.NativeError.HasValue);
        CudaNativeErrorInfo nativeError = exception.NativeError.Value;
        Assert.Equal(CudaNativeGateway.AbiVersion, nativeError.AbiVersion);
        Assert.True(nativeError.Sequence > 0);
        Assert.Equal(status, nativeError.Status);
        Assert.Equal(0, nativeError.DeviceIndex);
        Assert.Equal(CudaNativeOperation.DeviceName, nativeError.Operation);

        Assert.True(CudaNativeGateway.TryGetLastError(out var latest));
        Assert.Equal(nativeError.Sequence, latest.Sequence);
        Assert.Equal(nativeError.Status, latest.Status);
        Assert.Equal(nativeError.DeviceIndex, latest.DeviceIndex);
        Assert.Equal(nativeError.Operation, latest.Operation);
    }
}
