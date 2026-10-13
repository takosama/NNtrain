using System.Runtime.InteropServices;
using LLama.Native;

namespace NNtrain.Gemma;

/// <summary>Owns the process-wide native backend selection; callers must hold SyncRoot.</summary>
internal static class Gemma3NativeBackend
{
    // LLamaSharp selects one native backend per process. ModelParams itself calls
    // native code, so select the backend before constructing it.
    internal static readonly object SyncRoot = new();
    private static Gemma3Backend? _selectedBackend;
    private static bool _selectedAllOpenClDevices;
    private static bool _nativeBackendConfigurationFailed;

    internal static void Configure(Gemma3Backend backend, bool allOpenClDevices)
    {
        if (_nativeBackendConfigurationFailed)
            throw new InvalidOperationException(
                "The Gemma 3 native backend failed to initialize. Restart NNtrain before retrying.");
        if (_selectedBackend is Gemma3Backend alreadySelected)
        {
            if (alreadySelected != backend)
                throw new InvalidOperationException(
                    "LLamaSharp cannot switch between CPU and OpenCL after its native backend loads. " +
                    "Restart NNtrain to change the Gemma 3 backend.");
            if (backend == Gemma3Backend.OpenCl && _selectedAllOpenClDevices != allOpenClDevices)
                throw new InvalidOperationException(
                    "Changing the OpenCL device set requires restarting NNtrain.");
            return;
        }

        var config = NativeLibraryConfig.LLama;
        if (config.LibraryHasLoaded)
            throw new InvalidOperationException(
                "LLamaSharp native backend was loaded before Gemma 3 selected CPU or OpenCL. " +
                "Restart NNtrain and load Gemma 3 first.");

        if (backend == Gemma3Backend.Cpu)
        {
            config.WithCuda(false).WithVulkan(false).WithAutoFallback(true);
            _selectedBackend = backend;
            return;
        }

        if (!OperatingSystem.IsWindows())
            throw new PlatformNotSupportedException("The bundled Gemma 3 OpenCL backend requires Windows and Intel Arc.");

        string directory = Path.Combine(AppContext.BaseDirectory, "gemma-opencl");
        string library = Path.Combine(directory, "llama.dll");
        string openClPlugin = Path.Combine(directory, "ggml-opencl.dll");
        if (!File.Exists(library) || !File.Exists(openClPlugin))
            throw new FileNotFoundException(
                "The bundled Gemma 3 OpenCL backend is incomplete. " +
                "Build NNtrain with gemma-opencl/llama.dll and gemma-opencl/ggml-opencl.dll.",
                !File.Exists(library) ? library : openClPlugin);

        // Select Intel Arc instead of the older NVIDIA OpenCL platform. The
        // default uses one B580; the opt-in mode lets the OpenCL backend see
        // all Intel GPUs for a measured two-device layer split.
        if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("GGML_OPENCL_PLATFORM")))
            SetOpenClEnvironment("GGML_OPENCL_PLATFORM", "Intel");
        if (allOpenClDevices)
        {
            SetOpenClEnvironment("GGML_OPENCL_DEVICE", null);
            // The measured dual-Arc path uses a 16x32 XMX tile for Q8_0 prefill and reuses
            // programs compiled for the shared OpenCL context. Explicit zero
            // values retain the reference kernels for comparisons.
            if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("GGML_OPENCL_Q8_XMX_F16")))
                SetOpenClEnvironment("GGML_OPENCL_Q8_XMX_F16", "1");
            if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("GGML_OPENCL_Q8_XMX_F16_M16N32")))
                SetOpenClEnvironment("GGML_OPENCL_Q8_XMX_F16_M16N32", "1");
            if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("GGML_OPENCL_PROGRAM_CACHE")))
                SetOpenClEnvironment("GGML_OPENCL_PROGRAM_CACHE", "1");
        }
        else if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("GGML_OPENCL_DEVICE")))
            SetOpenClEnvironment("GGML_OPENCL_DEVICE", "B580");

        // WithLibrary is an explicit full-path override: LLamaSharp ignores its
        // ordinary CPU/Vulkan selection and cannot silently choose those DLLs.
        try
        {
            config.WithLibrary(library);
            if (!config.DryRun(out var nativeLibrary) || nativeLibrary is null)
                throw new InvalidOperationException(
                    $"The bundled Gemma 3 OpenCL backend could not be loaded from '{library}'.");
            if (!NativeApi.llama_supports_gpu_offload())
                throw new InvalidOperationException(
                    "The bundled Gemma 3 backend did not expose GPU offload. " +
                    "Check the Intel OpenCL driver and ggml-opencl.dll.");
            _selectedBackend = backend;
            _selectedAllOpenClDevices = allOpenClDevices;
        }
        catch
        {
            // LLamaSharp's explicit library override cannot be safely undone.
            // Do not let a later CPU request accidentally use this GPU DLL.
            _nativeBackendConfigurationFailed = true;
            throw;
        }
    }

    private static void SetOpenClEnvironment(string name, string? value)
    {
        Environment.SetEnvironmentVariable(name, value, EnvironmentVariableTarget.Process);
        // ggml-opencl reads getenv() from the Windows UCRT. Updating only the
        // Win32 process environment after startup does not update that cache.
        int result = PutEnvironmentVariable(name, value ?? string.Empty);
        if (result != 0)
            throw new InvalidOperationException($"Could not set native OpenCL environment {name}: {result}.");
    }

    [DllImport("ucrtbase.dll", EntryPoint = "_putenv_s", CallingConvention = CallingConvention.Cdecl,
        CharSet = CharSet.Ansi)]
    private static extern int PutEnvironmentVariable(string name, string value);
}
