# Gemma 3 OpenCL native runtime

These DLLs are a local Windows x64 build of `llama.cpp` commit
`3f7c29d318e317b63f54c558bc69803963d7d88c`, the native ABI used by
LLamaSharp 0.27.0. The upstream license is in `LICENSE.llama.cpp`.

The build uses Khronos OpenCL Headers and ICD Loader `v2024.10.24` as the
OpenCL SDK. `OpenCL.dll` and an Intel OpenCL driver must be installed on the
machine; this directory contains the llama.cpp and ggml DLLs only.

To reproduce the bundled OpenCL DLL, apply the production patches from
this directory to a clean checkout of the pinned commit before configuring
CMake: `ggml-opencl-dxgi-memory.patch`,
`ggml-opencl-unique-device-buffer-names.patch`, then
`ggml-opencl-device-buffer-locality.patch`, then
`ggml-opencl-q8-xmx-f16.patch`, `ggml-opencl-program-cache.patch`, and
`ggml-opencl-q8-xmx-f16-m16n32.patch`.

Configure the source with CMake for Windows x64 using these flags:

```text
-DBUILD_SHARED_LIBS=ON
-DGGML_OPENCL=ON
-DGGML_OPENCL_USE_ADRENO_KERNELS=OFF
-DGGML_OPENCL_EMBED_KERNELS=ON
-DGGML_BACKEND_DL=OFF
-DGGML_VULKAN=OFF
-DGGML_CUDA=OFF
-DLLAMA_BUILD_TESTS=OFF
-DLLAMA_BUILD_EXAMPLES=ON
-DLLAMA_BUILD_SERVER=OFF
```

Set `OpenCL_INCLUDE_DIR` and `OpenCL_LIBRARY` to the SDK paths, then build the
`llama` target in Release mode. Copy `llama.dll`, `ggml.dll`, `ggml-base.dll`,
`ggml-cpu.dll`, and `ggml-opencl.dll` from the build output into this directory.
The project file copies them beside the managed application under
`gemma-opencl/`.

The bundled `ggml-opencl.dll` includes the memory, placement, Q8_0 XMX and
program-reuse patches in `patches/`; the other DLLs retain the
pinned llama.cpp ABI. The unique names prevent the model loader and KV caches
from merging the two B580 buffer types into one map entry. The GUI selects
43 GPU layers at a 53:47 ratio for exactly two B580s with at least 11 GiB each,
at context 4096. Larger device sets retain the 40-layer automatic split; a
single B580 uses 20 layers. The CLI uses one B580 by default; the measured
two-device configuration is `--opencl-all-devices --gpu-layers 43
--tensor-split 53,47 --context 4096`.

The Q8_0 XMX path keeps signed bytes and fp16 scales in VRAM, converts only
the current weight/activation panels to F16 registers, and accumulates in F32.
The 16x32 tile shares two A panels and two B panels across four hardware matrix
operations. The native guard requires Intel FP16 and matrix-MAD support with
minimum subgroup size 16; unsupported devices use the existing kernels.
F16 rounding can change results relative to the F32 reference path.

The managed multi-device path enables `GGML_OPENCL_Q8_XMX_F16=1`,
`GGML_OPENCL_Q8_XMX_F16_M16N32=1`, and `GGML_OPENCL_PROGRAM_CACHE=1` when unset.
Set `GGML_OPENCL_Q8_XMX_F16=0` before startup to compare the reference GEMM.
Program reuse keys the shared context, complete source and options. Every
backend creates separate kernel handles; the cache releases its additional
program references immediately after device initialization.

The bundled OpenCL DLL SHA-256 is
`AB682510EC1877AE47D05BDEB6C9C38E4AFE49CFC4B149BF2B69D9E9AF4F1865`.
See [measurements and validation](../../../docs/benchmarks/gemma3-opencl-aggressive-2026-10-12.md).

`patches/ggml-opencl-dxgi-memory.patch` maps an OpenCL device to its Windows
DXGI adapter and reports its video-memory budget. Both B580s enumerated with
11,875 MiB total and 11,347 MiB free each.

`patches/ggml-opencl-device-buffer-locality.patch` restricts OpenCL tensors to
their owning device. On its own it still placed weights on one GPU and raised
graph splits to 223. `patches/ggml-opencl-unique-device-buffer-names.patch`
fixes the separate model-loader/KV-cache name collision; with all three patches,
40 layers split across the B580s and graph splits were 3 at context 4096.

The SG4 and row-tile/upload candidate patches are not included in the bundled
DLL. CPU host repacking made loading slower; the decode row-tile improvement
was within measurement variation. `ggml-opencl-q8-xmx-numeric-cases.patch` adds
the five numeric test shapes and does not modify the runtime. Build
`test-backend-ops` with `LLAMA_BUILD_TESTS=ON` to reproduce those checks.

`Gemma3LocalModel` selects this runtime explicitly for OpenCL generation. On
this machine, `GGML_OPENCL_PLATFORM=Intel` is required to choose the Arc B580
platform rather than the older NVIDIA OpenCL platform.
