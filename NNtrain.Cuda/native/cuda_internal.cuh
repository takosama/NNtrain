#pragma once

#include <cuda_runtime.h>

#if defined(_WIN32)
#define NNTRAIN_EXPORT extern "C" __declspec(dllexport)
#else
#define NNTRAIN_EXPORT extern "C" __attribute__((visibility("default")))
#endif

// cuda_runtime_bridge.cu is the sole owner of raw cudaSetDevice calls and of
// the thread-local selected-device cache. Native subsystems that must switch
// devices use this declaration so managed/native device authority cannot
// diverge when work resumes on a pooled host thread.
extern "C" int nntrain_cuda_set_device(int device);

namespace nntrain::cuda::internal {

inline cudaError_t select_device(int device) noexcept {
    return static_cast<cudaError_t>(nntrain_cuda_set_device(device));
}

}  // namespace nntrain::cuda::internal
