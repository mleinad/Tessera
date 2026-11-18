#include "common/WaveCommon.h"
#include "CudaKernels.hpp"

// Parameter order must match the bind indices: BIND_U_CURR (0), BIND_U_PREV (1), BIND_PARAMS (2)
__global__ void wave_step(const float *__restrict__ u,
                          float *__restrict__ u_prev,
                          WaveParams p)
{
    const u32 x = blockIdx.x * blockDim.x + threadIdx.x;
    const u32 y = blockIdx.y * blockDim.y + threadIdx.y;
    const u32 z = blockIdx.z;
    if (x >= p.nx || y >= p.ny || z >= p.nz)
        return;
    u64 i = wave_idx(x + 1, y + 1, z + 1, p.sy, p.sz); // +1 skips the ghost layer
    wave_update(u, u_prev, i, p.sy, p.sz, p.C);
}

const void *FindCudaKernel(const std::string &name)
{
    if (name == "wave_step")
        return (const void *)wave_step;
    return nullptr;
}
