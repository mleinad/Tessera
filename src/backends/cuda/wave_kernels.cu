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

// Same as wave_step, with rigid obstacles from a per-cell solid mask (see wave_update_obstacles)
// Parameter order must match: BIND_U_CURR (0), BIND_U_PREV (1), BIND_PARAMS (2), BIND_MASK (3)
__global__ void wave_step_obstacles(const float *__restrict__ u,
                                    float *__restrict__ u_prev,
                                    WaveParams p,
                                    const u8 *__restrict__ solid)
{
    const u32 x = blockIdx.x * blockDim.x + threadIdx.x;
    const u32 y = blockIdx.y * blockDim.y + threadIdx.y;
    const u32 z = blockIdx.z;
    if (x >= p.nx || y >= p.ny || z >= p.nz)
        return;
    u64 i = wave_idx(x + 1, y + 1, z + 1, p.sy, p.sz); // +1 skips the ghost layer
    wave_update_obstacles(u, u_prev, solid, i, p.sy, p.sz, p.C);
}

// Micikevicius-style single pass (same design as wave_step_tiled in kernel.metal):
// a TILE_X x TILE_Y block holds the current z-slice plus a 1-cell x/y halo in shared memory and
// marches along z, keeping the z-1 / z+1 neighbours in registers. Launch with DispatchColumns.
__global__ void wave_step_tiled(const float *__restrict__ u,
                                float *__restrict__ u_prev,
                                WaveParams p)
{
    __shared__ float tile[TILE_Y + 2][TILE_X + 2];

    const u32  tx     = threadIdx.x, ty = threadIdx.y;
    const u32  x      = blockIdx.x * TILE_X + tx + 1; // +1 skips the ghost layer
    const u32  y      = blockIdx.y * TILE_Y + ty + 1;
    const bool active = (x <= p.nx) && (y <= p.ny);

    u64   i       = wave_idx(x, y, 1, p.sy, p.sz);
    float below   = active ? u[i - p.sz] : 0.0f;
    float current = active ? u[i] : 0.0f;

    for (u32 z = 1; z <= p.nz; ++z, i += p.sz)
    {
        const float above = active ? u[i + p.sz] : 0.0f;

        tile[ty + 1][tx + 1] = current;
        if (active)
        {
            if (tx == 0)                       tile[ty + 1][0]      = u[i - 1];
            if (tx == TILE_X - 1 || x == p.nx) tile[ty + 1][tx + 2] = u[i + 1];
            if (ty == 0)                       tile[0][tx + 1]      = u[i - p.sy];
            if (ty == TILE_Y - 1 || y == p.ny) tile[ty + 2][tx + 1] = u[i + p.sy];
        }
        __syncthreads(); // every thread reaches both barriers: inactive threads don't return early

        if (active)
        {
            u_prev[i] = wave_update_values(current, tile[ty + 1][tx], tile[ty + 1][tx + 2],
                                           tile[ty][tx + 1], tile[ty + 2][tx + 1],
                                           below, above, u_prev[i], p.C);
        }
        __syncthreads(); // the next slice overwrites the tile

        below   = current;
        current = above;
    }
}

const void *FindCudaKernel(const std::string &name)
{
    if (name == "wave_step")
        return (const void *)wave_step;
    if (name == "wave_step_obstacles")
        return (const void *)wave_step_obstacles;
    if (name == "wave_step_tiled")
        return (const void *)wave_step_tiled;
    return nullptr;
}
