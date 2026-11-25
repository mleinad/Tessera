// metal/kernels.metal
#include <metal_stdlib>
using namespace metal;
#include "../../common/WaveCommon.h"

kernel void wave_step(const device float* u      [[buffer(BIND_U_CURR)]],
                      device float*       u_prev [[buffer(BIND_U_PREV)]],
                      constant WaveParams& p     [[buffer(BIND_PARAMS)]],
                      uint3 g [[thread_position_in_grid]])
{
    if (g.x >= p.nx || g.y >= p.ny || g.z >= p.nz) return;
    u64 i = wave_idx(g.x + 1, g.y + 1, g.z + 1, p.sy, p.sz);   // +1 skips the ghost layer
    wave_update(u, u_prev, i, p.sy, p.sz, p.C);
}


kernel void wave_step_obstacles(const device float* u      [[buffer(BIND_U_CURR)]],
                                device float*       u_prev [[buffer(BIND_U_PREV)]],
                                constant WaveParams& p     [[buffer(BIND_PARAMS)]],
                                const device uchar* solid  [[buffer(BIND_MASK)]],
                                uint3 g [[thread_position_in_grid]])
{
    if (g.x >= p.nx || g.y >= p.ny || g.z >= p.nz) return;
    u64 i = wave_idx(g.x + 1, g.y + 1, g.z + 1, p.sy, p.sz);
    wave_update_obstacles(u, u_prev, solid, i, p.sy, p.sz, p.C);
}


kernel void wave_step_tiled(const device float* u      [[buffer(BIND_U_CURR)]],
                            device float*       u_prev [[buffer(BIND_U_PREV)]],
                            constant WaveParams& p     [[buffer(BIND_PARAMS)]],
                            uint2 tid                  [[thread_position_in_threadgroup]],
                            uint2 gid                  [[threadgroup_position_in_grid]])
{
    threadgroup float tile[TILE_Y + 2][TILE_X + 2];

    const uint tx = tid.x, ty = tid.y;
    const uint x  = gid.x * TILE_X + tx + 1;
    const uint y  = gid.y * TILE_Y + ty + 1;
    const bool active = (x <= p.nx) && (y <= p.ny);

    u64   i       = wave_idx(x, y, 1, p.sy, p.sz);
    float below   = active ? u[i - p.sz] : 0.0f;
    float current = active ? u[i]        : 0.0f;

    for (uint z = 1; z <= p.nz; ++z, i += p.sz) {
        const float above = active ? u[i + p.sz] : 0.0f;

        tile[ty + 1][tx + 1] = current;
        if (active) {
            if (tx == 0)                       tile[ty + 1][0]      = u[i - 1];
            if (tx == TILE_X - 1 || x == p.nx) tile[ty + 1][tx + 2] = u[i + 1];
            if (ty == 0)                       tile[0][tx + 1]      = u[i - p.sy];
            if (ty == TILE_Y - 1 || y == p.ny) tile[ty + 2][tx + 1] = u[i + p.sy];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active) {
            u_prev[i] = wave_update_values(current, tile[ty + 1][tx], tile[ty + 1][tx + 2],
                                           tile[ty][tx + 1], tile[ty + 2][tx + 1],
                                           below, above, u_prev[i], p.C);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        below   = current;
        current = above;
    }
}
