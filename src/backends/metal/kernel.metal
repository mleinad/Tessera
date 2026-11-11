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
