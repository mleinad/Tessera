#ifndef Wave_Common_hpp
#define Wave_Common_hpp

#if defined(__METAL_VERSION__)
#define WAVE_FN     inline
#define WAVE_GLOBAL device
typedef uint  u32;
typedef ulong u64;
#else
#include <cstdint>
#if defined(__CUDACC__)
#define WAVE_FN __host__ __device__ inline
#else
#define WAVE_FN inline
#endif
#define WAVE_GLOBAL
typedef uint32_t u32;
typedef uint64_t u64;
#endif

#if defined(__METAL_VERSION__)
  typedef uchar u8;
#else
  typedef uint8_t u8;
#endif

// Bind indices: one source of truth for host and kernels
#define BIND_U_CURR 0
#define BIND_U_PREV 1
#define BIND_PARAMS 2
#define BIND_MASK   3

// https://developer.download.nvidia.com/CUDA/CUDA_Zone/papers/gpu_3dfd_rev.pdf
#define TILE_X 32
#define TILE_Y 8

struct WaveParams {
    // interior size (grid is (n+2)^3 with a zero ghost layer)
    u32 nx;
    u32 ny;
    u32 nz;

    u32   pad_;   // explicit padding so the layout is identical everywhere
    u64   sy, sz; // strides in elements: sy = nx+2, sz = (nx+2)*(ny+2)
    float C;      // (c*dt/dx)^2
};

WAVE_FN u64 wave_idx(u32 x, u32 y, u32 z, u64 sy, u64 sz) {
    return (u64)x + (u64)y * sy + (u64)z * sz;
}

// part of Micikevicius's improvements 
WAVE_FN float wave_update_values(float c, float xm, float xp, float ym, float yp,
                                 float zm, float zp, float prev, float C) {
    float lap = xm + xp + ym + yp + zm + zp - 6.0f * c;
    return 2.0f * c - prev + C * lap;
}

WAVE_FN void wave_update(const WAVE_GLOBAL float* u, WAVE_GLOBAL float* u_prev,
                         u64 i, u64 sy, u64 sz, float C) {
    u_prev[i] = wave_update_values(u[i], u[i - 1], u[i + 1], u[i - sy], u[i + sy],
                                   u[i - sz], u[i + sz], u_prev[i], C);
}


// Rigid (sound-hard) obstacles: a solid neighbour reflects by mirroring the centre value
WAVE_FN float wave_nb(const WAVE_GLOBAL float* u, const WAVE_GLOBAL u8* solid, u64 i, u64 j) {
    return solid[j] ? u[i] : u[j];
}

WAVE_FN void wave_update_obstacles(const WAVE_GLOBAL float* u, WAVE_GLOBAL float* u_prev,
                                   const WAVE_GLOBAL u8* solid, u64 i, u64 sy, u64 sz, float C) {
    if (solid[i]) { u_prev[i] = 0.0f; return; }          // no pressure inside objects
    const float c = u[i];
    float lap = wave_nb(u, solid, i, i - 1)  + wave_nb(u, solid, i, i + 1)
              + wave_nb(u, solid, i, i - sy) + wave_nb(u, solid, i, i + sy)
              + wave_nb(u, solid, i, i - sz) + wave_nb(u, solid, i, i + sz) - 6.0f * c;
    u_prev[i] = 2.0f * c - u_prev[i] + C * lap;
}

#endif
