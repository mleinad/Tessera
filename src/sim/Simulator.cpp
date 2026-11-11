#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cassert>
#include <chrono>

#include "Simulator.hpp"
#include "common/Benchmark.hpp"
#include "common/Engine.hpp"

// sin(pi * i / (n+1)) for i = 0..n+1; zero exactly on the ghost layer
static std::vector<double> SineTable(u32 n) {
    std::vector<double> t(n + 2);
    for (u32 i = 0; i < n + 2; ++i) {
        t[i] = std::sin(M_PI * i / (n + 1));
    }
    t[0]     = 0.0;
    t[n + 1] = 0.0;
    return t;
}

Simulator::Simulator(u32 n, float courant) {
    m_params.nx = m_params.ny = m_params.nz = n;
    m_params.sy                             = n + 2;
    m_params.sz                             = (u64)(n + 2) * (n + 2);
    m_params.C                              = courant * courant;
    size_t bytes                            = (size_t)(n + 2) * m_params.sz * sizeof(float);
    m_curr                                  = Engine::GetInstance()->CreateBuffer(bytes);
    m_prev                                  = Engine::GetInstance()->CreateBuffer(bytes);
    m_kernel                                = Engine::GetInstance()->GetKernel("wave_step");
}

void Simulator::InitStandingWave() {
    const auto tx = SineTable(m_params.nx), ty = SineTable(m_params.ny), tz = SineTable(m_params.nz);

    // Exact phase advance per leapfrog step: sin^2(theta/2) = C * sum sin^2(pi / (2(n+1)))
    auto h = [](u32 n) {
        double s = std::sin(M_PI / (2.0 * (n + 1)));
        return s * s;
    };
    const double arg = m_params.C * (h(m_params.nx) + h(m_params.ny) + h(m_params.nz));
    assert(arg <= 1.0 && "unstable: reduce the Courant number");
    m_theta      = 2.0 * std::asin(std::sqrt(arg));
    m_steps_done = 0;

    float* cur  = m_curr->As<float>();
    float* prev = m_prev->As<float>();
    std::memset(cur, 0, m_curr->Size()); // ghost layer must be exactly zero
    std::memset(prev, 0, m_prev->Size());

    const double c = std::cos(m_theta); // u at t = -dt is S * cos(theta)
    for (u32 z = 1; z <= m_params.nz; ++z) {
        for (u32 y = 1; y <= m_params.ny; ++y) {
            const double tyz = ty[y] * tz[z];
            for (u32 x = 1; x <= m_params.nx; ++x) {
                const u64    i = wave_idx(x, y, z, m_params.sy, m_params.sz);
                const double s = tx[x] * tyz;
                cur[i]         = (float)s;       // t = 0
                prev[i]        = (float)(s * c); // t = -dt
            }
        }
    }
}


void Simulator::InitPulse(float sigma_cells) {
    float* cur  = m_curr->As<float>();
    float* prev = m_prev->As<float>();
    std::memset(cur,  0, m_curr->Size());
    std::memset(prev, 0, m_prev->Size());

    const double cx = (m_params.nx + 1) * 0.5, cy = (m_params.ny + 1) * 0.5, cz = (m_params.nz + 1) * 0.5;
    const double k  = 1.0 / (2.0 * sigma_cells * sigma_cells);
    for (u32 z = 1; z <= m_params.nz; ++z)
        for (u32 y = 1; y <= m_params.ny; ++y)
            for (u32 x = 1; x <= m_params.nx; ++x) {
                const double r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz);
                const u64 i = wave_idx(x, y, z, m_params.sy, m_params.sz);
                cur[i] = prev[i] = (float)std::exp(-r2 * k);   // equal values: starts at rest
            }
    m_steps_done = 0;
}

RunTiming Simulator::Run(u32 steps, u32 batch) {
    RunTiming timing;
    const auto           start = std::chrono::steady_clock::now();
    
    for (u32 done = 0; done < steps; done += batch) {
        timing.gpu_s += EncodeAndSubmit(std::min(batch, steps - done));
    }
    
    timing.wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return timing;
}

double Simulator::MaxErrorVsExact() {
    const auto   tx = SineTable(m_params.nx), ty = SineTable(m_params.ny), tz = SineTable(m_params.nz);
    const double amp = std::cos(m_theta * (double)m_steps_done); // exact amplitude now
    const float* u   = m_curr->As<float>();                      // newest field after the swaps

    double max_err = 0.0;
    for (u32 z = 1; z <= m_params.nz; ++z) {
        for (u32 y = 1; y <= m_params.ny; ++y) {
            const double tyz = ty[y] * tz[z];
            for (u32 x = 1; x <= m_params.nx; ++x) {
                const u64    i     = wave_idx(x, y, z, m_params.sy, m_params.sz);
                const double exact = amp * tx[x] * tyz;
                max_err            = std::max(max_err, std::abs((double)u[i] - exact));
            }
        }
    }
    return max_err;
}

double Simulator::EncodeAndSubmit(u32 steps) {
    Engine* engine = Engine::GetInstance();
    engine->Begin();
    engine->SetBytes(&m_params, sizeof(m_params), BIND_PARAMS);
    // iterate sim steps
    for (u32 s = 0; s < steps; ++s) {
        engine->SetBuffer(m_curr.get(), BIND_U_CURR);
        engine->SetBuffer(m_prev.get(), BIND_U_PREV);
        engine->Dispatch3D(m_kernel, m_params.nx, m_params.ny, m_params.nz);
        std::swap(m_curr, m_prev);
    }
    m_steps_done += steps;
    return engine->Submit();
}


bool Simulator::DumpSliceZ(u32 z, const char* path) {
    const u32 w = m_params.nx + 2, h = m_params.ny + 2;          // include the ghost layer
    const float* plane = m_curr->As<float>() + (u64)z * m_params.sz;

    FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    const u64 step = m_steps_done;
    std::fwrite(&w,    sizeof(w),    1, f);                      // 16-byte header: w, h, step
    std::fwrite(&h,    sizeof(h),    1, f);
    std::fwrite(&step, sizeof(step), 1, f);
    std::fwrite(plane, sizeof(float), (size_t)w * h, f);
    std::fclose(f);
    return true;
}


bool Simulator::DumpVolume(const char* path, u32 stride) {
    const u32 W = m_params.nx + 2, H = m_params.ny + 2, D = m_params.nz + 2;   // with ghost layer
    const u32 w = (W + stride - 1) / stride;
    const u32 h = (H + stride - 1) / stride;
    const u32 d = (D + stride - 1) / stride;

    const float* u = m_curr->As<float>();
    std::vector<float> out((size_t)w * h * d);
    size_t o = 0;
    for (u32 z = 0; z < D; z += stride)
        for (u32 y = 0; y < H; y += stride)
            for (u32 x = 0; x < W; x += stride)
                out[o++] = u[wave_idx(x, y, z, m_params.sy, m_params.sz)];

    FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    const u64 step = m_steps_done;
    
    std::fwrite(&w, sizeof(u32), 1, f);                    // 24-byte header: w, h, d, stride, step
    std::fwrite(&h, sizeof(u32), 1, f);
    std::fwrite(&d, sizeof(u32), 1, f);
    std::fwrite(&stride, sizeof(u32), 1, f);
    std::fwrite(&step, sizeof(u64), 1, f);
    std::fwrite(out.data(), sizeof(float), out.size(), f);  // x fastest, as VTK expects
    std::fclose(f);
    return true;
}
