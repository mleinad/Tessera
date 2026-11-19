#ifndef Simulator_hpp
#define Simulator_hpp
#include "common/WaveCommon.h"
#include "common/Engine.hpp"
#include <memory>
#include <stdio.h>

struct RunTiming {
    double gpu_s  = 0.0; // GPU execution time, summed over batches
    double wall_s = 0.0; // host elapsed time for the whole run
};

class Buffer;
class Kernel;
class Simulator {
public:
    Simulator(u32 n, float courant);

    // Memory a Simulator(n) allocates, ghost layer included: two float32 fields + the u8 obstacle mask.
    // Pass to Engine::CheckFit before constructing.
    static size_t FieldBytes(u32 n) {
        return (size_t)(n + 2) * (n + 2) * (n + 2) * sizeof(float);
    }
    static size_t TotalBytes(u32 n) {
        return 2 * FieldBytes(n) + (size_t)(n + 2) * (n + 2) * (n + 2);
    }

    void InitStandingWave();

    void InitPulse(float sigma_cells, double cx, double cy, double cz);
    void InitPulse(float sigma_cells);

    RunTiming Run(u32 steps, u32 batch = 50); // returns GPU seconds for the batch
    double    MaxErrorVsExact();

    double EncodeAndSubmit(u32 steps);

    u64 InteriorCells() const {
        return (u64)m_params.nx * m_params.ny * m_params.nz;
    }

    bool LoadMask(const char* path);

    bool DumpSliceZ(u32 z, const char* path);
    bool DumpVolume(const char* path, u32 stride);

    WaveParams GetParams() const {
        return m_params;
    }
private:
    WaveParams              m_params{};
    std::unique_ptr<Buffer> m_curr;
    std::unique_ptr<Buffer> m_prev;
    Kernel*                 m_kernel{nullptr};
    u64                     m_steps_done{0};
    double                  m_theta{0}; // phase advance per step

    std::unique_ptr<Buffer> m_mask;
    Kernel*                 m_kernel_obstacles{nullptr};
    bool                    m_has_obstacles = false;
};

#endif /* Simulator_hpp */
