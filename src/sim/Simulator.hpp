#ifndef Simulator_hpp
#define Simulator_hpp
#include "common/WaveCommon.h"
#include "Metal/Metal.hpp"
#include <stdio.h>

struct RunTiming {
    double gpu_s  = 0.0; // GPU execution time, summed over batches
    double wall_s = 0.0; // host elapsed time for the whole run
};

class Buffer;
class Simulator {
public:
    Simulator(u32 n, float courant);

    void InitStandingWave();
    void InitPulse(float sigma_cells);

    RunTiming Run(u32 steps, u32 batch = 50); // returns GPU seconds for the batch
    double    MaxErrorVsExact();

    double EncodeAndSubmit(u32 steps);

    u64 InteriorCells() const {
        return (u64)m_params.nx * m_params.ny * m_params.nz;
    }

    bool DumpSliceZ(u32 z, const char* path);
    bool DumpVolume(const char* path, u32 stride);
    
private:
    WaveParams                 m_params{};
    std::unique_ptr<Buffer>    m_curr;
    std::unique_ptr<Buffer>    m_prev;
    MTL::ComputePipelineState* m_pso;
    u64                        m_steps_done{0};
    double                     m_theta{0}; // phase advance per step (see validation)
};

#endif /* Simulator_hpp */
