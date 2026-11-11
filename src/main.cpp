#include <iostream>
#include <filesystem>

#include "common/Engine.hpp"
#include "common/logger.hpp"
#include "sim/Simulator.hpp"

#if defined(TESSERA_BACKEND_METAL)
#include <Foundation/Foundation.hpp>
#endif

void RunSineTest() {
    Simulator sim(256, 0.5f);
    sim.InitStandingWave();

    sim.Run(10); // warm-up

    const u32 steps = 200;
    RunTiming t     = sim.Run(steps);

    double bytes = 12.0 * sim.InteriorCells() * steps;

    log("Max error vs exact: {:.2e} | GPU: {:.3f} ms/step ({:.1f} GB/s) | Wall: {:.3f} ms/step",
        sim.MaxErrorVsExact(),
        t.gpu_s / steps * 1e3,
        bytes / t.gpu_s / 1e9,
        t.wall_s / steps * 1e3);
}

void RunVisualization() {
    std::filesystem::create_directories("frames");
    Simulator sim(256, 0.5f);
    sim.InitPulse(3.0f);

    char path[64];
    for (u32 frame = 0; frame < 240; ++frame) {
        std::snprintf(path, sizeof(path), "frames/vol_%04u.bin", frame);
        sim.DumpVolume(path, 2); // middle plane (1-based interior + ghost layer)
        sim.Run(4);                // steps per frame: more = faster motion
    }
}

int main(int argc, const char* argv[]) {
#if defined(TESSERA_BACKEND_METAL)
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
#endif
    Engine::GetInstance()->Initialize();

    RunVisualization();

#if defined(TESSERA_BACKEND_METAL)
    pool->release();
#endif
}
