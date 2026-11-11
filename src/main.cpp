#include <iostream>
#include <filesystem>

#include "common/Engine.hpp"
#include "common/logger.hpp"
#include "sim/Simulator.hpp"

#if defined(TESSERA_BACKEND_METAL)
#include "Foundation/Foundation.hpp"
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
        sim.Run(4);              // steps per frame: more = faster motion
    }
}

void Benchmark() {
    struct Row {
        u32    n;
        double gb;
        u32    rep;
        double ms, gbps, err;
    };

    std::vector<Row> rows;

    const u32 sizes[] = {256, 512, 800, 1000, 1150};
    const u32 reps    = 5;
    const u32 steps   = 20;

    for (u32 n : sizes) {
        double gb = 2.0 * (n + 2.0) * (n + 2.0) * (n + 2.0) * 4.0 / 1e9; // both fields

        Simulator sim(n, 0.5f);
        sim.InitStandingWave();
        sim.Run(5, 5); // warm-up

        const size_t first = rows.size();
        for (u32 r = 0; r < reps; ++r) {
            RunTiming t    = sim.Run(steps, 5); // small batches: large steps take ~50 ms each
            double    ms   = t.gpu_s / steps * 1e3;
            double    gbps = 12.0 * sim.InteriorCells() * steps / t.gpu_s / 1e9;
            rows.push_back({n, gb, r, ms, gbps, 0.0});
        }

        const double err = sim.MaxErrorVsExact(); // after the timed runs, outside timing
        for (size_t i = first; i < rows.size(); ++i) {
            rows[i].err = err;
        }
    }

    std::printf("n,GB,rep,ms_per_step,GBps,max_error\n");
    for (const Row& row : rows) {
        std::printf("%u,%.2f,%u,%.3f,%.1f,%.2e\n", row.n, row.gb, row.rep, row.ms, row.gbps, row.err);
    }
}

int main(int argc, const char* argv[]) {
#if defined(TESSERA_BACKEND_METAL)
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
#endif
    Engine::GetInstance()->Initialize();

    Benchmark();

#if defined(TESSERA_BACKEND_METAL)
    pool->release();
#endif
}
