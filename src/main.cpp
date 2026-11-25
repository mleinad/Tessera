#include <iostream>
#include <filesystem>
#include <vector>

#include "common/Benchmark.hpp"
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

static u32 SizeForGB(double gb) {
    const double target = gb * 1e9;
    u32 lo = 1, hi = 1u << 13;
    while (lo < hi) {
        const u32 mid = lo + (hi - lo + 1) / 2;
        if (static_cast<double>(Simulator::TotalBytes(mid)) <= target) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}

void RunDeviceMemoryTest(const char* mask_path) {
    std::filesystem::create_directories("results");

    // results/oversub_bunny_960.csv, label "oversub-bunny_960-v1"
    const std::string model = std::filesystem::path(mask_path).stem().string();
    const std::string csv   = "results/oversub_" + model + ".csv";
    const std::string label = "oversub-" + model + "-v1";
    
    Benchmark bench(csv, label);
    log("recording to {}", csv);

    const double sizes_gb[] = {1.2, 2.5, 3.8};
    const u32 reps    = 3;
    const u32 steps   = 10;
    const u32 batch   = 5;

    for (double gb : sizes_gb) {
        const u32 n = SizeForGB(gb);    
        log("target {:.1f} GB -> n={} ({:.2f} GB actual)", gb, n, Simulator::TotalBytes(n) / 1e9);


        const FitReport fit = Engine::GetInstance()->CheckFit(Simulator::FieldBytes(n), Simulator::TotalBytes(n));
        if (!fit.safe) {
            log("skip n={}: {:.2f} GB needed, {:.2f} GB host RAM available: {}",
                n, fit.host_needed / 1e9, fit.host_available / 1e9, fit.reason);
            continue;
        }

        Simulator sim(n, 0.5f);
        if (!sim.LoadMask(mask_path)) { // resampled from the model's grid to n
            log("could not load {}", mask_path);
            return;
        }
        const double mid = (n + 1) * 0.5;
        sim.InitPulse(3.0f, n / 5.0, mid, mid);

        sim.Run(2, 1); // warm-up: the first steps migrate every page that was created on the CPU

        for (u32 r = 0; r < reps; ++r) {
            bench.Measure(sim, steps, batch, r);
        }
    }
    bench.PrintSummary();
}


void RunMaskedSimulation(const char* mask_path) {
    std::filesystem::create_directories("frames");

    const u32 n = 256; // must match the n the mask was voxelized for
    Simulator sim(n, 0.5f);

    // Load the mask before initializing, so the pulse skips solid cells
    if (!sim.LoadMask(mask_path)) {
        log("could not load {}\n", mask_path);
        return;
    }

    // Pulse on the centre line, a fifth of the way along x: beside a model centred in the box
    const double mid = (n + 1) * 0.5;
    sim.InitPulse(3.0f, n / 5.0, mid, mid);

    char path[64];
    for (u32 frame = 0; frame < 240; ++frame) {
        std::snprintf(path, sizeof(path), "frames/vol_%04u.bin", frame);
        sim.DumpVolume(path, 2); // whole volume, every 2nd cell per axis
        sim.Run(4);              // steps per frame: more = faster motion
    }
}

void RunOversubscriptionTest(const char* mask_path) {
    std::filesystem::create_directories("results");

    // results/oversub_bunny_960.csv, label "oversub-bunny_960-v1"
    const std::string model = std::filesystem::path(mask_path).stem().string();
    const std::string csv   = "results/oversub_" + model + ".csv";
    const std::string label = "oversub-" + model + "-v1";
    
    Benchmark bench(csv, label);
    log("recording to {}", csv);

    const u32 sizes[] = {512, 640, 768, 864, 928, 960, 992, 1024}; // 1.2 .. 4.1 GB incl. mask
    const u32 reps    = 3;
    const u32 steps   = 10;
    const u32 batch   = 5;

    for (u32 n : sizes) {
        const FitReport fit = Engine::GetInstance()->CheckFit(Simulator::FieldBytes(n), Simulator::TotalBytes(n));
        if (!fit.safe) {
            log("skip n={}: {:.2f} GB needed, {:.2f} GB host RAM available: {}",
                n, fit.host_needed / 1e9, fit.host_available / 1e9, fit.reason);
            continue;
        }

        Simulator sim(n, 0.5f);
        if (!sim.LoadMask(mask_path)) { // resampled from the model's grid to n
            log("could not load {}", mask_path);
            return;
        }
        const double mid = (n + 1) * 0.5;
        sim.InitPulse(3.0f, n / 5.0, mid, mid);

        sim.Run(2, 1); // warm-up: the first steps migrate every page that was created on the CPU

        for (u32 r = 0; r < reps; ++r) {
            bench.Measure(sim, steps, batch, r);
        }
    }
    bench.PrintSummary();
}

void RunBasicBenchmark() {
    struct Row {
        u32    n;
        double gb;
        bool   fast; // whole working set in fast GPU memory (VRAM / Metal working set)
        u32    rep;
        double ms, gbps, err;
    };

    std::vector<Row> rows;

    const u32 sizes[] = {256, 512};
    const u32 reps    = 5;
    const u32 steps   = 20;

    for (u32 n : sizes) {
        double gb = 2.0 * (n + 2.0) * (n + 2.0) * (n + 2.0) * 4.0 / 1e9; // both fields

        // Skip sizes that would exhaust memory instead of crashing / being OOM-killed
        const FitReport fit = Engine::GetInstance()->CheckFit(Simulator::FieldBytes(n), Simulator::TotalBytes(n));
        if (!fit.safe) {
            log("skip n={}: needs {:.2f} GB host RAM (+1 GB headroom), {:.2f} GB available: {}",
                n, fit.host_needed / 1e9, fit.host_available / 1e9, fit.reason);
            continue;
        }

        Simulator sim(n, 0.5f);
        sim.InitStandingWave();
        sim.Run(5, 5); // warm-up

        const size_t first = rows.size();
        for (u32 r = 0; r < reps; ++r) {
            RunTiming t    = sim.Run(steps, 5); // small batches: large steps take ~50 ms each
            double    ms   = t.gpu_s / steps * 1e3;
            double    gbps = 12.0 * sim.InteriorCells() * steps / t.gpu_s / 1e9;
            rows.push_back({n, gb, fit.fits_fast, r, ms, gbps, 0.0});
        }

        const double err = sim.MaxErrorVsExact(); // after the timed runs, outside timing
        for (size_t i = first; i < rows.size(); ++i) {
            rows[i].err = err;
        }
    }

    std::printf("n,GB,fast,rep,ms_per_step,GBps,max_error\n");
    for (const Row& row : rows) {
        std::printf("%u,%.2f,%d,%u,%.3f,%.1f,%.2e\n", row.n, row.gb, row.fast, row.rep, row.ms, row.gbps, row.err);
    }
}

int main(int argc, const char* argv[]) {
#if defined(TESSERA_BACKEND_METAL)
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
#endif
    Engine::GetInstance()->Initialize();
#if defined(TESSERA_BACKEND_METAL)
    RunOversubscriptionTest("/Users/daniel.malheiro/Documents/fcul/PPC/Projects/Tessera/tools/bunny_256.nrrd");
    RunOversubscriptionTest("/Users/daniel.malheiro/Documents/fcul/PPC/Projects/Tessera/tools/bunny_920.nrrd");
    RunOversubscriptionTest("/Users/daniel.malheiro/Documents/fcul/PPC/Projects/Tessera/tools/bunny_960.nrrd");
#elif defined(TESSERA_BACKEND_CUDA)
    RunDeviceMemoryTest("/home/daniel/Dev/Tessera/tools/bunny_256.nrrd");
    //RunOversubscriptionTest("/home/daniel/Dev/Tessera/tools/bunny_920.nrrd");
    //RunOversubscriptionTest("/home/daniel/Dev/Tessera/tools/bunny_960.nrrd");
#endif

#if defined(TESSERA_BACKEND_METAL)
    pool->release();
#endif
}
