#ifndef Benchmark_hpp
#define Benchmark_hpp

#include "common/Engine.hpp"
#include "common/WaveCommon.h"

#include <cstdio>
#include <string>
#include <vector>

class Simulator;
struct BenchmarkRow {
    std::string label;          // free text for this experiment, e.g. "sweep-v1"
    std::string backend;        // "metal" | "cuda"
    std::string device;
    std::string memory_model;   // "unified" | "discrete"
    std::string buffer_mode;    // "shared" | "managed" | "device"
    std::string build;          // "release" | "debug": never trust timings from debug builds
    std::string kernel;         // "naive" | "tiled"
    u32    n         = 0;
    double grid_gb   = 0;       // both fields, ghost layer included, decimal GB
    bool   fits      = false;   // grid fits in GPU-usable memory (no paging / oversubscription)
    u32    rep       = 0;
    u32    steps     = 0;
    u32    batch     = 0;
    double gpu_s     = 0;       // GPU execution time for the whole run
    double wall_s    = 0;       // host elapsed time for the whole run
    double ms_step   = 0;       // GPU ms per step
    double wall_ms   = 0;       // wall ms per step
    double gbps      = 0;       // effective bandwidth: 12 B per interior cell per step / GPU time
    double pct_peak  = 0;       // gbps / theoretical peak, 0 if the peak is unknown
    double max_error = -1;      // validation error, -1 if not validated in this run
};

class Benchmark {
public:
    // Appends to csv_path, writing the header first if the file is new or empty.
    explicit Benchmark(const std::string& csv_path, const std::string& label = "");
    ~Benchmark();
    Benchmark(const Benchmark&)            = delete;
    Benchmark& operator=(const Benchmark&) = delete;

    // Runs `steps` steps on sim and records them. With validate = true, also computes
    // sim.MaxErrorVsExact() after the timed run (outside the timing).
    const BenchmarkRow& Measure(Simulator& sim, u32 steps, u32 batch, u32 rep, bool validate = false);

    // Median GB/s and ms/step per grid size, for the rows recorded by this object.
    void PrintSummary() const;

private:
    void WriteRow(const BenchmarkRow& r);

    std::string               m_label;
    FILE*                     m_csv{nullptr};
    std::vector<BenchmarkRow> m_rows;
};
#endif
