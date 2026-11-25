#include "Benchmark.hpp"
#include "sim/Simulator.hpp"
#include "logger.hpp"

#include <algorithm>
#include <map>
constexpr double kBytesPerCell     = 12.0; // per interior cell per step: read u, read+write u_prev
constexpr double kMaskBytesPerCell = 1.0;  // obstacle runs: 7 mask reads per cell, ~1 B from DRAM after caching


// Measure swaps, should use it before and after timing to be sure
static uint64_t MajorFaults() {
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    return (uint64_t)ru.ru_majflt;
}


static double Median(std::vector<double> v) {
    if (v.empty()) {
        return 0.0;
    }
    std::sort(v.begin(), v.end());
    const size_t m = v.size() / 2;
    return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

Benchmark::Benchmark(const std::string& csv_path, const std::string& label) : m_label(label) {
    m_csv = std::fopen(csv_path.c_str(), "a");
    if (!m_csv) {
        log("Benchmark: cannot open {}, logging to console only", csv_path);
        return;
    }
    std::fseek(m_csv, 0, SEEK_END);
    if (std::ftell(m_csv) == 0) {
        std::fprintf(m_csv,
                     "label,backend,device,memory_model,buffer_mode,kernel,build,n,grid_gb,fits_in_gpu_mem,"
                     "rep,steps,batch,gpu_s,wall_s,ms_per_step,wall_ms_per_step,gbps,pct_peak,max_error\n");
    }
}

Benchmark::~Benchmark() {
    if (m_csv) {
        std::fclose(m_csv);
    }
}

const BenchmarkRow& Benchmark::Measure(Simulator& sim, u32 steps, u32 batch, u32 rep, bool validate) {
    const EngineInfo  info = Engine::GetInstance()->Info();
    const WaveParams& p    = sim.GetParams();

    const RunTiming t = sim.Run(steps, batch); // the only timed part

    BenchmarkRow r;
    r.label        = m_label;
    r.backend      = info.backend;
    r.device       = info.device;
    r.memory_model = info.memory_model;
    r.buffer_mode  = info.buffer_mode;
    r.kernel       = sim.KernelName();
#ifdef NDEBUG
    r.build = "release";
#else
    r.build = "debug";
#endif
    r.n        = p.nx;
    r.grid_gb  = 2.0 * (p.nx + 2.0) * (p.ny + 2.0) * (p.nz + 2.0) * sizeof(float) / 1e9;
    // Everything the simulator allocated (fields + mask) must fit, not just the fields
    r.fits     = info.memory_gb <= 0 || Simulator::TotalBytes(p.nx) / 1e9 <= info.memory_gb;
    r.rep      = rep;
    r.steps    = steps;
    r.batch    = batch;
    r.gpu_s    = t.gpu_s;
    r.wall_s   = t.wall_s;
    r.ms_step  = t.gpu_s / steps * 1e3;
    r.wall_ms  = t.wall_s / steps * 1e3;
    const double bytes_per_cell = kBytesPerCell + (sim.HasObstacles() ? kMaskBytesPerCell : 0.0);
    r.gbps     = bytes_per_cell * (double)sim.InteriorCells() * steps / t.gpu_s / 1e9;
    r.pct_peak = info.peak_gbs > 0 ? 100.0 * r.gbps / info.peak_gbs : 0.0;
    validate   = validate && !sim.HasObstacles(); // the exact standing-wave solution has no obstacles
    if (validate) {
        r.max_error = sim.MaxErrorVsExact(); // after timing; CPU loop over the grid
    }

    log("[{}/{} - {}] n={} ({:.2f} GB{}) rep {}: {:.3f} ms/step, {:.1f} GB/s ({:.0f}% of peak), wall {:.3f} ms/step{}{}",
        r.backend,
        r.buffer_mode,
        r.kernel,
        r.n,
        r.grid_gb,
        r.fits ? "" : ", exceeds GPU memory",
        r.rep,
        r.ms_step,
        r.gbps,
        r.pct_peak,
        r.wall_ms,
        r.build == "debug" ? "  [DEBUG BUILD]" : "",
        validate ? std::format(", error {:.2e}", r.max_error) : "");

    WriteRow(r);
    m_rows.push_back(r);
    return m_rows.back();
}

void Benchmark::WriteRow(const BenchmarkRow& r) {
    if (!m_csv) {
        return;
    }
    std::fprintf(m_csv,
                 "%s,%s,\"%s\",%s,%s,%s,%s,%u,%.4f,%d,%u,%u,%u,%.6f,%.6f,%.4f,%.4f,%.2f,%.1f,",
                 r.label.c_str(),
                 r.backend.c_str(),
                 r.device.c_str(),
                 r.memory_model.c_str(),
                 r.buffer_mode.c_str(),
                 r.kernel.c_str(),
                 r.build.c_str(),
                 r.n,
                 r.grid_gb,
                 r.fits ? 1 : 0,
                 r.rep,
                 r.steps,
                 r.batch,
                 r.gpu_s,
                 r.wall_s,
                 r.ms_step,
                 r.wall_ms,
                 r.gbps,
                 r.pct_peak);
    if (r.max_error >= 0) {
        std::fprintf(m_csv, "%.3e", r.max_error); // empty cell if not validated
    }
    std::fprintf(m_csv, "\n");
    std::fflush(m_csv); // keep what's measured even if a later size crashes (e.g. out of memory)
}

void Benchmark::PrintSummary() const {
    // Key (n, kernel): sizes in order, kernels next to each other within each size
    std::map<std::pair<u32, std::string>, std::vector<const BenchmarkRow*>> groups;
    for (const BenchmarkRow& r : m_rows) {
        groups[{r.n, r.kernel}].push_back(&r);
    }

    // Median GB/s of the naive kernel per size, for the "vs naive" column
    std::map<u32, double> naive_gbps;
    for (const auto& [key, rows] : groups) {
        if (key.second == "naive") {
            std::vector<double> g;
            for (const BenchmarkRow* r : rows) {
                g.push_back(r->gbps);
            }
            naive_gbps[key.first] = Median(g);
        }
    }

    log("{:>8} {:>10} {:>9} {:>6} {:>12} {:>10} {:>8} {:>9} {:>10}",
        "n", "kernel", "GB", "reps", "ms/step", "GB/s", "spread", "vs naive", "error");
    for (const auto& [key, rows] : groups) {
        const auto& [n, kernel] = key;

        std::vector<double> gbps, ms;
        double              err = -1;
        for (const BenchmarkRow* r : rows) {
            gbps.push_back(r->gbps);
            ms.push_back(r->ms_step);
            if (r->max_error >= 0) {
                err = r->max_error;
            }
        }
        const auto [lo, hi] = std::minmax_element(gbps.begin(), gbps.end());
        const double med    = Median(gbps);
        const double spread = med > 0 ? 100.0 * (*hi - *lo) / med : 0.0;

        const auto   base    = naive_gbps.find(n);
        std::string  speedup = "-";
        if (kernel != "naive" && base != naive_gbps.end() && base->second > 0) {
            speedup = std::format("{:.2f}x", med / base->second);
        }

        log("{:>8} {:>10} {:>9.2f} {:>6} {:>12.3f} {:>10.1f} {:>7.1f}% {:>9} {:>10}",
            n,
            kernel,
            rows.front()->grid_gb,
            rows.size(),
            Median(ms),
            med,
            spread,
            speedup,
            err >= 0 ? std::format("{:.2e}", err) : "-");
    }
}
