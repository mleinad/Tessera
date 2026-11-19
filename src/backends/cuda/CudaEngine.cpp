#include "CudaEngine.hpp"
#include "common/logger.hpp"
#include "common/WaveCommon.h"

#include "CudaHelper.hpp"
#include "CudaBuffer.hpp"
#include "CudaKernelLibrary.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

CudaEngine *CudaEngine::m_instance = nullptr;

// The shared code asks for Engine::GetInstance(); the CUDA build answers with CudaEngine
Engine *Engine::GetInstance()
{
    return CudaEngine::GetInstance();
}

CudaEngine::CudaEngine()
{
}

CudaEngine::~CudaEngine()
{
}

bool CudaEngine::Initialize()
{
    if (m_initialized)
    {
        return false;
    }

    CUDA_CHECK(cudaSetDevice(m_device)); // also creates the CUDA context

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, m_device);

    size_t free_bytes = 0;
    size_t total_bytes = 0;

    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));

    CUDA_CHECK(cudaStreamCreate(&m_stream));
    CUDA_CHECK(cudaEventCreate(&m_start));
    CUDA_CHECK(cudaEventCreate(&m_stop));
    m_kernels = std::make_unique<CudaKernelLibrary>();
    m_initialized = true;

    if (const char *mem = std::getenv("TESSERA_CUDA_MEMORY"); mem && std::strcmp(mem, "device") == 0)
    {
        m_buffer_mode = CudaBuffer::Mode::Device;
    }

    // Peak DRAM bandwidth: the stencil is memory-bound, so this is the roofline to compare GB/s against
    int mem_clock_khz = 0, bus_width_bits = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, m_device));
    CUDA_CHECK(cudaDeviceGetAttribute(&bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, m_device));
    const double peak_gbs = 2.0 * mem_clock_khz * 1e3 * (bus_width_bits / 8.0) / 1e9; // x2: double data rate

    int driver = 0;
    int runtime = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driver));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));

    // Largest cubic grid whose two float32 fields fit in free VRAM: 2 * (n+2)^3 * 4 bytes
    const u64 n_fit = (u64)std::cbrt(free_bytes / 8.0) - 2;

    constexpr double GB = 1024.0 * 1024.0 * 1024.0;

    log("==== Tessera | CUDA backend ====");
    log("Device              : {} (sm_{}{}, {} SMs)", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    log("Driver / runtime    : {}.{} / {}.{}", driver / 1000, (driver % 1000) / 10, runtime / 1000, (runtime % 1000) / 10);
    log("Memory model        : {}", prop.integrated ? "integrated (GPU shares system RAM)" : "discrete (separate VRAM; managed memory migrates pages over PCIe)");
    log("VRAM                : {:.2f} GB total, {:.2f} GB free", total_bytes / GB, free_bytes / GB);    log("Peak DRAM bandwidth : {:.0f} GB/s", peak_gbs);
    log("Oversubscription    : {}", prop.concurrentManagedAccess ? "supported (managed memory can exceed VRAM)" : "NOT supported (grids larger than VRAM will fail)");
    log("Largest in-VRAM grid: {}^3 interior cells (2 x float32 fields)", n_fit);
    log("Buffer memory       : {}",
        m_buffer_mode == CudaBuffer::Mode::Device ? "device (VRAM only, cudaMalloc; capped by free VRAM)" : "managed (cudaMallocManaged; can oversubscribe)");
    if (prop.major * 10 + prop.minor != 86)
    {
        log("WARNING: kernels are built for sm_86 but this device is sm_{}{}", prop.major, prop.minor);
    }

    m_info.backend = "cuda";
    m_info.device = prop.name;
    m_info.memory_model = prop.integrated ? "unified" : "discrete";
    m_info.buffer_mode = m_buffer_mode == CudaBuffer::Mode::Device ? "device" : "managed";
    m_info.driver = std::to_string(driver) + " / " + std::to_string(runtime);
    m_info.peak_gbs = peak_gbs;
    m_info.memory_gb = free_bytes / 1e9; // what the grid can use without oversubscribing

    return true;
}
CudaEngine *CudaEngine::GetInstance()
{
    if (!m_instance)
    {
        m_instance = new CudaEngine();
    }
    return m_instance;
}

std::unique_ptr<Buffer> CudaEngine::CreateBuffer(size_t bytes)
{
    return std::make_unique<CudaBuffer>(bytes, m_buffer_mode);
}
Kernel *CudaEngine::GetKernel(const std::string &name)
{
    return m_kernels->Get(name);
}

bool CudaEngine::Begin()
{
    CUDA_CHECK(cudaEventRecord(m_start, m_stream));
    return true;
}
void CudaEngine::SetBuffer(Buffer *buffer, uint32_t index)
{
    assert(index < kMaxBinds);
    m_buffer_slots[index] = static_cast<CudaBuffer *>(buffer)->DevicePtr(); // Device mode: uploads if the host copy is newer
    m_args[index] = &m_buffer_slots[index];
}
void CudaEngine::SetBytes(const void *data, size_t bytes, uint32_t index)
{
    assert(index < kMaxBinds && bytes <= kMaxBindBytes);
    std::memcpy(m_byte_slots[index], data, bytes);
    m_args[index] = m_byte_slots[index];
}

void CudaEngine::Dispatch3D(Kernel *kernel, size_t x, size_t y, size_t z)
{
    // 32 wide in x (one warp, contiguous in memory) like Metal's threadExecutionWidth; 32 x 8 = 256 threads
    const dim3 block(32, 8, 1);
    const dim3 grid((unsigned)((x + block.x - 1) / block.x), (unsigned)((y + block.y - 1) / block.y), (unsigned)z);
    CUDA_CHECK(cudaLaunchKernel(static_cast<CudaKernel *>(kernel)->Get(), grid, block, m_args, 0, m_stream));
}

double CudaEngine::Submit()
{
    CUDA_CHECK(cudaEventRecord(m_stop, m_stream));
    CUDA_CHECK(cudaEventSynchronize(m_stop)); // host may touch managed buffers after this
    CUDA_CHECK(cudaGetLastError());           // errors raised while the kernels ran

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, m_start, m_stop));
    return ms / 1e3;
}

// Host RAM the kernel can hand out without swapping (MemAvailable). Swap is not counted:
// managed pages in use by the driver are not reliably swappable.
static size_t AvailableHostBytes()
{
    std::FILE *f = std::fopen("/proc/meminfo", "r");
    if (!f)
    {
        return 0;
    }
    char line[256];
    size_t kb = 0;
    while (std::fgets(line, sizeof(line), f))
    {
        if (std::sscanf(line, "MemAvailable: %zu kB", &kb) == 1)
        {
            break;
        }
    }
    std::fclose(f);
    return kb * 1024;
}

FitReport CudaEngine::CheckFit(size_t bytes_per_buffer, size_t total_bytes) const
{
    (void)bytes_per_buffer; // CUDA has no per-allocation limit below free memory

    size_t free_vram = 0, total_vram = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_vram, &total_vram));

    FitReport r;
    r.fits_fast = total_bytes <= free_vram;
    r.host_available = AvailableHostBytes();

    // The simulator initialises and validates on the CPU, so every page is in host RAM at some point:
    // Managed pages are first created on the CPU, Device mode keeps a host mirror of each buffer.
    // Peak host use is therefore the whole working set. Lower this to (total - free VRAM) once
    // initialisation and validation run on the GPU.
    r.host_needed = total_bytes;

    if (m_buffer_mode == CudaBuffer::Mode::Device && !r.fits_fast)
    {
        r.reason = "device mode needs the whole working set in free VRAM";
    }
    else if (r.host_needed + kHostHeadroom > r.host_available)
    {
        r.reason = "not enough host RAM (the process would be OOM-killed)";
    }
    else
    {
        r.safe = true;
    }
    return r;
}
