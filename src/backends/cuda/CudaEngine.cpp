#include "CudaEngine.hpp"
#include "common/logger.hpp"
#include "common/WaveCommon.h"

#include <cmath>
#include <cuda_runtime.h>

CudaEngine* CudaEngine::m_instance = nullptr;

// The shared code asks for Engine::GetInstance(); the CUDA build answers with CudaEngine
Engine* Engine::GetInstance() {
    return CudaEngine::GetInstance();
}

CudaEngine::CudaEngine()
{
}

CudaEngine::~CudaEngine()
{
}

bool CudaEngine::Initialize() {
    if (m_initialized) {
        return false;
    }

    cudaError_t err = cudaSetDevice(m_device); // also creates the CUDA context
    if (err != cudaSuccess) {
        log("CUDA init failed: {}", cudaGetErrorString(err));
        return false;
    }
    
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, m_device);

    size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    m_initialized = true;

    // Peak DRAM bandwidth: the stencil is memory-bound, so this is the roofline to compare GB/s against
    int mem_clock_khz = 0, bus_width_bits = 0;
    cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, m_device);
    cudaDeviceGetAttribute(&bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, m_device);
    const double peak_gbs = 2.0 * mem_clock_khz * 1e3 * (bus_width_bits / 8.0) / 1e9; // x2: double data rate

    int driver = 0, runtime = 0;
    cudaDriverGetVersion(&driver);
    cudaRuntimeGetVersion(&runtime);

    // Largest cubic grid whose two float32 fields fit in free VRAM: 2 * (n+2)^3 * 4 bytes
    const u64 n_fit = (u64)std::cbrt(free_bytes / 8.0) - 2;

    constexpr double GB = 1024.0 * 1024.0 * 1024.0;

    log("==== Tessera | CUDA backend ====");
    log("Device              : {} (sm_{}{}, {} SMs)", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    log("Driver / runtime    : {}.{} / {}.{}", driver / 1000, (driver % 1000) / 10, runtime / 1000, (runtime % 1000) / 10);
    log("Memory model        : {}", prop.integrated
                                        ? "integrated (GPU shares system RAM)"
                                        : "discrete (separate VRAM; managed memory migrates pages over PCIe)");
    log("VRAM                : {:.2f} GB total, {:.2f} GB free", total_bytes / GB, free_bytes / GB);
    log("Peak DRAM bandwidth : {:.0f} GB/s", peak_gbs);
    log("Oversubscription    : {}", prop.concurrentManagedAccess
                                        ? "supported (managed memory can exceed VRAM)"
                                        : "NOT supported (grids larger than VRAM will fail)");
    log("Largest in-VRAM grid: {}^3 interior cells (2 x float32 fields)", n_fit);
    if (prop.major * 10 + prop.minor != 86) {
        log("WARNING: kernels are built for sm_86 but this device is sm_{}{}", prop.major, prop.minor);
    }

    return true;
}
CudaEngine* CudaEngine::GetInstance(){
 if (!m_instance) {
        m_instance = new CudaEngine();
    }
return m_instance;
}

std::unique_ptr<Buffer> CudaEngine::CreateBuffer(size_t bytes) {
    return nullptr;
}
Kernel* CudaEngine::GetKernel(const std::string &name) {
    return nullptr;
}

bool CudaEngine::Begin() {
    return true;
}
void CudaEngine::SetBuffer(Buffer *buffer, uint32_t index){

}
void CudaEngine::SetBytes(const void *data, size_t bytes, uint32_t index){

}

void CudaEngine::Dispatch3D(Kernel *kernel, size_t x, size_t y, size_t z) {

}

double CudaEngine::Submit(){
    return 0.0f;
}

bool CudaEngine::CanFit(size_t bytes_per_buffer, size_t total_bytes) const {
    
    return true;
}