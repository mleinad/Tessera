#include "MetalEngine.hpp"
#include "MetalBuffer.hpp"
#include "MetalKernelLibrary.hpp"
#include "common/logger.hpp"
#include "common/WaveCommon.h"

#include <cassert>
#include <cmath>
#include <mach/mach.h> // host_statistics64: free RAM for CheckFit

MetalEngine* MetalEngine::m_instance = nullptr;

// The shared code asks for Engine::GetInstance(); the Metal build answers with MetalEngine
Engine* Engine::GetInstance() {
    return MetalEngine::GetInstance();
}

MetalEngine::MetalEngine() {
}

MetalEngine::~MetalEngine() {
}

MetalEngine* MetalEngine::GetInstance() {
    if (!m_instance) {
        m_instance = new MetalEngine(); // declares a local; the static stays null
    }
    return m_instance;
}

bool MetalEngine::Initialize() {
    if (m_initialized) {
        return false;
    }
    m_device = MTL::CreateSystemDefaultDevice();

    assert(m_device);
    if (!m_device) {
        log("Metal: no device available");
        return false;
    }

    m_info.backend      = "metal";
    m_info.device       = m_device->name()->utf8String();
    m_info.memory_model = m_device->hasUnifiedMemory() ? "unified" : "discrete";
    m_info.buffer_mode  = "shared";
    m_info.peak_gbs     = m_info.device.find("M4 Pro") != std::string::npos ? 273.0 : 0.0;   // Apple's spec
    m_info.memory_gb    = m_device->recommendedMaxWorkingSetSize() / 1e9;

    m_command_queue = m_device->newCommandQueue();
    m_kernels       = std::make_unique<MetalKernelLibrary>(m_device);
    m_initialized   = true;

    const double working_set = (double)m_device->recommendedMaxWorkingSetSize();
    const double ram         = (double)NS::ProcessInfo::processInfo()->physicalMemory();

    // Largest cubic grid whose two float32 fields fit in the working set: 2 * (n+2)^3 * 4 bytes
    const u64 n_fit = (u64)std::cbrt(working_set / 8.0) - 2;

    constexpr double GB = 1024.0 * 1024.0 * 1024.0;

    log("==== Tessera | Metal backend ====");
    log("Device              : {}", m_device->name()->utf8String());
    log("Memory model        : {}", m_device->hasUnifiedMemory() ? "unified (CPU and GPU share one memory pool, no transfers)" : "discrete (separate VRAM)");
    log("Physical RAM        : {:.2f} GB", ram / GB);
    log("GPU working set     : {:.2f} GB (recommended max before paging)", working_set / GB);
    log("Max buffer length   : {:.2f} GB (per field)", m_device->maxBufferLength() / GB);
    log("Largest in-set grid : {}^3 interior cells (2 x float32 fields)", n_fit);

    return true;
}

std::unique_ptr<Buffer> MetalEngine::CreateBuffer(size_t bytes) {
    return std::make_unique<MetalBuffer>(bytes);
}

Kernel* MetalEngine::GetKernel(const std::string& name) {
    return m_kernels->Get(name);
}

bool MetalEngine::Begin() {
    m_pool                    = NS::AutoreleasePool::alloc()->init(); // per-batch objects die with this pool
    m_command_buffer          = m_command_queue->commandBuffer();
    m_compute_command_encoder = m_command_buffer->computeCommandEncoder(); // serial by default: dispatches run in order

    return true;
}

void MetalEngine::SetBuffer(Buffer* buffer, uint32_t index) {
    m_compute_command_encoder->setBuffer(static_cast<MetalBuffer*>(buffer)->Get(), 0, index);
}

void MetalEngine::SetBytes(const void* data, size_t bytes, uint32_t index) {
    m_compute_command_encoder->setBytes(data, bytes, index);
}

double MetalEngine::Submit() {
    m_compute_command_encoder->endEncoding();
    m_command_buffer->commit();
    m_command_buffer->waitUntilCompleted();
    if (m_command_buffer->status() == MTL::CommandBufferStatusError) {
        log("GPU error: {}", m_command_buffer->error()->localizedDescription()->utf8String());
    }

    // extract out in seperate metrics layer
    double gpu = m_command_buffer->GPUEndTime() - m_command_buffer->GPUStartTime();
    m_pool->release(); // frees m_cmd and m_encoder

    m_command_buffer          = nullptr;
    m_compute_command_encoder = nullptr;
    m_pool                    = nullptr;

    return gpu;
}

void MetalEngine::Dispatch3D(Kernel* kernel, size_t x, size_t y, size_t z) {
    Dispatch3D(static_cast<MetalKernel*>(kernel)->Get(), MTL::Size(x, y, z));
}

void MetalEngine::Dispatch3D(MTL::ComputePipelineState* pso, MTL::Size grid) {
    NS::UInteger w = pso->threadExecutionWidth(); // 32 on Apple GPUs
    NS::UInteger h = pso->maxTotalThreadsPerThreadgroup() / w;

    m_compute_command_encoder->setComputePipelineState(pso);
    m_compute_command_encoder->dispatchThreads(grid, MTL::Size(w, h, 1)); // x is contiguous in memory
}

// RAM macOS can hand out without swapping: free + inactive + purgeable + speculative pages.
// Conservative: the memory compressor can usually free a bit more.
static size_t AvailableHostBytes() {
    vm_size_t page = 0;
    host_page_size(mach_host_self(), &page);

    vm_statistics64_data_t vm{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vm, &count) != KERN_SUCCESS) {
        return 0;
    }
    return (size_t)(vm.free_count + vm.inactive_count + vm.purgeable_count + vm.speculative_count) * page;
}

FitReport MetalEngine::CheckFit(size_t bytes_per_buffer, size_t total_bytes) const {
    FitReport r;
    // Unified memory: GPU buffers are host RAM. Beyond the recommended working set the GPU still works,
    // but the OS may start paging, so that is the "fast memory" limit.
    r.fits_fast      = total_bytes <= m_device->recommendedMaxWorkingSetSize();
    r.host_needed    = total_bytes;
    r.host_available = AvailableHostBytes();

    if (bytes_per_buffer > m_device->maxBufferLength()) {
        r.reason = "a buffer exceeds maxBufferLength (newBuffer would fail)";
    } else if (r.host_needed + kHostHeadroom > r.host_available) {
        r.reason = "not enough free RAM (macOS would swap heavily or kill the process)";
    } else {
        r.safe = true;
    }
    return r;
}



