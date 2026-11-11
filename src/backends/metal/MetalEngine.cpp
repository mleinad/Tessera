#include "MetalEngine.hpp"
#include "MetalBuffer.hpp"
#include "MetalKernelLibrary.hpp"
#include "common/logger.hpp"
#include "common/WaveCommon.h"

#include <cmath>

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
    log("Memory model        : {}", m_device->hasUnifiedMemory()
                                        ? "unified (CPU and GPU share one memory pool, no transfers)"
                                        : "discrete (separate VRAM)");
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
    m_command_buffer = nullptr;
    m_command_buffer = nullptr;
    m_pool           = nullptr;
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

bool MetalEngine::CanFit(size_t bytes_per_buffer, size_t total_bytes) const {
    return bytes_per_buffer <= m_device->maxBufferLength()
        && total_bytes      <= m_device->recommendedMaxWorkingSetSize();
}
