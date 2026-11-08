#include "Engine.hpp"
#include "Buffer.hpp"
#include "KernelLibrary.hpp"
#include "../Common/logger.hpp"

Engine* Engine::m_instance = nullptr;

Engine::Engine() {
}

Engine::~Engine() {
}

Engine* Engine::GetInstance() {
    if (!m_instance) {
        m_instance = new Engine(); // declares a local; the static stays null
    }
    return m_instance;
}

bool Engine::Initialize() {
    if (m_initialized) {
        return false;
    }
    m_device = MTL::CreateSystemDefaultDevice();
    assert(m_device);
    m_command_queue = m_device->newCommandQueue();
    m_kernels       = std::make_unique<KernelLibrary>(m_device);
    m_initialized   = true;
    
    constexpr double MB = 1024.0 * 1024.0;

    log("Device: {}", m_device->name()->utf8String());
    log("Unified memory: {}", m_device->hasUnifiedMemory() ? "yes" : "no");
    log("Recommended max working set size: {:.0f} MB", m_device->recommendedMaxWorkingSetSize() / MB);
    log("Max buffer length: {:.0f} MB", m_device->maxBufferLength() / MB);
    
    return true;
}

bool Engine::Begin() {
    m_pool                    = NS::AutoreleasePool::alloc()->init(); // per-batch objects die with this pool
    m_command_buffer          = m_command_queue->commandBuffer();
    m_compute_command_encoder = m_command_buffer->computeCommandEncoder(); // serial by default: dispatches run in order

    return true;
}

double Engine::Submit() {
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

void Engine::Dispatch3D(MTL::ComputePipelineState* pso, MTL::Size grid) {
    NS::UInteger w = pso->threadExecutionWidth(); // 32 on Apple GPUs
    NS::UInteger h = pso->maxTotalThreadsPerThreadgroup() / w;
    
    m_compute_command_encoder->setComputePipelineState(pso);
    m_compute_command_encoder->dispatchThreads(grid, MTL::Size(w, h, 1)); // x is contiguous in memory
}

bool Engine::CanFit(size_t bytes_per_buffer, size_t total_bytes) const {
    return bytes_per_buffer <= m_device->maxBufferLength()
        && total_bytes      <= m_device->recommendedMaxWorkingSetSize();
}
