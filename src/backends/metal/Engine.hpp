#ifndef Engine_hpp
#define Engine_hpp

#include <stdio.h>
#include "Metal/Metal.hpp"

class Buffer;
class KernelLibrary;
class Engine {
public:
    Engine();
    ~Engine();

    Engine(const Engine&)            = delete;
    Engine(Engine&&)                 = delete;
    Engine& operator=(Engine const&) = delete;
    Engine& operator=(Engine&&)      = delete;

    bool           Initialize();
    static Engine* GetInstance();

    MTL::Device* GetDevice() {
        assert(m_initialized && "Engine::Initialize() not called");
        return m_device;
    }

    KernelLibrary& Kernels() {
        return *m_kernels;
    }

    MTL::ComputeCommandEncoder* Encoder() {
        return m_compute_command_encoder;
    }

    bool   Begin();
    double Submit();
    
    void Dispatch3D(MTL::ComputePipelineState* pso, MTL::Size grid);
    
    bool CanFit(size_t bytes_per_buffer, size_t total_bytes) const;

private:
    MTL::Device*                   m_device{nullptr};
    MTL::CommandQueue*             m_command_queue{nullptr};
    std::unique_ptr<KernelLibrary> m_kernels;

    NS::AutoreleasePool*        m_pool{nullptr};
    MTL::CommandBuffer*         m_command_buffer{nullptr};
    MTL::ComputeCommandEncoder* m_compute_command_encoder{nullptr};
    bool                        m_initialized{false};

    static Engine* m_instance;
};

#endif /* Engine_hpp */
