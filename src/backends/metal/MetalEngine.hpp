#ifndef MetalEngine_hpp
#define MetalEngine_hpp

#include <stdio.h>
#include "Metal/Metal.hpp"
#include "common/Engine.hpp"

class MetalKernelLibrary;
class MetalEngine : public Engine {
public:
    MetalEngine();
    ~MetalEngine() override;

    MetalEngine(const MetalEngine&)            = delete;
    MetalEngine(MetalEngine&&)                 = delete;
    MetalEngine& operator=(MetalEngine const&) = delete;
    MetalEngine& operator=(MetalEngine&&)      = delete;

    bool                Initialize() override;
    static MetalEngine* GetInstance(); // hides Engine::GetInstance with the concrete type

    MTL::Device* GetDevice() {
        assert(m_initialized && "Engine::Initialize() not called");
        return m_device;
    }

    MetalKernelLibrary& Kernels() {
        return *m_kernels;
    }

    MTL::ComputeCommandEncoder* Encoder() {
        return m_compute_command_encoder;
    }

    // Engine interface
    std::unique_ptr<Buffer> CreateBuffer(size_t bytes) override;
    Kernel*                 GetKernel(const std::string& name) override;

    bool   Begin() override;
    void   SetBuffer(Buffer* buffer, uint32_t index) override;
    void   SetBytes(const void* data, size_t bytes, uint32_t index) override;
    void   Dispatch3D(Kernel* kernel, size_t x, size_t y, size_t z) override;
    double Submit() override;

    void Dispatch3D(MTL::ComputePipelineState* pso, MTL::Size grid);

    bool CanFit(size_t bytes_per_buffer, size_t total_bytes) const override;

private:
    MTL::Device*                        m_device{nullptr};
    MTL::CommandQueue*                  m_command_queue{nullptr};
    std::unique_ptr<MetalKernelLibrary> m_kernels;

    NS::AutoreleasePool*        m_pool{nullptr};
    MTL::CommandBuffer*         m_command_buffer{nullptr};
    MTL::ComputeCommandEncoder* m_compute_command_encoder{nullptr};
    bool                        m_initialized{false};

    static MetalEngine* m_instance;
};

#endif /* MetalEngine_hpp */
