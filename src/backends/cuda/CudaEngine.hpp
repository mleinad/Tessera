#ifndef CudaEngine_hpp
#define CudaEngine_hpp

#include "common/Engine.hpp"

class CudaEngine : public Engine {
public:
    CudaEngine();
    ~CudaEngine() override;

    CudaEngine(const CudaEngine&)            = delete;
    CudaEngine(CudaEngine&&)                 = delete;
    CudaEngine& operator=(const CudaEngine&) = delete;
    CudaEngine& operator=(CudaEngine&&)      = delete;


    bool                Initialize() override;
    static CudaEngine* GetInstance(); // hides Engine::GetInstance with the concrete type


 // Engine interface
    std::unique_ptr<Buffer> CreateBuffer(size_t bytes) override;
    Kernel*                 GetKernel(const std::string& name) override;

    bool   Begin() override;
    void   SetBuffer(Buffer* buffer, uint32_t index) override;
    void   SetBytes(const void* data, size_t bytes, uint32_t index) override;

    void   Dispatch3D(Kernel* kernel, size_t x, size_t y, size_t z) override;

    double Submit() override;

    bool CanFit(size_t bytes_per_buffer, size_t total_bytes) const override;

private:
    int  m_device{0};          // CUDA device index
    bool m_initialized{false};

static CudaEngine* m_instance;
};

#endif /* CudaEngine_hpp */