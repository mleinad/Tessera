#ifndef CudaEngine_hpp
#define CudaEngine_hpp

#include "common/Engine.hpp"
#include "CudaBuffer.hpp"

#include <cuda_runtime.h>

class CudaKernelLibrary;

class CudaEngine : public Engine
{
public:
    CudaEngine();
    ~CudaEngine() override;

    CudaEngine(const CudaEngine &) = delete;
    CudaEngine(CudaEngine &&) = delete;
    CudaEngine &operator=(const CudaEngine &) = delete;
    CudaEngine &operator=(CudaEngine &&) = delete;

    bool Initialize() override;
    static CudaEngine *GetInstance(); // hides Engine::GetInstance with the concrete type

    // Engine interface
    std::unique_ptr<Buffer> CreateBuffer(size_t bytes) override;
    Kernel *GetKernel(const std::string &name) override;

    bool Begin() override;
    void SetBuffer(Buffer *buffer, uint32_t index) override;
    void SetBytes(const void *data, size_t bytes, uint32_t index) override;

    void Dispatch3D(Kernel *kernel, size_t x, size_t y, size_t z) override;
    void DispatchColumns(Kernel *kernel, size_t x, size_t y) override;

    double Submit() override;

    FitReport CheckFit(size_t bytes_per_buffer, size_t total_bytes) const override;

    // Memory mode for buffers created from now on. Default: Managed,
    // or Device when the environment variable TESSERA_CUDA_MEMORY=device is set.
    void SetBufferMode(CudaBuffer::Mode mode) { m_buffer_mode = mode; }
    CudaBuffer::Mode GetBufferMode() const { return m_buffer_mode; }

    EngineInfo Info() const { return m_info; }

private:
    int m_device{0}; // CUDA device index
    CudaBuffer::Mode m_buffer_mode{CudaBuffer::Mode::Managed};
    bool m_initialized{false};

    std::unique_ptr<CudaKernelLibrary> m_kernels;
    cudaStream_t m_stream{nullptr};
    cudaEvent_t m_start{nullptr}; // GPU timing for Submit()
    cudaEvent_t m_stop{nullptr};

    // Bind slots, emulating Metal's setBuffer/setBytes: slot index == kernel parameter position.
    // cudaLaunchKernel copies the values at launch, so rebinding between dispatches is safe.
    static constexpr uint32_t kMaxBinds = 8;
    static constexpr size_t kMaxBindBytes = 256;

    void *m_buffer_slots[kMaxBinds]{};                                  // device pointers (SetBuffer)
    alignas(16) unsigned char m_byte_slots[kMaxBinds][kMaxBindBytes]{}; // by-value structs (SetBytes)
    void *m_args[kMaxBinds]{};                                          // what cudaLaunchKernel reads

    static CudaEngine *m_instance;
};

#endif /* CudaEngine_hpp */
