#ifndef Engine_hpp
#define Engine_hpp

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

// Backend-neutral interface. The simulator only talks to these types;
// each backend (backends/metal, backends/cuda) provides the implementation.

// Opaque handle to a compiled kernel (Metal: pipeline state, CUDA: kernel function).
class Kernel {
public:
    virtual ~Kernel() = default;
};

class Buffer {
public:
    virtual ~Buffer() = default;

    virtual size_t Size() const = 0;
    virtual void*  Contents()   = 0; // host-visible pointer (shared / managed memory)

    template <class T> T* As() {
        return static_cast<T*>(Contents());
    }
};

class Engine {
public:
    virtual ~Engine() = default;

    static Engine* GetInstance(); // defined by the active backend

    virtual bool Initialize() = 0;

    virtual std::unique_ptr<Buffer> CreateBuffer(size_t bytes) = 0;
    virtual Kernel*                 GetKernel(const std::string& name) = 0;

    // One batch of work: Begin, then any number of Set*/Dispatch3D, then Submit.
    virtual bool   Begin() = 0;
    virtual void   SetBuffer(Buffer* buffer, uint32_t index) = 0;
    virtual void   SetBytes(const void* data, size_t bytes, uint32_t index) = 0;
    virtual void   Dispatch3D(Kernel* kernel, size_t x, size_t y, size_t z) = 0;
    virtual double Submit() = 0; // waits for completion, returns GPU seconds

    virtual bool CanFit(size_t bytes_per_buffer, size_t total_bytes) const = 0;
};

#endif /* Engine_hpp */
