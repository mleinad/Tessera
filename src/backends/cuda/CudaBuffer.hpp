#ifndef CudaBuffer_hpp
#define CudaBuffer_hpp

#include "common/Engine.hpp"

#include <memory>

// Two memory modes:
//  Managed: one pointer valid on host and device, so Buffer::As<T>() works unchanged.
//           Pages migrate on demand, which lets a grid grow past VRAM (oversubscription).
//  Device:  plain VRAM (cudaMalloc), the naive discrete-GPU baseline. Limited to free VRAM
//           (~8 GB on the 3070 Ti Laptop). The host works on a mirror copy that is synced lazily:
//           uploaded when the buffer is bound to a kernel (DevicePtr), downloaded when the
//           host asks for it (Contents). Both copies are only touched outside the hot loop.
class CudaBuffer : public Buffer
{
public:
    enum class Mode
    {
        Managed,
        Device
    };

    explicit CudaBuffer(size_t bytes, Mode mode = Mode::Managed);
    ~CudaBuffer() override;

    CudaBuffer(const CudaBuffer&)            = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    size_t Size() const override;
    void*  Contents() override; // host pointer (Device mode: downloads if the GPU copy is newer)
    void*  DevicePtr();         // pointer for kernels (Device mode: uploads if the host copy is newer)
    Mode   GetMode() const { return m_mode; }

private:
    void*  m_ptr{nullptr}; // managed pointer, or VRAM pointer in Device mode
    size_t m_size{0};
    Mode   m_mode{Mode::Managed};

    // Device mode only
    std::unique_ptr<unsigned char[]> m_host;           // host mirror, allocated on first Contents()
    bool                             m_host_valid{false};   // mirror holds the newest data
    bool                             m_device_valid{false}; // VRAM holds the newest data
};

#endif /* CudaBuffer_hpp */
