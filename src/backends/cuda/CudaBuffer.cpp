#include "CudaBuffer.hpp"
#include "CudaHelper.hpp"
#include "common/logger.hpp"

#include <cstdlib>

CudaBuffer::CudaBuffer(size_t bytes, Mode mode) : m_size(bytes), m_mode(mode)
{
    if (m_mode == Mode::Managed)
    {
        CUDA_CHECK(cudaMallocManaged(&m_ptr, bytes));
        return;
    }

    // Device mode cannot exceed VRAM: fail with a clear message instead of a bare "out of memory"
    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    if (bytes > free_bytes)
    {
        constexpr double GB = 1024.0 * 1024.0 * 1024.0;
        log("VRAM-only buffer of {:.2f} GB does not fit: {:.2f} GB free of {:.2f} GB. "
            "Use managed memory for grids larger than VRAM.",
            bytes / GB, free_bytes / GB, total_bytes / GB);
        std::exit(1);
    }
    CUDA_CHECK(cudaMalloc(&m_ptr, bytes));
}

CudaBuffer::~CudaBuffer()
{
    cudaFree(m_ptr);
}

size_t CudaBuffer::Size() const
{
    return m_size;
}

void* CudaBuffer::Contents()
{
    if (m_mode == Mode::Managed)
    {
        return m_ptr;
    }

    if (!m_host)
    {
        m_host.reset(new unsigned char[m_size]); // uninitialised, like cudaMalloc
    }
    if (m_device_valid && !m_host_valid)
    {
        CUDA_CHECK(cudaMemcpy(m_host.get(), m_ptr, m_size, cudaMemcpyDeviceToHost));
    }
    // The caller may write through this pointer, so the mirror becomes the newest copy
    m_host_valid   = true;
    m_device_valid = false;
    return m_host.get();
}

void* CudaBuffer::DevicePtr()
{
    if (m_mode == Mode::Managed)
    {
        return m_ptr;
    }

    if (!m_device_valid && m_host_valid)
    {
        CUDA_CHECK(cudaMemcpy(m_ptr, m_host.get(), m_size, cudaMemcpyHostToDevice));
    }
    // A kernel may write to it, so VRAM becomes the newest copy
    m_device_valid = true;
    m_host_valid   = false;
    return m_ptr;
}
