#include "CudaKernelLibrary.hpp"
#include "CudaKernels.hpp"

#include <cassert>

CudaKernel* CudaKernelLibrary::Get(const std::string& name)
{
    if (auto it = m_cache.find(name); it != m_cache.end())
    {
        return it->second.get();
    }

    const void* fn = FindCudaKernel(name);
    assert(fn && "kernel not found in wave_kernels.cu");

    m_cache[name] = std::make_unique<CudaKernel>(fn);
    return m_cache[name].get();
}
