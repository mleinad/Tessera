#ifndef CudaKernelLibrary_hpp
#define CudaKernelLibrary_hpp

#include <memory>
#include <string>
#include <unordered_map>
#include "common/Engine.hpp"

// Kernel handle for CUDA: wraps the __global__ function pointer passed to cudaLaunchKernel
class CudaKernel : public Kernel
{
public:
    explicit CudaKernel(const void* fn) : m_fn(fn) {}
    const void* Get() const { return m_fn; }
private:
    const void* m_fn;
};

class CudaKernelLibrary
{
public:
    CudaKernel* Get(const std::string& name);
private:
    std::unordered_map<std::string, std::unique_ptr<CudaKernel>> m_cache;
};

#endif /* CudaKernelLibrary_hpp */
