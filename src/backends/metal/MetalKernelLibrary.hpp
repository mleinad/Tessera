#ifndef MetalKernelLibrary_hpp
#define MetalKernelLibrary_hpp

#include <Metal/Metal.hpp>
#include <stdio.h>
#include <memory>
#include <string>
#include <unordered_map>
#include "common/Engine.hpp"

// Kernel handle for Metal: wraps the compute pipeline state
class MetalKernel : public Kernel {
public:
    explicit MetalKernel(MTL::ComputePipelineState* pso) : m_pso(pso) {}
    MTL::ComputePipelineState* Get() const { return m_pso; }
private:
    MTL::ComputePipelineState* m_pso;
};

class MetalKernelLibrary {
public:
    explicit MetalKernelLibrary(MTL::Device* device);
    ~MetalKernelLibrary();

    MetalKernel* Get(const std::string& name);
private:
    MTL::Device*                                                  m_device;
    MTL::Library*                                                 m_library{nullptr};
    std::unordered_map<std::string, std::unique_ptr<MetalKernel>> m_cache;
};

#endif /* MetalKernelLibrary_hpp */
