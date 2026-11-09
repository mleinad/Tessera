#ifndef Kernel_hpp
#define Kernel_hpp

#include "Metal/Metal.hpp"
#include <stdio.h>

class KernelLibrary {
public:
    explicit KernelLibrary(MTL::Device* device);
    ~KernelLibrary();

    MTL::ComputePipelineState* Get(const std::string& name);
private:
    MTL::Device*                                                m_device;
    MTL::Library*                                               m_library{nullptr};
    std::unordered_map<std::string, MTL::ComputePipelineState*> m_cache;
};

#endif /* Kernel_hpp */
