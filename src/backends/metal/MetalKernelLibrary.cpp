#include "MetalKernelLibrary.hpp"
#include "MetalEngine.hpp"
#include "common/logger.hpp"

static constexpr auto kKernalPath = "/Users/daniel.malheiro/Documents/fcul/PPC/Projects/Tessera/Tessera/Metal/lib/default.metallib";

MetalKernelLibrary::MetalKernelLibrary(MTL::Device* device) : m_device(device) {
    NS::Error* err = nullptr;
    auto*      url = NS::URL::fileURLWithPath(NS::String::string(kKernalPath, NS::UTF8StringEncoding));
    m_library      = m_device->newLibrary(url, &err);
    if (!m_library) {
        log("metallib load failed: {}", err->localizedDescription()->utf8String());
    }

    NS::Array* names = m_library->functionNames();
    for (NS::UInteger i = 0; i < names->count(); ++i) {
        log("metallib function: {}", names->object<NS::String>(i)->utf8String());
    }
}

MetalKernelLibrary::~MetalKernelLibrary() {
}

MetalKernel* MetalKernelLibrary::Get(const std::string& name) {
    if (auto it = m_cache.find(name); it != m_cache.end()) {
        return it->second.get();
    }

    MTL::Function* fn = m_library->newFunction(NS::String::string(name.c_str(), NS::UTF8StringEncoding));
    assert(fn && "kernel not found in library");
    NS::Error*                 err = nullptr;
    MTL::ComputePipelineState* pso = m_device->newComputePipelineState(fn, &err);
    fn->release(); // the PSO keeps what it needs
    if (!pso) {
        log("Failed to build pipeline, Err: {}", err->localizedDescription()->utf8String());
        assert(false);
    }

    m_cache[name] = std::make_unique<MetalKernel>(pso);
    return m_cache[name].get();
}
