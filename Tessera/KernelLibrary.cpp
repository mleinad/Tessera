#include "KernelLibrary.hpp"
#include "Engine.hpp"

KernelLibrary::KernelLibrary() {
    m_library  = Engine::GetInstace()->GetDevice()->newDefaultLibrary();
    m_function = m_library->newFunction(NS::String::string("add_one", NS::UTF8StringEncoding));

    NS::Error* err   = nullptr;
    m_pipeline_state = Engine::GetInstace()->GetDevice()->newComputePipelineState(m_function, &err);
    
    if (!m_pipeline_state) {
        printf("%s\n", err->localizedDescription()->utf8String());
        return;
    }
}

KernelLibrary::~KernelLibrary() {
}

MTL::ComputePipelineState* KernelLibrary::GetPSO() {
    return m_pipeline_state;
}
