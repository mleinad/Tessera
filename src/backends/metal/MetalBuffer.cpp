#include "MetalBuffer.hpp"
#include "MetalEngine.hpp"

MetalBuffer::MetalBuffer(size_t bytes, MTL::ResourceOptions opts)  : m_size(bytes) {
    MTL::Device* dev = MetalEngine::GetInstance()->GetDevice();
    assert(bytes <= dev->maxBufferLength() && "buffer exceeds maxBufferLength");
    m_buffer = dev->newBuffer(bytes, opts);
    assert(m_buffer && "newBuffer failed");
}

MetalBuffer::~MetalBuffer() {
}

MTL::Buffer* MetalBuffer::Get() const {
    return m_buffer;
}
size_t MetalBuffer::Size() const {
    return m_size;
}
void* MetalBuffer::Contents() {
    return m_buffer->contents();
}
