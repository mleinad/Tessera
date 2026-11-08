#include "Buffer.hpp"
#include "Engine.hpp"

Buffer::Buffer(size_t bytes, MTL::ResourceOptions opts)  : m_size(bytes) {
    MTL::Device* dev = Engine::GetInstance()->GetDevice();
    assert(bytes <= dev->maxBufferLength() && "buffer exceeds maxBufferLength");
    m_buffer = dev->newBuffer(bytes, opts);
    assert(m_buffer && "newBuffer failed");
}

Buffer::~Buffer() {
}

MTL::Buffer* Buffer::Get() const {
    return m_buffer;
}
size_t Buffer::Size() const {
    return m_size;
}

