#ifndef MetalBuffer_hpp
#define MetalBuffer_hpp

#include <stdio.h>
#include "Metal/Metal.hpp"
#include "common/Engine.hpp"

class MetalBuffer : public Buffer {
public:
    MetalBuffer(size_t bytes, MTL::ResourceOptions opts = MTL::ResourceStorageModeShared);
    ~MetalBuffer() override;

    MetalBuffer(const MetalBuffer&)            = delete;
    MetalBuffer& operator=(const MetalBuffer&) = delete;

    MTL::Buffer* Get() const;
    size_t       Size() const override;
    void*        Contents() override;

private:
    MTL::Buffer* m_buffer{nullptr};
    size_t       m_size{0};
};

#endif /* MetalBuffer_hpp */
