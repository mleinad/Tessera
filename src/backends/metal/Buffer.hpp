#ifndef Buffer_hpp
#define Buffer_hpp

#include <stdio.h>
#include "Metal/Metal.hpp"

class Buffer {
public:
    Buffer(size_t bytes, MTL::ResourceOptions opts = MTL::ResourceStorageModeShared);
    ~Buffer();

    Buffer(const Buffer&)            = delete;
    Buffer& operator=(const Buffer&) = delete;

    MTL::Buffer* Get() const;
    size_t       Size() const;
    
    template <class T> T* As() {
        return static_cast<T*>(m_buffer->contents());
    }
private:
    MTL::Buffer* m_buffer{nullptr};
    size_t       m_size{0};
};

#endif /* Buffer_hpp */
