#ifndef CUDA_HELPER_HPP
#define CUDA_HELPER_HPP

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, \
                         __LINE__, cudaGetErrorString(err_));                     \
            std::exit(1);                                                         \
        }                                                                         \
    } while (0)


#endif // CUDA_HELPER_HPP