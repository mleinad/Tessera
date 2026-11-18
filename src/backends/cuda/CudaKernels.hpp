#ifndef CudaKernels_hpp
#define CudaKernels_hpp

#include <string>

// CUDA has no runtime lookup by name like Metal's newFunction, so this table replaces it.
const void* FindCudaKernel(const std::string& name);

#endif /* CudaKernels_hpp */
