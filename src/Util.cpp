#include "Util.hpp"

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>

#include <iostream>

CUDA_HOST void CheckCudaError(const char* call, const char* fname, int line)
{
    cudaError_t result_ = cudaGetLastError();
    if (result_ != cudaSuccess) {
        std::cerr << "At: " << fname << ":" << line << std::endl \
           << " Cuda call: " << call << " Error: " << cudaGetErrorString(result_) << std::endl;
        throw std::runtime_error("Error in CUDA call");
    }
}

#endif
