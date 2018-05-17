#ifndef UTIL_HPP
#define UTIL_HPP

#ifdef __CUDACC__
  #define CUDA_HOST_DEVICE __host__ __device__
  #define CUDA_HOST __host__
#else
  #define CUDA_HOST_DEVICE
  #define CUDA_HOST
#endif

#ifndef NDEBUG
  #define CUDA_CHECK(call) do { \
          call;\
          CheckCudaError(#call, __FILE__, __LINE__); \
      } while (0)

#else
  #define CUDA_CHECK(call) call
#endif

CUDA_HOST void CheckCudaError(const char* stmt, const char* fname, int line);

#endif
