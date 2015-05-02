#ifndef CUDAKERNEL
#define CUDAKERNEL

#include <cuda_runtime.h>

const static float DIFF_RATE = 1.0;  // Non unity will be non convergent
const static int NX = 300;
const static int NY = 300;

__global__ void testKernel();

__global__ void testKernel2(int &val);

__global__ void testKernel3(float *val);

__global__ void testKernel4(float *data1, float *data2);
__global__ void testKernel4r(float *data1, float *data2);

__global__ void testKernel5();

__global__ void testKernelInject(float *data);



#endif // CUDAKERNEL

