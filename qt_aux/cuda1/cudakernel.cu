
#include <stdio.h>
#include <cmath>

#include "cudakernel.h"

__global__ void testKernel()
{
    printf("hi!\n");
}

__global__ void testKernel2(int &val)
{
    printf("[%d, %d]:\t\tValue is: %d\n",\
            blockIdx.y*gridDim.x+blockIdx.x,\
            threadIdx.z*blockDim.x*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x,\
            val);
}

__global__ void testKernel3(float *val)
{
    float xx;
    for(float i = 0; i < *val; i+=1.0)
    {
        xx = cosf(i) + sinf(i);
    }
}

__global__ void testKernel4(float *data1, float *data2)
{
    float t = 0.0f;
    float c = 0.0f;

    printf("d = %f\n", data1[NX*blockIdx.x + threadIdx.x]);
    
    if(blockIdx.x > 0)
    {
        t += (data1[NX*(blockIdx.x-1)+threadIdx.x] - data1[NX*blockIdx.x+threadIdx.x]);
        c += 1.0f;
    }
    if(blockIdx.x < NX-1)
    {
        t += (data1[NX*(blockIdx.x+1)+threadIdx.x] - data1[NX*blockIdx.x+threadIdx.x]);
        c+=1.0f;
    }
    if(threadIdx.x > 0)
    {
        t += (data1[NX*blockIdx.x+threadIdx.x-1] - data1[NX*blockIdx.x+threadIdx.x]);
        c+=1.0f;
    }
    if(threadIdx.x < NX-1)
    {
        t += (data1[NX*blockIdx.x+threadIdx.x+1] - data1[NX*blockIdx.x+threadIdx.x]);
        c+=1.0f;
    }
    //printf("block %i, %i, %i\n", blockIdx.x, threadIdx.x, 1024*blockIdx.x+threadIdx.x);
    //data2[1024*blockIdx.x+threadIdx.x] = 2*data1[1024*blockIdx.x+threadIdx.x];
    data2[NX*blockIdx.x+threadIdx.x] = t/c*DIFF_RATE;
    return;
}

__global__ void testKernel4r(float *data1, float *data2)
{
    float t = 0.0f;
    float c = 0.0f;

    printf("r = %f\n", data2[NX*blockIdx.x + threadIdx.x]);
    
    if(blockIdx.x > 0)
    {
        t += (data2[NX*(blockIdx.x-1)+threadIdx.x] - data2[NX*blockIdx.x+threadIdx.x]);
        c += 1.0f;
    }
    if(blockIdx.x < NX-1)
    {
        t += (data2[NX*(blockIdx.x+1)+threadIdx.x] - data2[NX*blockIdx.x+threadIdx.x]);
        c+=1.0f;
    }
    if(threadIdx.x > 0)
    {
        t += (data2[NX*blockIdx.x+threadIdx.x-1] - data2[NX*blockIdx.x+threadIdx.x]);
        c+=1.0f;
    }
    if(threadIdx.x < NX-1)
    {
        t += (data2[NX*blockIdx.x+threadIdx.x+1] - data2[NX*blockIdx.x+threadIdx.x]);
        c+=1.0f;
    }
    //printf("block %i, %i, %i\n", blockIdx.x, threadIdx.x, 1024*blockIdx.x+threadIdx.x);
    //data2[1024*blockIdx.x+threadIdx.x] = 2*data1[1024*blockIdx.x+threadIdx.x];
    data1[NX*blockIdx.x+threadIdx.x] = t/c*DIFF_RATE;
    return;
}

__global__ void testKernel5()
{
    float xx;
    for(float i = 0; i < 100000.0; i+=1.0)
    {
        xx = cosf(i) + sinf(i);
    }
}

__global__ void testKernelInject(float *data)
{
    data[1024*512+512] += 1000.0;
    data[1024*512+513] += 1000.0;
    data[1024*513+512] += 1000.0;
    data[1024*513+513] += 1000.0;
}










