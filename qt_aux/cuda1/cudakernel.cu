
#include <stdio.h>
#include <cmath>

#include "cudakernel.h"

__global__ void testKernel(int &val)
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
    
    if(blockIdx.x > 0)
    {
        t += (data1[1024*(blockIdx.x-1)+threadIdx.x] - data1[1024*blockIdx.x+threadIdx.x]);
        c += 1.0f;
    }
    if(blockIdx.x < 1023)
    {
        t += (data1[1024*(blockIdx.x+1)+threadIdx.x] - data1[1024*blockIdx.x+threadIdx.x]);
        c+=1.0f;
    }
    if(threadIdx.x > 0)
    {
        t += (data1[1024*blockIdx.x+threadIdx.x-1] - data1[1024*blockIdx.x+threadIdx.x]);
        c+=1.0f;
    }
    if(threadIdx.x < 1023)
    {
        t += (data1[1024*blockIdx.x+threadIdx.x+1] - data1[1024*blockIdx.x+threadIdx.x]);
        c+=1.0f;
    }
    //printf("block %i, %i, %i\n", blockIdx.x, threadIdx.x, 1024*blockIdx.x+threadIdx.x);
    //data2[1024*blockIdx.x+threadIdx.x] = 2*data1[1024*blockIdx.x+threadIdx.x];
    data2[1024*blockIdx.x+threadIdx.x] = t/c*DIFF_RATE;
    return;
}

__global__ void testKernel4r(float *data1, float *data2)
{
    float t = 0.0f;
    float c = 0.0f;
    
    if(blockIdx.x > 0)
    {
        t += (data2[1024*(blockIdx.x-1)+threadIdx.x] - data2[1024*blockIdx.x+threadIdx.x]);
        c += 1.0f;
    }
    if(blockIdx.x < 1023)
    {
        t += (data2[1024*(blockIdx.x+1)+threadIdx.x] - data2[1024*blockIdx.x+threadIdx.x]);
        c+=1.0f;
    }
    if(threadIdx.x > 0)
    {
        t += (data2[1024*blockIdx.x+threadIdx.x-1] - data2[1024*blockIdx.x+threadIdx.x]);
        c+=1.0f;
    }
    if(threadIdx.x < 1023)
    {
        t += (data2[1024*blockIdx.x+threadIdx.x+1] - data2[1024*blockIdx.x+threadIdx.x]);
        c+=1.0f;
    }
    //printf("block %i, %i, %i\n", blockIdx.x, threadIdx.x, 1024*blockIdx.x+threadIdx.x);
    //data2[1024*blockIdx.x+threadIdx.x] = 2*data1[1024*blockIdx.x+threadIdx.x];
    data1[1024*blockIdx.x+threadIdx.x] = t/c*DIFF_RATE;
    return;
}










