
#include <assert.h>

#include <cuda_runtime.h>

#include "cuda_common/inc/helper_functions.h"
#include "cuda_common/inc/helper_cuda.h"
//#include <helper_functions.h>
//#include <helper_cuda.h>

#include <iostream>
#include <ctime>

#include "./cudakernel.h"

using namespace std;

int main()
{

    // Rand seed for consistency
    srand(0);
    std::clock_t start;
    int simcycles = 10000;

    int devID;
    cudaDeviceProp props;

    int devCount;
    cudaGetDeviceCount(&devCount);
    cout << "Cuda devices: " << devCount << endl;

    checkCudaErrors(cudaGetDevice(&devID));
    checkCudaErrors(cudaGetDeviceProperties(&props, devID));

    cout << "Device " << devID << ": " << props.name << " with compute " 
         << props.major << "." << props.minor << " capability" << endl;

    start = std::clock();
    float *cpu_data;
    int nx = 1024, ny = 1024;

    // Work with streams
    int *hp = new int, *lp = new int;
    cudaDeviceGetStreamPriorityRange(lp, hp);
    cout << "HP = " << *hp << "   LP = " << *lp << endl;

    int *v = new int;
    cudaDeviceGetAttribute(v, cudaDevAttrMaxThreadsPerBlock, 0);
    cout << "cudaDevAttrMaxThreadsPerBlock = " << *v << endl;

    cudaDeviceGetAttribute(v, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    cout << "cudaDevAttrMaxSharedMemoryPerBlock = " << *v << endl;

    cudaDeviceGetAttribute(v, cudaDevAttrTotalConstantMemory, 0);
    cout << "cudaDevAttrTotalConstantMemory = " << *v << endl;

    cudaDeviceGetAttribute(v, cudaDevAttrWarpSize, 0);
    cout << "cudaDevAttrWarpSize = " << *v << endl;

    cudaDeviceGetAttribute(v, cudaDevAttrMaxPitch, 0);
    cout << "cudaDevAttrMaxPitch = " << *v << endl;

    cudaDeviceGetAttribute(v, cudaDevAttrMaxRegistersPerBlock, 0);
    cout << "cudaDevAttrMaxRegistersPerBlock = " << *v << endl;

    cudaDeviceGetAttribute(v, cudaDevAttrClockRate, 0);
    cout << "cudaDevAttrClockRate = " << *v << endl;

    cudaDeviceGetAttribute(v, cudaDevAttrGpuOverlap, 0);
    cout << "cudaDevAttrGpuOverlap = " << *v << endl;

    cudaDeviceGetAttribute(v, cudaDevAttrMultiProcessorCount, 0);
    cout << "cudaDevAttrMultiProcessorCount = " << *v << endl;

    cudaDeviceGetAttribute(v, cudaDevAttrKernelExecTimeout, 0);
    cout << "cudaDevAttrKernelExecTimeout = " << *v << endl;

    cudaDeviceGetAttribute(v, cudaDevAttrIntegrated, 0);
    cout << "cudaDevAttrIntegrated = " << *v << endl;

    cudaDeviceGetAttribute(v, cudaDevAttrMaxRegistersPerBlock, 0);
    cout << "cudaDevAttrMaxRegistersPerBlock = " << *v << endl;

    cudaDeviceGetAttribute(v, cudaDevAttrCanMapHostMemory, 0);
    cout << "cudaDevAttrCanMapHostMemory = " << *v << endl;

    cudaDeviceGetAttribute(v, cudaDevAttrComputeMode, 0);
    cout << "cudaDevAttrComputeMode = " << *v << endl;

    cudaDeviceGetAttribute(v, cudaDevAttrConcurrentKernels, 0);
    cout << "cudaDevAttrConcurrentKernels = " << *v << endl;

    cudaDeviceGetAttribute(v, cudaDevAttrMemoryClockRate, 0);
    cout << "cudaDevAttrMemoryClockRate = " << *v << endl;

    cudaDeviceGetAttribute(v, cudaDevAttrGlobalMemoryBusWidth, 0);
    cout << "cudaDevAttrGlobalMemoryBusWidth = " << *v << endl;

    cudaDeviceGetAttribute(v, cudaDevAttrL2CacheSize, 0);
    cout << "cudaDevAttrL2CacheSize = " << *v << endl;

    cudaDeviceGetAttribute(v, cudaDevAttrMaxThreadsPerMultiProcessor, 0);
    cout << "cudaDevAttrMaxThreadsPerMultiProcessor = " << *v << endl;

    cudaDeviceGetAttribute(v, cudaDevAttrStreamPrioritiesSupported, 0);
    cout << "cudaDevAttrStreamPrioritiesSupported = " << *v << endl;

    cudaStream_t stream;
    //cudaStreamCreate(&stream);
    cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, *lp);
    cudaStream_t hpStream;
    cudaStreamCreateWithPriority(&hpStream, cudaStreamNonBlocking, *hp);

    //cpu_data = new float*[nx];
    cpu_data = (float*) malloc(nx*ny*sizeof(float));
    for(int i = 0; i < nx*ny; i++)
    {
        cpu_data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX); 
    }
    cout << "Init Time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC/1000) 
         << " ms" << endl;

    // Copy to the GPU
    start = std::clock();
    float *gpu_data1;
    float *gpu_data2;
    cudaMalloc(&gpu_data1, nx*ny*sizeof(float));
    cudaMemcpyAsync(gpu_data1, cpu_data, nx*ny*sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMalloc(&gpu_data2, nx*ny*sizeof(float));
    //cudaDeviceSynchronize();
    cout << "Copy Time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC/1000) 
         << " ms" << endl;

    dim3 dimGrid(1024);
    dim3 dimBlock(1024);

    // Start launching the kernel
    start = std::clock();
    for(int i = 0; i < simcycles; i++)
    {
        testKernel4<<<dimGrid, dimBlock, 0, stream>>>(gpu_data1, gpu_data2);
        //cudaDeviceSynchronize();
        testKernel4r<<<dimGrid, dimBlock, 0, stream>>>(gpu_data1, gpu_data2);
        //cudaDeviceSynchronize();
    }
    
    testKernelInject<<<1,1,0,hpStream>>>(gpu_data1);
    cudaStreamSynchronize(hpStream);
    cout << "HP Time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC/1000) 
         << " ms" << endl;

    cudaDeviceSynchronize();
    cout << "Kernel Time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC/1000) 
         << " ms" << endl;

    cudaError err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        printf("Error: %s\n\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

    return EXIT_SUCCESS;
}
