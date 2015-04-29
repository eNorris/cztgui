
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
    cudaStream_t stream;
    cudaStreamCreate(&stream);

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
        cudaDeviceSynchronize();
        testKernel4r<<<dimGrid, dimBlock, 0, stream>>>(gpu_data1, gpu_data2);
        cudaDeviceSynchronize();
    }
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
