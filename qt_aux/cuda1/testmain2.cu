
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
    int simcycles = 10;

    int devID;
    cudaDeviceProp props;
    //devID = findCudaDevice(0, 0);

    int devCount;
    cudaGetDeviceCount(&devCount);
    cout << "Cuda devices: " << devCount << endl;

    checkCudaErrors(cudaGetDevice(&devID));
    checkCudaErrors(cudaGetDeviceProperties(&props, devID));

    cout << "Device " << devID << ": " << props.name << " with compute " 
         << props.major << "." << props.minor << " capability" << endl;

    //cudaSetDevice(1);
    //checkCudaErrors(cudaGetDevice(&devID));
    //checkCudaErrors(cudaGetDeviceProperties(&props, devID));
    //cout << "Device " << devID << ": " << props.name << " with compute "
    //     << props.major << "." << props.minor << " capability" << endl;

    start = std::clock();
    float *cpu_data;
    int nx = 1024, ny = 1024;

    //cpu_data = new float*[nx];
    cpu_data = (float*) malloc(nx*ny*sizeof(float));
    for(int i = 0; i < nx*ny; i++)
    {
        //cpu_data[i] = new float[ny];
        cpu_data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX); 
        //(float*)malloc(ny*sizeof(float));
        //for(int j = 0; j < ny; j++)
        //    cpu_data[i][j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
    cout << "Init Time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC/1000) 
         << " ms" << endl;

    start = std::clock();
    float *gpu_data1;
    float *gpu_data2;
    cudaMalloc(&gpu_data1, nx*ny*sizeof(float));
    cudaMemcpy(gpu_data1, cpu_data, nx*ny*sizeof(float), cudaMemcpyHostToDevice);\
    cudaMalloc(&gpu_data2, nx*ny*sizeof(float));
    cudaDeviceSynchronize();
    //cudaMemcpy(gpu_data1, cpu_data, nx*ny*sizeof(float), cudaMemcpyHostToDevice);
    cout << "Copy Time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC/1000) 
         << " ms" << endl;

    dim3 dimGrid(1024);
    dim3 dimBlock(1024);

    //float q = 1000000.0;

    
    start = std::clock();
    for(int i = 0; i < simcycles; i++)
    {
        testKernel4<<<dimGrid, dimBlock>>>(gpu_data1, gpu_data2);
        cudaDeviceSynchronize();
        testKernel4r<<<dimGrid, dimBlock>>>(gpu_data1, gpu_data2);
        cudaDeviceSynchronize();
    }
    cout << "Kernel Time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC/1000) 
         << " ms" << endl;

    cudaError err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        //cudaError err = cudaGetLastError();
        printf("Error: %s\n\n", cudaGetErrorString(err));
    }
    //checkCudaErrors(cudaGetDevice(&devID));

    cudaDeviceSynchronize();

    return EXIT_SUCCESS;
}
