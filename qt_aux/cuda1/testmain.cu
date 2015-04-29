
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

    int devID;
    cudaDeviceProp props;
    //devID = findCudaDevice(0, 0);

    checkCudaErrors(cudaGetDevice(&devID));
    checkCudaErrors(cudaGetDeviceProperties(&props, devID));

    cout << "Device " << devID << ": " << props.name << " with compute " 
         << props.major << "." << props.minor << " capability" << endl;

    //devID = 1;
    cudaSetDevice(1);
    checkCudaErrors(cudaGetDevice(&devID));
    checkCudaErrors(cudaGetDeviceProperties(&props, devID));
    cout << "Device " << devID << ": " << props.name << " with compute "
         << props.major << "." << props.minor << " capability" << endl;

    dim3 dimGrid(32, 32);
    dim3 dimBlock(16, 16, 4);

    //float q = 1000000.0;

    float *gpu_q;
    float *cpu_q = new float(100000.0);
    cudaMalloc(&gpu_q, sizeof(float));
    cudaMemcpy(gpu_q, cpu_q, sizeof(float), cudaMemcpyHostToDevice); 

    std::clock_t start;
    start = std::clock();
    testKernel3<<<dimGrid, dimBlock>>>(gpu_q);
    cudaDeviceSynchronize();
    cout << "Time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC/1000) 
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
