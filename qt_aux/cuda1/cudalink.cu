
#include "cudalink.h"

float **init_gpu(int nx, int ny, double *cpu_data)
{
    if(nx <= 0 || ny <= 0)
        return NULL;

    std::cout << "Initializing GPU resources" << std::endl;

    // Find a gpu
    int devID = 0;
    cudaDeviceProp props;
    //devID = findCudaDevice(0, 0);

    checkCudaErrors(cudaGetDevice(&devID));
    checkCudaErrors(cudaGetDeviceProperties(&props, devID));

    std::cout << "Device " << devID << ": " << props.name << " with compute "
         << props.major << "." << props.minor << " capability" << std::endl;

    std::clock_t start = std::clock();
    float *gpu_data1;
    float *gpu_data2;
    cudaMalloc(&gpu_data1, nx*ny*sizeof(float));
    cudaMemcpyAsync(gpu_data1, cpu_data, nx*ny*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&gpu_data2, nx*ny*sizeof(float));
    std::cout << "Copy Time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC/1000)
         << " ms" << std::endl;

    float **gpu_datas = new float*[2];
    gpu_datas[0] = gpu_data1;
    gpu_datas[1] = gpu_data2;

    //double *gpu_data;
    //gpu_data = new double*[nx];
    //data_cpu = new double*[nx];
    //for(int i = 0; i < nx; i++)
    //{
    //    gpu_data[i] = new double[ny];
        //prevdata[i] = new double[ny];
    //    for(int j = 0;j < ny; j++)
    //    {
     //       gpu_data[i][j] = double(rand())/RAND_MAX;
            //prevdata[i][j] = data[i][j];
     //   }
    //}

    return gpu_datas;
}

int launch_testKernel(int &val)
{
    // Define the grid and blocks
    dim3 dimGrid(2, 2);
    dim3 dimBlock(2, 2, 2);

    testKernel<<<dimGrid, dimBlock>>>(val);

    cudaDeviceSynchronize();

    return EXIT_SUCCESS;
}
