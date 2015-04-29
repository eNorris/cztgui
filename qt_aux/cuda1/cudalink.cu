
#include "cudalink.h"

double **init_gpu(int nx, int ny, double **cpu_data)
{
    if(nx <= 0 || ny <= 0)
        return NULL;

    // Find a gpu
    int devID;
    cudaDeviceProp props;
    devId = findCudaDevice(0, 0);

    checkCudaErrors(cudaGetDevice(&devID));
    checkCudaErrors(cudaGetDeviceProperties(&props, devID));

    cout << "Device " << devID << ": " << props.name << " with compute "
         << props.major << "." << props.minor << " capability" << endl;




    data_cpu = new double*[nx];
    //data_cpu = new double*[nx];
    for(int i = 0; i < nx; i++)
    {
        data[i] = new double[ny];
        //prevdata[i] = new double[ny];
        for(int j = 0;j < ny; j++)
        {
            data[i][j] = double(rand())/RAND_MAX;
            //prevdata[i][j] = data[i][j];
        }
    }
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
