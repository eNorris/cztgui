#ifndef CUDALINK
#define CUDALINK

#include <iostream>
#include <ctime>

#include <cuda_runtime.h>
#include "cuda_common/inc/helper_functions.h"
#include "cuda_common/inc/helper_cuda.h"

#include "cudakernel.h"

using std::cout;
using std::endl;

float **init_gpu(int Nx, int Ny, float *cpu_data);
void updateCpuData(float *data_cpu, float *data_gpu1, int nx, int ny);

int launch_testKernel();

int launch_diffusionKernel(int nx, int ny, float *gpu1, float *gpu2);
int launch_addDiffuseKernel(float *data_gpu1, int x, int y, float pressure);
int launch_subDiffuseKernel(float *data_gpu1, int x, int y, float pressure);

#endif // CUDALINK

