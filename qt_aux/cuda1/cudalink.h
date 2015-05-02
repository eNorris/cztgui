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

float **init_gpu(int Nx, int Ny, double *cpu_data);

int launch_testKernel(int &val);

#endif // CUDALINK

