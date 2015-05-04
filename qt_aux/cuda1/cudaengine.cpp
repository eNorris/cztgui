#include "cudaengine.h"

#include "cudalink.h"

CudaEngine::CudaEngine() : QObject(), built(false), running(false), simRateBounded(false), nx(0), ny(0), t(0), maxSimRate(1.0), pressure(1.0),
    data_cpu(NULL), data_gpu1(NULL), data_gpu2(NULL)
{
    simTimer = new QTimer(this);
    connect(simTimer, SIGNAL(timeout()), this, SLOT(unboundSimRate()));
}

CudaEngine::CudaEngine(const int nx, const int ny) : CudaEngine()
{
    build(nx, ny);
}

CudaEngine::~CudaEngine()
{
    if(built)
    {
        for(int i = 0; i < nx; i++)
        {
            delete [] data_cpu;
        }
        //delete data_cpu;
        data_cpu = NULL;

        // TODO - dealloc cuda memory

    }
}

void CudaEngine::build()
{
    if(built)
        return;
    if(nx <= 0 || ny <= 0)
        return;

    // Allocate the cpu array
    data_cpu = new float[nx*ny];
    for(int i = 0; i < nx*ny; i++)
        data_cpu[i] = float(rand())/RAND_MAX;

    // Allocate GPU resources
    float **gpu_datas = init_gpu(nx, ny, data_cpu);
    data_gpu1 = gpu_datas[0];
    data_gpu2 = gpu_datas[1];

    built = true;
}

void CudaEngine::build(const int nx, const int ny)
{
    this->nx = nx;
    this->ny = ny;
    build();
}

void CudaEngine::setSimRate(const double simRate)
{
    maxSimRate = simRate;
}

void CudaEngine::run()
{
    if(!built)
        return;

    running = true;
    simTimer->start(1000.0 / maxSimRate);

    while(running)
    {
        launch_diffusionKernel(nx, ny, data_gpu1, data_gpu2);

        QApplication::processEvents(QEventLoop::AllEvents, 100);

        t += 1.0;

        if(!simRateBounded)
        {
            // Copy from GPU
            updateCpuData(data_cpu, data_gpu1, nx, ny);

            // Emit data
            emit update(t, data_cpu);

            simRateBounded = true;
        }
    }
    simTimer->stop();
    return;
}

void CudaEngine::adddiffuse(int x, int y)
{
    launch_addDiffuseKernel(data_gpu1, x, y, pressure);
}

void CudaEngine::subdiffuse(int x, int y)
{
    launch_subDiffuseKernel(data_gpu1, x, y, pressure);
}

void CudaEngine::setPressure(double p)
{
    pressure = p;
}

void CudaEngine::stop()
{
    running = false;
}

void CudaEngine::unboundSimRate()
{
    simRateBounded = false;
}
