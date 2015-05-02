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
    /*
    for(int i = 0; i < nx; i++)
    {
        data_cpu[i] = new double[ny];
        for(int j = 0;j < ny; j++)
        {
            data_cpu[i][j] = double(rand())/RAND_MAX;
        }
    }
    */

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
    //simTimer->setInterval(1.0/maxSimRate);
}

void CudaEngine::run()
{
    if(!built)
        return;

    running = true;
    simTimer->start(1000.0 / maxSimRate);
    //double diffrate = .25;

    while(running)
    {
        //while(simRateBounded)
        //{
        launch_diffusionKernel(nx, ny, data_gpu1, data_gpu2);
        //QThread::msleep(500);
        //updateCpuData(data_cpu, data_gpu1, nx, ny);

        //qDebug() << "t = " << t;

        QApplication::processEvents(QEventLoop::AllEvents, 100);
        //}
        //simRateBounded = true;

        t += 1.0;

        /*
        double d = 0;
        for(int i = 0; i < nx; i++)
            for(int j = 0; j < ny; j++)
            {
                d = 0;
                if(i != 0)
                    d += diffrate * (prevdata[i-1][j] - prevdata[i][j]);
                if(i != nx-1)
                    d += diffrate * (prevdata[i+1][j] - prevdata[i][j]);
                if(j != 0)
                    d += diffrate * (prevdata[i][j-1] - prevdata[i][j]);
                if(j != ny-1)
                    d += diffrate * (prevdata[i][j+1] - prevdata[i][j]);
                data[i][j] = prevdata[i][j] + d;
                if(j == 0)
                    data[i][j] = 1.0;
            }
        */

        if(!simRateBounded)
        {
            // Copy from GPU
            updateCpuData(data_cpu, data_gpu1, nx, ny);

            //for(int i = 0; i < nx*ny; i++)
            //    qDebug() << data_cpu[i];

            // Emit data
            //qDebug() << "emit!";
            emit update(t, data_cpu);

            //double **tmp = data;
            //data = prevdata;
            //prevdata = tmp;

            // TODO relaunch the kernel
            simRateBounded = true;
        }

        //QApplication::processEvents(QEventLoop::AllEvents, 100);
    }
    simTimer->stop();
    return;
}

void CudaEngine::adddiffuse(int x, int y)
{
    launch_addDiffuseKernel(data_gpu1, x, y, pressure);
    /*
    if(x >= 0 && x < nx && y >= 0 && y < ny)
        data_cpu[nx*x+y] += pressure;
    if(x > 0 && x < nx && y >= 0 && y < ny)
        data_cpu[nx*(x-1) + y] += pressure/2.0;
    if(x < nx-1 && x >= 0 && y >= 0 && y < ny)
        data_cpu[nx*(x+1) + y] += pressure/2.0;
    if(y > 0 && y < ny && x >= 0 && x < nx)
        data_cpu[nx * x + y-1] += pressure/2.0;
    if(y <= ny-1 && y >= 0 && x >= 0 && x < nx)
        data_cpu[nx*x + y+1] += pressure/2.0;
    */
}

void CudaEngine::subdiffuse(int x, int y)
{
    launch_subDiffuseKernel(data_gpu1, x, y, pressure);
    /*
    if(x >= 0 && x < nx && y >= 0 && y < ny)
        data_cpu[nx*x + y] -= pressure;
    if(x > 0 && x < nx && y >= 0 && y < ny)
        data_cpu[nx*(x-1) + y] -= pressure/2.0;
    if(x < nx-1 && x >= 0 && y >= 0 && y < ny)
        data_cpu[nx*(x+1) + y] -= pressure/2.0;
    if(y > 0 && y < ny && x >= 0 && x < nx)
        data_cpu[nx*x + y-1] -= pressure/2.0;
    if(y <= ny-1 && y >= 0 && x >= 0 && x < nx)
        data_cpu[nx * x + y+1] -= pressure/2.0;
    */
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
