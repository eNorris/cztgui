#include "cudaengine.h"

CudaEngine::CudaEngine() : QObject(), built(false), running(false), nx(0), ny(0), t(0), pressure(1.0),
    data(NULL), prevdata(NULL), maxSimRate(1.0), simRateBounded(false)
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
            delete [] data[i];
        delete data;
    }
}

void CudaEngine::build()
{
    if(built)
        return;
    if(nx <= 0 || ny <= 0)
        return;

    // Build the CPU matrix
    data = new double*[nx];
    //prevdata = new double*[nx];
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

    // Copy the CPU data to the GPU

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
    double diffrate = .25;

    while(running)
    {
        // wait for the GPU to update
        while(simRateBounded)
        {
            QApplication::processEvents(QEventLoop::AllEvents, 100);
        }
        simRateBounded = true;

	// copy the GPU solution

        t += 1.0;

        //double d = 0;
        //for(int i = 0; i < nx; i++)
        //    for(int j = 0; j < ny; j++)
        //    {
        //        d = 0;
        //        if(i != 0)
        //            d += diffrate * (prevdata[i-1][j] - prevdata[i][j]);
        //        if(i != nx-1)
        //            d += diffrate * (prevdata[i+1][j] - prevdata[i][j]);
        //        if(j != 0)
        //            d += diffrate * (prevdata[i][j-1] - prevdata[i][j]);
        //        if(j != ny-1)
        //            d += diffrate * (prevdata[i][j+1] - prevdata[i][j]);
        //        data[i][j] = prevdata[i][j] + d;
        //        if(j == 0)
        //            data[i][j] = 1.0;
        //    }

        emit update(t, data);

        double **tmp = data;
        data = prevdata;
        prevdata = tmp;

        QApplication::processEvents(QEventLoop::AllEvents, 100);
    }
    simTimer->stop();
    return;
}

void CudaEngine::adddiffuse(int x, int y)
{
    if(x >= 0 && x < nx && y >= 0 && y < ny)
        prevdata[x][y] += pressure;
    if(x > 0 && x < nx && y >= 0 && y < ny)
        prevdata[x-1][y] += pressure/2.0;
    if(x < nx-1 && x >= 0 && y >= 0 && y < ny)
        prevdata[x+1][y] += pressure/2.0;
    if(y > 0 && y < ny && x >= 0 && x < nx)
        prevdata[x][y-1] += pressure/2.0;
    if(y <= ny-1 && y >= 0 && x >= 0 && x < nx)
        prevdata[x][y+1] += pressure/2.0;
}

void CudaEngine::subdiffuse(int x, int y)
{
    if(x >= 0 && x < nx && y >= 0 && y < ny)
        prevdata[x][y] -= pressure;
    if(x > 0 && x < nx && y >= 0 && y < ny)
        prevdata[x-1][y] -= pressure/2.0;
    if(x < nx-1 && x >= 0 && y >= 0 && y < ny)
        prevdata[x+1][y] -= pressure/2.0;
    if(y > 0 && y < ny && x >= 0 && x < nx)
        prevdata[x][y-1] -= pressure/2.0;
    if(y <= ny-1 && y >= 0 && x >= 0 && x < nx)
        prevdata[x][y+1] -= pressure/2.0;
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
