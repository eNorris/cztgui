
#include "SimEngine.h"


SimEngine::~SimEngine()
{
    if(built)
    {
        for(int i = 0; i < nx; i++)
            delete [] data[i];
        delete data;
    }
}

void SimEngine::build()
{
    if(built)
        return;
    if(nx <= 0 || ny <= 0)
        return;
    data = new double*[nx];
    prevdata = new double*[nx];
    for(int i = 0; i < nx; i++)
    {
        data[i] = new double[ny];
        prevdata[i] = new double[ny];
        for(int j = 0;j < ny; j++)
        {
            data[i][j] = double(rand())/RAND_MAX;
            prevdata[i][j] = data[i][j];
        }
    }

    built = true;
}

void SimEngine::build(const int nx, const int ny)
{
    this->nx = nx;
    this->ny = ny;
    build();
}

void SimEngine::run()
{
    if(!built)
        return;

    running = true;
    double diffrate = .25;

    while(running)
    {
        t += 1.0;
        QThread::msleep(30);

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
                //if(!(i % 10 == 0))
                data[i][j] = prevdata[i][j] + d;
                if(j == 0)
                    data[i][j] = 1.0;
            }

        emit update(t, data);

        double **tmp = data;
        data = prevdata;
        prevdata = tmp;
    }
    return;
}

void SimEngine::adddiffuse(int x, int y)
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

void SimEngine::subdiffuse(int x, int y)
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

void SimEngine::setPressure(double p)
{
    pressure = p;
}

void SimEngine::stop()
{
    running = false;
}

double& SimEngine::operator()(const int xIndex, const int yIndex)
{
    return data[xIndex][yIndex];
}
