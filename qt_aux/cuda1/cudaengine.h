#ifndef CUDAENGINE_H
#define CUDAENGINE_H

#include <QObject>
#include <QApplication>
#include <QThread>
#include <QDebug>
#include <QTimer>

class CudaEngine : public QObject
{

    Q_OBJECT

public:
    CudaEngine();
    CudaEngine(const int nx, const int ny);
    ~CudaEngine();

    void build();
    void build(const int nx, const int ny);
    void setSimRate(const double simRate);

signals:
    void finished();
    void update(double, double**);

protected slots:
    void unboundSimRate();

public slots:
    void adddiffuse(int x, int y);
    void subdiffuse(int x, int y);
    void setPressure(double p);
    void stop();
    void run();

protected:
    bool built;
    bool running;
    bool simRateBounded;
    int nx, ny;
    double t;
    double maxSimRate;
    double pressure;
    double **data_cpu;
    double **data_gpu;
    QTimer *simTimer;


};

#endif // CUDAENGINE_H
