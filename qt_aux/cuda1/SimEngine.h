#ifndef SIMENGINE
#define SIMENGINE

#include <QObject>
#include <QThread>
#include <QDebug>

class SimEngine : public QObject
{

    Q_OBJECT

public:
    SimEngine() : QObject(), built(false), running(false), nx(0), ny(0), t(0), pressure(1.0), data(NULL), prevdata(NULL) {}
    SimEngine(const int nx, const int ny) : SimEngine() {build(nx, ny);}
    ~SimEngine();

    void build();
    void build(const int nx, const int ny);

    double& operator()(const int xIndex, const int yIndex);

signals:
    void finished();
    void update(double, double**);

public slots:
    void adddiffuse(int x, int y);
    void subdiffuse(int x, int y);
    void setPressure(double p);
    void stop();
    void run();

protected:
    bool built;
    bool running;
    int nx, ny;
    double t;
    double pressure;
    double **data;
    double **prevdata;


};

#endif // SIMENGINE

