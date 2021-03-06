#ifndef SIMENGINE
#define SIMENGINE

#include "globals.h"

#include <QObject>
#include <QThread>
#include <QDebug>
#include <QTime>
#include <QTimer>
#include <QCoreApplication>
#include <QString>
#include <QFile>
#include <QMessageBox>

#include "spectdmdll.h"
#include <limits>


class SimEngine : public QObject
{

    Q_OBJECT

public:
    SimEngine() : QObject(), built(false), running(false), nx(0), ny(0), t(0), pressure(1.0), data(NULL), prevdata(NULL) {}
    SimEngine(const int nx, const int ny) : SimEngine() {build(nx, ny);}
    ~SimEngine();

    void build();
    void build(const int nx, const int ny);
    void zeroData();

    double& operator()(const int xIndex, const int yIndex);

signals:
    void finished();
    void update(double, double**);

public slots:
    void stop();
    void run(int imgct, float exp, float lat);

    void savePhotons(QString str);
    void loadPhotons(QString str, double **data);

    void delay(int millisecs);

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

