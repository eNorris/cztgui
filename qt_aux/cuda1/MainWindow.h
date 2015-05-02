#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>
#include <QThread>

#include "utils.h"
//#include "SimEngine.h"
#include "cudaengine.h"

class QCPColorMap;
class QCPColorScale;
class QCPMarginGroup;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    static const int NX = 300, NY = 300;

    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

protected:
    //virtual void wheelEvent(QWheelEvent* event);
    //virtual void mouseMoveEvent(QMouseEvent *event);

private:
    Ui::MainWindow *ui;
    QTimer dataTimer;

    //int nx, ny;
    QCPColorMap *colorMap;
    QCPColorScale *colorScale;
    QCPMarginGroup *marginGroup;

    //SimEngine *engine;
    CudaEngine *engine;

    const static float FPS_LIMIT = 30.0;
    bool fpsBound;


private slots:
    void updatesurf(double t, float *data);
    void fpsUnbound();
};

#endif // MAINWINDOW_H
