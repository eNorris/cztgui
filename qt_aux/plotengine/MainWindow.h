#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>
#include <QThread>

#include "utils.h"
#include "SimEngine.h"

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
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

protected:
    //virtual void wheelEvent(QWheelEvent* event);
    //virtual void mouseMoveEvent(QMouseEvent *event);

private:
    Ui::MainWindow *ui;
    QTimer dataTimer;

    int nx, ny;
    QCPColorMap *colorMap;
    QCPColorScale *colorScale;
    QCPMarginGroup *marginGroup;

    SimEngine *engine;

    const static float FPS_LIMIT = 30.0;
    bool fpsBound;

    //const float SCALE_FACTOR = 1.10;

private slots:
    void realtimeDataSlot();
    //void rt2DDataSlot(int nx, int ny, QCPColorMap *colorMap);
    void rt2DDataSlot();
    void updatesurf(double t, double **data);
    void fpsUnbound();
};

#endif // MAINWINDOW_H
