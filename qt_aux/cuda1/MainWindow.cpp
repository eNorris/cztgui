#include "MainWindow.h"
#include "ui_ui_MainWindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    fpsBound = false;

    ClickablePlot *customPlot = ui->customPlot;
    customPlot->setDims(NX, NY);

    engine = new CudaEngine(NX, NY);
    engine->setSimRate(30);

    connect(ui->customPlot, SIGNAL(addmousemoved(int, int)), engine, SLOT(adddiffuse(int,int)), Qt::QueuedConnection);
    connect(ui->customPlot, SIGNAL(submousemoved(int, int)), engine, SLOT(subdiffuse(int,int)), Qt::QueuedConnection);
    connect(ui->actionStop, SIGNAL(triggered()), engine, SLOT(stop()), Qt::DirectConnection);

    connect(ui->customPlot, SIGNAL(wheelscroll(int)), ui->verticalSlider, SLOT(autoscroll(int)));
    connect(ui->verticalSlider, SIGNAL(newpressure(double)), engine, SLOT(setPressure(double)), Qt::DirectConnection);

    connect(&dataTimer, SIGNAL(timeout()), this, SLOT(fpsUnbound()));
    dataTimer.start(1000.0 / FPS_LIMIT); // Interval 0 means to refresh as fast as possible

    QThread *thread = new QThread;
    connect(engine, SIGNAL(update(double, double*)), this, SLOT(updatesurf(double, double*)));
    connect(engine, SIGNAL(finished()), thread, SLOT(quit()));
    connect(engine, SIGNAL(finished()), engine, SLOT(deleteLater()));
    //connect(engine, SIGNAL(error(QString)), this, SLOT(errorString(QString)));
    connect(thread, SIGNAL(started()), engine, SLOT(run()));
    connect(thread, SIGNAL(finished()), thread, SLOT(deleteLater()));
    engine->moveToThread(thread);
    thread->start();
}

MainWindow::~MainWindow()
{
    delete ui;
}



void MainWindow::updatesurf(double t, float *data)
{

    //qDebug() << "updating surf t = " << t;

    static int simcount = 0;
    static double lastt = 0.0;
    simcount++;
    if(fpsBound)
        return;

#if QT_VERSION < QT_VERSION_CHECK(4, 7, 0)
    double key = 0;
#else
    double key = QDateTime::currentDateTime().toMSecsSinceEpoch()/1000.0;
#endif

    static double lastPointKey = key;
    double total = 0.0;
    if (key-lastPointKey > 0.01) // at most add point every 10 ms
    {
        //double x, y, z;

        for (int xIndex=0; xIndex<NX; ++xIndex)
        {
          for (int yIndex=0; yIndex<NY; ++yIndex)
          {
              //colorMap->data()->setCell(xIndex, yIndex, data[xIndex][yIndex]);
              ui->customPlot->setData(xIndex, yIndex, data[NX*xIndex + yIndex]);
              total += data[NX*xIndex + yIndex];
            //colorMap->data()->cellToCoord(xIndex, yIndex, &x, &y);
            //double xx = qSin(x + 5 * t);
            //int s = sgn(xx);
            //z = qPow(qAbs(xx), t/10.0) * qPow(qAbs(qSin(y)), t/10.0);
            //colorMap->data()->setCell(xIndex, yIndex, z);
          }
        }
        total /= (NX * NY);
        ui->progressBar->setValue(int(1000 * total));

        // set the color gradient of the color map to one of the presets:
        //colorMap->setGradient(QCPColorGradient::gpPolar);
        //colorMap->rescaleDataRange();

        lastPointKey = key;
    }
    // make key axis range scroll with the data (at a constant range size of 8):
    //ui->customPlot->xAxis->setRange(key+0.25, 8, Qt::AlignRight);
    ui->customPlot->replot();

    // calculate frames per second:
    static double lastFpsKey;
    static int frameCount;
    ++frameCount;
    if (key-lastFpsKey > 2) // average fps over 2 seconds
    {
      ui->statusBar->showMessage(
            QString("%1 FPS,   %2 Sim Updates/sec, %3 Sim Ticks/sec    Average Intensity: %4")
            .arg(frameCount/(key-lastFpsKey), 0, 'f', 2)
            .arg(simcount/(key-lastFpsKey), 0, 'f', 2)
            .arg((t-lastt)/(key-lastFpsKey), 2, 'f', 4)
            .arg(total, 2, 'f', 4)
            , 0);
      lastFpsKey = key;
      lastt = t;
      frameCount = 0;
      simcount = 0;
    }
    fpsBound = true;
}

void MainWindow::fpsUnbound()
{
    fpsBound = false;
}
