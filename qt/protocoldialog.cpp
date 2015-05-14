#include "protocoldialog.h"
#include "ui_protocoldialog.h"
#include "ui_mainwindow.h"

#include "systemform.h"
#include "mainwindow.h"

#include "clickableplot.h"

ProtocolDialog::ProtocolDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::ProtocolDialog)
{
    ui->setupUi(this);

    fpsBound = false;

    ClickablePlot *customPlot = ui->customPlot;
    customPlot->setDims(NX, NY);

    sysForm = static_cast<MainWindow*>(parent)->ui->widget_6;

    engine = new SimEngine(NX, NY);  // Pixel array


    if(SpectDMDll::SetGMUpdateType(GMUpdateType_Broadcast))
    {
        // need to disable all single GM widgets
        //EnableSingleGMWidgets(false);
    }
    else
    {
        QMessageBox::warning(this, "SpectDM", SpectDMDll::GetLastError().c_str());
    }

    //connect(configDialog, SIGNAL(updateAnodeItems()), this, SLOT(doAnodeUpdate()));
    //connect(configDialog, SIGNAL(updateCathodeItems()), this, SLOT(doCathodeUpdate()));
    //connect(configDialog, SIGNAL(updateAnodeCathodeHG(bool)), this, SLOT(doAnodeCathodeHG(bool)));

    //connect(connectDialog, SIGNAL(connected()), this, SLOT(loadDefaults()));

    connect(&dataTimer, SIGNAL(timeout()), this, SLOT(fpsUnbound()));
    dataTimer.start(1000.0 / FPS_LIMIT); // Interval 0 means to refresh as fast as possible

    QThread *thread = new QThread;
    connect(engine, SIGNAL(update(double, double**)), this, SLOT(updatesurf(double, double**)));
    connect(engine, SIGNAL(finished()), thread, SLOT(quit()));
    connect(engine, SIGNAL(finished()), engine, SLOT(deleteLater()));
    connect(this, SIGNAL(startRunning(int,float,float)), engine, SLOT(run(int, float, float)));
    connect(thread, SIGNAL(finished()), thread, SLOT(deleteLater()));
    engine->moveToThread(thread);
    thread->start();
}

ProtocolDialog::~ProtocolDialog()
{
    delete ui;
}

void ProtocolDialog::setProtocolText(QString str)
{
    ui->protocolLabel->setText(str);
}

void ProtocolDialog::setProtocolTime(const double t)
{
    ui->countingTime->setValue(t);
}

void ProtocolDialog::loadDefaults()
{
    sysForm->loadDefaults();
    /*
    qDebug() << "Loading Defaults";


    delay(500);
    systemConfigDialog->loadDefaults();
    delay(5000);

    configDialog->loadDefaults();
    delay(500);

    fpgaDialog->loadDefaults();
    delay(500);

    anodeDialog->loadDefaults();
    delay(500);

    cathodeDialog->loadDefaults();
    */
}

void ProtocolDialog::updatesurf(double t, double **data)
{

    qDebug() << "updating surf t = " << t;

    static int simcount = 0;
    simcount++;
    if(fpsBound)
        return;

#if QT_VERSION < QT_VERSION_CHECK(4, 7, 0)
    double key = 0;
#else
    double key = QDateTime::currentDateTime().toMSecsSinceEpoch()/1000.0;
#endif

    ClickablePlot *plot = ui->customPlot;

    static double lastPointKey = key-1;
    double total = 0.0;
    if (key-lastPointKey > 0.01) // at most add point every 10 ms
    {
        for (int xIndex=0; xIndex<NX; ++xIndex)
        {
          for (int yIndex=0; yIndex<NY; ++yIndex)
          {
              plot->setValue(xIndex, yIndex, data[xIndex][yIndex]);
              //colorMap->data()->setCell(xIndex, yIndex, data[xIndex][yIndex]);
              total += data[xIndex][yIndex];
          }
        }
        total /= (NX * NY);
        qDebug() << "SystemForm[206]: total = " << total;

        // set the color gradient of the color map to one of the presets:
        //colorMap->setGradient(QCPColorGradient::gpPolar);
        //colorMap->rescaleDataRange();
        plot->rescale();

        lastPointKey = key;
    }

    qDebug() << "At the surface updater:";
    QString str = "";
    for(int i = 0; i < NY; i++)
    {
        for(int j = 0; j < NX; j++)
            str.append(QString::number(data[i][j]) + "   ");
        str.append("\n");
    }
    qDebug() << str;

    // make key axis range scroll with the data (at a constant range size of 8):
    //ui->customPlot->xAxis->setRange(key+0.25, 8, Qt::AlignRight);
    qDebug() << "replotted";
    ui->customPlot->replot();

    // calculate frames per second:
    static double lastFpsKey;
    static int frameCount;
    ++frameCount;
    if (key-lastFpsKey > 2) // average fps over 2 seconds
    {
      lastFpsKey = key;
      frameCount = 0;
      simcount = 0;
    }
    fpsBound = true;
}

void ProtocolDialog::fpsUnbound()
{
    fpsBound = false;
}

void ProtocolDialog::on_startButton_clicked()
{
    sysForm->doStartRunning(1, static_cast<float>(ui->countingTime->value()), 0);
}



