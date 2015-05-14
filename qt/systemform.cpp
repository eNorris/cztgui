#include "systemform.h"
#include "ui_systemform.h"

#include "connectdialog.h"
#include "fpgadialog.h"
#include "globalconfigdialog.h"
#include "anodedialog.h"
#include "cathodedialog.h"
#include "systemconfigdialog.h"

#include "clickableplot.h"


SystemForm::SystemForm(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::SystemForm)
{
    ui->setupUi(this);

    connectDialog = new ConnectDialog(this);
    fpgaDialog = new FpgaDialog(this);
    configDialog = new GlobalConfigDialog(this);
    anodeDialog = new AnodeDialog(this);
    cathodeDialog = new CathodeDialog(this);
    systemConfigDialog = new SystemConfigDialog(this);


    ui->lcdNumber->display(28);


    // Set up the renderer
    fpsBound = false;

    ClickablePlot *customPlot = ui->customPlot;
    customPlot->setDims(NX, NY);

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

    connect(configDialog, SIGNAL(updateAnodeItems()), this, SLOT(doAnodeUpdate()));
    connect(configDialog, SIGNAL(updateCathodeItems()), this, SLOT(doCathodeUpdate()));
    connect(configDialog, SIGNAL(updateAnodeCathodeHG(bool)), this, SLOT(doAnodeCathodeHG(bool)));

    connect(connectDialog, SIGNAL(connected()), this, SLOT(loadDefaults()));

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

SystemForm::~SystemForm()
{
    delete ui;

    delete connectDialog;
    delete fpgaDialog;
    delete configDialog;
    delete anodeDialog;
    delete cathodeDialog;
    delete systemConfigDialog;
}

void SystemForm::loadDefaults()
{
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
}

void SystemForm::on_browseButton_clicked()
{
    QString dirname = QFileDialog::getExistingDirectory(this, tr("Open Directory"),
                                                        "/home",
                                                        QFileDialog::ShowDirsOnly
                                                        | QFileDialog::DontResolveSymlinks);
    if(dirname.length() != 0)
        ui->browseLineEdit->setText(dirname);
    qDebug() << dirname;
}

void SystemForm::on_connectButton_clicked()
{
    connectDialog->exec();
}

void SystemForm::on_fpgaButton_clicked()
{
    fpgaDialog->exec();
}

void SystemForm::on_globalConfigButton_clicked()
{
    configDialog->exec();
}

void SystemForm::on_anodeButton_clicked()
{
    anodeDialog->exec();
}

void SystemForm::on_cathodeButton_clicked()
{
    cathodeDialog->exec();
}

void SystemForm::on_systemConfigButton_clicked()
{
    systemConfigDialog->exec();
}

void SystemForm::on_startButton_clicked()
{
    emit startRunning(
                ui->imageCountSpinBox->value(),
                ui->exposureSpinBox->value(),
                ui->latencySpinBox->value());
}

void SystemForm::on_stopButton_clicked()
{

}

void SystemForm::doAnodeUpdate()
{
    anodeDialog->UpdateASICAnodeItems();
}

void SystemForm::doCathodeUpdate()
{
    cathodeDialog->UpdateASICCathodeItems();
}

void SystemForm::doAnodeCathodeHG(bool hgSet)
{
    anodeDialog->UpdateHGAffectedWidgets(hgSet);
    cathodeDialog->UpdateHGAffectedWidgets(hgSet);
}

void SystemForm::updatesurf(double t, double **data)
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

void SystemForm::fpsUnbound()
{
    fpsBound = false;
}


