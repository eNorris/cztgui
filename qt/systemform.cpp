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

    // From mainwindow.cpp
    /*
    for(int i = 0; i < 128; i++)
    {
        anodeDialog->ui->anodeChannelNoCombo->addItem(QString("%1").arg(i + 1));
        anodeDialog->ui->anodeChannelMonitorCombo->addItem(QString("%1").arg(i + 1));
        //ui->anodeChannelNoCombo->addItem(QString("%1").arg(i + 1));
        //ui->anodeChannelMonitorCombo->addItem(QString("%1").arg(i + 1));
    }
    */

    /*
    for(int i = 0; i < 2; i++)
    {
        ui->cathodeChannelNoCombo->addItem(QString("%2").arg((i + 1)));
    }
    */

    // anode and cathode
    //UpdateHGAffectedWidgets(false);

    /*
    ui->individualAnodeRadio->setChecked(true);  // anode
    ui->individualCathodeRadio->setChecked(true); // cathode

    ui->disconnectButton->setEnabled(false);  // connect
    ui->stopSysBtn->setEnabled(false);  //sysconfig
    ui->startCollectBtn->setEnabled(false);  //sysconfig
    ui->stopCollectBtn->setEnabled(false);  //sysconfig

    ui->DACScombo->setEnabled(false);  //globalconfig
    ui->cathEnergyTimingCombo->setEnabled(false);   //cathode
    ui->anodeChannelMonitorCombo->setEnabled(false);  //andoe

    connect(ui->actionExit, SIGNAL(triggered()), this, SLOT(close()));

    QScrollArea* l_ScrollArea = new QScrollArea(this);
    l_ScrollArea->setWidget(ui->tabWidget);

    setCentralWidget(l_ScrollArea);

    // disable all non-connect tabs until we connect
    EnableNonConnectTabs(false);
    */

    //connect(ui->browseButton, SIGNAL(clicked()), this, SLOT(on_browseClicked()));
    //connect(ui->connectButton, SIGNAL(clicked()), this, SLOT(on_connectClicked()));
    //connect(ui->fpgaButton, SIGNAL(clicked()), this, SLOT(on_fpgaClicked()));
    //connect(ui->configButton, SIGNAL(clicked()), this, SLOT(on_globalClicked()));
    //connect(ui->anodeButton, SIGNAL(clicked()), this, SLOT(on_anodeClicked()));
    //connect(ui->cathodeButton, SIGNAL(clicked()), this, SLOT(on_cathodeClicked()));

    ui->lcdNumber->display(28);


    // Set up the renderer
    fpsBound = false;

    ClickablePlot *customPlot = ui->customPlot;
    customPlot->setDims(NX, NY);
    //colorScale = new QCPColorScale(ui->customPlot);
    //marginGroup = new QCPMarginGroup(ui->customPlot);

    engine = new SimEngine(NX, NY);  // Pixel array

    //customPlot->axisRect()->setupFullAxesBox(true);
    //customPlot->xAxis->setLabel("x");
    //customPlot->yAxis->setLabel("y");
    //customPlot->addGraph();

    // set up the QCPColorMap:
    //nx = 22;
    //ny = 22;
    //colorMap = new QCPColorMap(customPlot->xAxis, customPlot->yAxis);
    //customPlot->addPlottable(colorMap);
    //ui->customPlot->setColorMap(colorMap);

    //colorMap->data()->setSize(NX, NY); // we want the color map to have nx * ny data points
    //colorMap->data()->setRange(QCPRange(0, NX), QCPRange(0, NY)); // and span the coordinate range -4..4 in both key (x) and value (y) dimensions
    // now we assign some data, by accessing the QCPColorMapData instance of the color map:

    //ui->customPlot->plotLayout()->addElement(0, 1, colorScale); // add it to the right of the main axis rect
    //colorScale->setType(QCPAxis::atRight); // scale shall be vertical bar with tick/axis labels right (actually atRight is already the default)
    //colorMap->setColorScale(colorScale); // associate the color map with the color scale
    //colorScale->axis()->setLabel("Intensity [counts]");

    // make sure the axis rect and color scale synchronize their bottom and top margins (so they line up):
    //QCPMarginGroup *marginGroup = new QCPMarginGroup(ui->customPlot);
    //ui->customPlot->axisRect()->setMarginGroup(QCP::msBottom|QCP::msTop, marginGroup);
    //colorScale->setMarginGroup(QCP::msBottom|QCP::msTop, marginGroup);

    // set the color gradient of the color map to one of the presets:
    //colorMap->setGradient(QCPColorGradient::gpPolar);
    //colorMap->setGradient(QCPColorGradient::gpJet);
    //colorMap->rescaleDataRange();  // Put this in the real time loop to auto scale
    //colorMap->setDataRange(QCPRange(0.0, 1.0));

    // rescale the key (x) and value (y) axes so the whole color map is visible:
    //ui->customPlot->rescaleAxes();

    //connect(&dataTimer, SIGNAL(timeout()), this, SLOT(rt2DDataSlot()));
    //dataTimer.start(30); // Interval 0 means to refresh as fast as possible
    //connect()

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

    //connect(ui->customPlot, SIGNAL(addmousemoved(int, int)), engine, SLOT(adddiffuse(int,int)), Qt::DirectConnection);
    //connect(ui->customPlot, SIGNAL(submousemoved(int, int)), engine, SLOT(subdiffuse(int,int)), Qt::DirectConnection);
    //connect(ui->actionStop, SIGNAL(triggered()), engine, SLOT(stop()), Qt::DirectConnection);

    //connect(ui->customPlot, SIGNAL(wheelscroll(int)), ui->verticalSlider, SLOT(autoscroll(int)));
    //connect(ui->verticalSlider, SIGNAL(newpressure(double)), engine, SLOT(setPressure(double)), Qt::DirectConnection);

    connect(&dataTimer, SIGNAL(timeout()), this, SLOT(fpsUnbound()));
    dataTimer.start(1000.0 / FPS_LIMIT); // Interval 0 means to refresh as fast as possible

    QThread *thread = new QThread;
    connect(engine, SIGNAL(update(double, double**)), this, SLOT(updatesurf(double, double**)));
    connect(engine, SIGNAL(finished()), thread, SLOT(quit()));
    connect(engine, SIGNAL(finished()), engine, SLOT(deleteLater()));
    //connect(engine, SIGNAL(error(QString)), this, SLOT(errorString(QString)));
    //connect(thread, SIGNAL(started()), engine, SLOT(run()));
    connect(this, SIGNAL(startRunning(int,float,float)), engine, SLOT(run(int, float, float)));
    connect(thread, SIGNAL(finished()), thread, SLOT(deleteLater()));
    engine->moveToThread(thread);
    thread->start();

    //QString dir = QFileDialog::getExistingDirectory(this, tr("Open Directory"),
    //                                             "/home",
    //                                             QFileDialog::ShowDirsOnly
    //                                             | QFileDialog::DontResolveSymlinks);
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
    delay(500);
    systemConfigDialog->loadDefaults();
    delay(500);
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
      //ui->statusBar->showMessage(
      //      QString("%1 FPS, %2 Sim Updates/sec, Average Intensity: %3")
       //     .arg(frameCount/(key-lastFpsKey), 0, 'f', 2)
       //     .arg(simcount/(key-lastFpsKey), 0, 'f', 2)
       //     .arg(total, 2, 'f', 4)
       //     , 0);
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


