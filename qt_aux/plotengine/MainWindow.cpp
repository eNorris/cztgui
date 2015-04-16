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
    //colorScale = new QCPColorScale(ui->customPlot);
    //marginGroup = new QCPMarginGroup(ui->customPlot);

    engine = new SimEngine(NX, NY);

    /*
    //demoName = "Quadratic Demo";
    // generate some data:
    QVector<double> x(101), y(101); // initialize with entries 0..100
    for (int i=0; i<101; ++i)
    {
      x[i] = i/100.0; // x goes from -1 to 1
      y[i] = x[i]*cos(x[i]);  // let's plot a quadratic function
    }
    // create graph and assign data to it:
    customPlot->addGraph();
    customPlot->graph(0)->setData(x, y);
    // give the axes some labels:
    customPlot->xAxis->setLabel("x");
    customPlot->yAxis->setLabel("y");
    // set axes ranges, so we see all data:
    customPlot->xAxis->setRange(0, 1);
    customPlot->yAxis->setRange(0, .6);
    */

    /*

    //demoName = "Sinc Scatter Demo";
    customPlot->legend->setVisible(true);
    customPlot->legend->setFont(QFont("Helvetica",9));
    // set locale to english, so we get english decimal separator:
    customPlot->setLocale(QLocale(QLocale::English, QLocale::UnitedKingdom));
    // add confidence band graphs:
    customPlot->addGraph();
    QPen pen;
    pen.setStyle(Qt::DotLine);
    pen.setWidth(1);
    pen.setColor(QColor(180,180,180));
    customPlot->graph(0)->setName("Confidence Band 68%");
    customPlot->graph(0)->setPen(pen);
    customPlot->graph(0)->setBrush(QBrush(QColor(255,50,30,20)));
    customPlot->addGraph();
    customPlot->legend->removeItem(customPlot->legend->itemCount()-1); // don't show two confidence band graphs in legend
    customPlot->graph(1)->setPen(pen);
    customPlot->graph(0)->setChannelFillGraph(customPlot->graph(1));
    // add theory curve graph:
    customPlot->addGraph();
    pen.setStyle(Qt::DashLine);
    pen.setWidth(2);
    pen.setColor(Qt::red);
    customPlot->graph(2)->setPen(pen);
    customPlot->graph(2)->setName("Theory Curve");
    // add data point graph:
    customPlot->addGraph();
    customPlot->graph(3)->setPen(QPen(Qt::blue));
    customPlot->graph(3)->setLineStyle(QCPGraph::lsNone);
    customPlot->graph(3)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCross, 4));
    customPlot->graph(3)->setErrorType(QCPGraph::etValue);
    customPlot->graph(3)->setErrorPen(QPen(QColor(180,180,180)));
    customPlot->graph(3)->setName("Measurement");

    // generate ideal sinc curve data and some randomly perturbed data for scatter plot:
    QVector<double> x0(250), y0(250);
    QVector<double> yConfUpper(250), yConfLower(250);
    for (int i=0; i<250; ++i)
    {
      x0[i] = (i/249.0-0.5)*30+0.01; // by adding a small offset we make sure not do divide by zero in next code line
      y0[i] = qSin(x0[i])/x0[i]; // sinc function
      yConfUpper[i] = y0[i]+0.15;
      yConfLower[i] = y0[i]-0.15;
      x0[i] *= 1000;
    }
    QVector<double> x1(50), y1(50), y1err(50);
    for (int i=0; i<50; ++i)
    {
      // generate a gaussian distributed random number:
      double tmp1 = rand()/(double)RAND_MAX;
      double tmp2 = rand()/(double)RAND_MAX;
      double r = qSqrt(-2*qLn(tmp1))*qCos(2*M_PI*tmp2); // box-muller transform for gaussian distribution
      // set y1 to value of y0 plus a random gaussian pertubation:
      x1[i] = (i/50.0-0.5)*30+0.25;
      y1[i] = qSin(x1[i])/x1[i]+r*0.15;
      x1[i] *= 1000;
      y1err[i] = 0.15;
    }
    // pass data to graphs and let QCustomPlot determine the axes ranges so the whole thing is visible:
    customPlot->graph(0)->setData(x0, yConfUpper);
    customPlot->graph(1)->setData(x0, yConfLower);
    customPlot->graph(2)->setData(x0, y0);
    customPlot->graph(3)->setDataValueError(x1, y1, y1err);
    customPlot->graph(2)->rescaleAxes();
    customPlot->graph(3)->rescaleAxes(true);
    // setup look of bottom tick labels:
    customPlot->xAxis->setTickLabelRotation(30);
    customPlot->xAxis->setAutoTickCount(9);
    customPlot->xAxis->setNumberFormat("ebc");
    customPlot->xAxis->setNumberPrecision(1);
    customPlot->xAxis->moveRange(-10);
    // make top right axes clones of bottom left axes. Looks prettier:
    customPlot->axisRect()->setupFullAxesBox();
    */

    /*
#if QT_VERSION < QT_VERSION_CHECK(4, 7, 0)
  QMessageBox::critical(this, "", "You're using Qt < 4.7, the realtime data demo needs functions that are available with Qt 4.7 to work properly");
#endif
  //demoName = "Real Time Data Demo";

  customPlot->addGraph(); // blue line
  customPlot->graph(0)->setPen(QPen(Qt::blue));
  customPlot->graph(0)->setBrush(QBrush(QColor(240, 255, 200)));
  customPlot->graph(0)->setAntialiasedFill(false);
  customPlot->addGraph(); // red line
  customPlot->graph(1)->setPen(QPen(Qt::red));
  customPlot->graph(0)->setChannelFillGraph(customPlot->graph(1));

  customPlot->addGraph(); // blue dot
  customPlot->graph(2)->setPen(QPen(Qt::blue));
  customPlot->graph(2)->setLineStyle(QCPGraph::lsNone);
  customPlot->graph(2)->setScatterStyle(QCPScatterStyle::ssDisc);
  customPlot->addGraph(); // red dot
  customPlot->graph(3)->setPen(QPen(Qt::red));
  customPlot->graph(3)->setLineStyle(QCPGraph::lsNone);
  customPlot->graph(3)->setScatterStyle(QCPScatterStyle::ssDisc);

  customPlot->xAxis->setTickLabelType(QCPAxis::ltDateTime);
  customPlot->xAxis->setDateTimeFormat("hh:mm:ss");
  customPlot->xAxis->setAutoTickStep(false);
  customPlot->xAxis->setTickStep(2);
  customPlot->axisRect()->setupFullAxesBox();

  // make left and bottom axes transfer their ranges to right and top axes:
  connect(customPlot->xAxis, SIGNAL(rangeChanged(QCPRange)), customPlot->xAxis2, SLOT(setRange(QCPRange)));
  connect(customPlot->yAxis, SIGNAL(rangeChanged(QCPRange)), customPlot->yAxis2, SLOT(setRange(QCPRange)));

  // setup a timer that repeatedly calls MainWindow::realtimeDataSlot:
  connect(&dataTimer, SIGNAL(timeout()), this, SLOT(realtimeDataSlot()));
  dataTimer.start(10); // Interval 0 means to refresh as fast as possible
  */

    /*
    // configure axis rect:
    //customPlot->setInteractions(QCP::iRangeDrag|QCP::iRangeZoom); // this will also allow rescaling the color scale by dragging/zooming
    customPlot->axisRect()->setupFullAxesBox(true);
    customPlot->xAxis->setLabel("x");
    customPlot->yAxis->setLabel("y");
    customPlot->addGraph();

    // set up the QCPColorMap:
    nx = 200;
    ny = 200;
    colorMap = new QCPColorMap(customPlot->xAxis, customPlot->yAxis);
    customPlot->addPlottable(colorMap);
    ui->customPlot->setColorMap(colorMap);

    colorMap->data()->setSize(nx, ny); // we want the color map to have nx * ny data points
    colorMap->data()->setRange(QCPRange(-4, 4), QCPRange(-4, 4)); // and span the coordinate range -4..4 in both key (x) and value (y) dimensions
    // now we assign some data, by accessing the QCPColorMapData instance of the color map:

    ui->customPlot->plotLayout()->addElement(0, 1, colorScale); // add it to the right of the main axis rect
    colorScale->setType(QCPAxis::atRight); // scale shall be vertical bar with tick/axis labels right (actually atRight is already the default)
    colorMap->setColorScale(colorScale); // associate the color map with the color scale
    colorScale->axis()->setLabel("Magnetic Field Strength");

    // make sure the axis rect and color scale synchronize their bottom and top margins (so they line up):
    //QCPMarginGroup *marginGroup = new QCPMarginGroup(ui->customPlot);
    ui->customPlot->axisRect()->setMarginGroup(QCP::msBottom|QCP::msTop, marginGroup);
    colorScale->setMarginGroup(QCP::msBottom|QCP::msTop, marginGroup);

    // set the color gradient of the color map to one of the presets:
    //colorMap->setGradient(QCPColorGradient::gpPolar);
    colorMap->setGradient(QCPColorGradient::gpJet);
    //colorMap->rescaleDataRange();  // Put this in the real time loop to auto scale
    colorMap->setDataRange(QCPRange(0.0, 1.0));

    // rescale the key (x) and value (y) axes so the whole color map is visible:
    ui->customPlot->rescaleAxes();

    //connect(&dataTimer, SIGNAL(timeout()), this, SLOT(rt2DDataSlot()));
    //dataTimer.start(30); // Interval 0 means to refresh as fast as possible
    //connect()
    */


    connect(ui->customPlot, SIGNAL(addmousemoved(int, int)), engine, SLOT(adddiffuse(int,int)), Qt::DirectConnection);
    connect(ui->customPlot, SIGNAL(submousemoved(int, int)), engine, SLOT(subdiffuse(int,int)), Qt::DirectConnection);
    connect(ui->actionStop, SIGNAL(triggered()), engine, SLOT(stop()), Qt::DirectConnection);

    connect(ui->customPlot, SIGNAL(wheelscroll(int)), ui->verticalSlider, SLOT(autoscroll(int)));
    connect(ui->verticalSlider, SIGNAL(newpressure(double)), engine, SLOT(setPressure(double)), Qt::DirectConnection);

    connect(&dataTimer, SIGNAL(timeout()), this, SLOT(fpsUnbound()));
    dataTimer.start(1000.0 / FPS_LIMIT); // Interval 0 means to refresh as fast as possible

    QThread *thread = new QThread;
    connect(engine, SIGNAL(update(double, double**)), this, SLOT(updatesurf(double, double**)));
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
/*
void MainWindow::wheelEvent(QWheelEvent* event)
{
    if(event->delta() > 0) {
        // Zoom in
        //scale(ZOOM_FACTOR, ZOOM_FACTOR);
        ui->verticalSlider->setValue(ui->verticalSlider->value() * SCALE_FACTOR);
    } else {
        // Zooming out
        ui->verticalSlider->setValue(ui->verticalSlider->value() / SCALE_FACTOR);
    }
}
*/

void MainWindow::realtimeDataSlot()
{
  // calculate two new data points:
#if QT_VERSION < QT_VERSION_CHECK(4, 7, 0)
  double key = 0;
#else
  double key = QDateTime::currentDateTime().toMSecsSinceEpoch()/1000.0;
#endif
  static double lastPointKey = 0;
  if (key-lastPointKey > 0.01) // at most add point every 10 ms
  {
    double value0 = qSin(key); //qSin(key*1.6+qCos(key*1.7)*2)*10 + qSin(key*1.2+0.56)*20 + 26;
    double value1 = qCos(key); //qSin(key*1.3+qCos(key*1.2)*1.2)*7 + qSin(key*0.9+0.26)*24 + 26;
    // add data to lines:
    ui->customPlot->graph(0)->addData(key, value0);
    ui->customPlot->graph(1)->addData(key, value1);
    // set data of dots:
    ui->customPlot->graph(2)->clearData();
    ui->customPlot->graph(2)->addData(key, value0);
    ui->customPlot->graph(3)->clearData();
    ui->customPlot->graph(3)->addData(key, value1);
    // remove data of lines that's outside visible range:
    ui->customPlot->graph(0)->removeDataBefore(key-8);
    ui->customPlot->graph(1)->removeDataBefore(key-8);
    // rescale value (vertical) axis to fit the current data:
    ui->customPlot->graph(0)->rescaleValueAxis();
    ui->customPlot->graph(1)->rescaleValueAxis(true);
    lastPointKey = key;
  }
  // make key axis range scroll with the data (at a constant range size of 8):
  ui->customPlot->xAxis->setRange(key+0.25, 8, Qt::AlignRight);
  ui->customPlot->replot();

  // calculate frames per second:
  static double lastFpsKey;
  static int frameCount;
  ++frameCount;
  if (key-lastFpsKey > 2) // average fps over 2 seconds
  {
    ui->statusBar->showMessage(
          QString("%1 FPS, Total Data points: %2")
          .arg(frameCount/(key-lastFpsKey), 0, 'f', 0)
          .arg(ui->customPlot->graph(0)->data()->count()+ui->customPlot->graph(1)->data()->count())
          , 0);
    lastFpsKey = key;
    frameCount = 0;
  }
}

//void MainWindow::rt2DDataSlot(int nx, int ny, QCPColorMap *colorMap)
void MainWindow::rt2DDataSlot()
{

#if QT_VERSION < QT_VERSION_CHECK(4, 7, 0)
    double key = 0;
#else
    double key = QDateTime::currentDateTime().toMSecsSinceEpoch()/1000.0;
#endif
    static double lastPointKey = key; //0;
    static double firstKey = key;
    double t = key - firstKey;
    if (key-lastPointKey > 0.01) // at most add point every 10 ms
    {
        double x, y, z;
        for (int xIndex=0; xIndex<NX; ++xIndex)
        {
          for (int yIndex=0; yIndex<NY; ++yIndex)
          {
            colorMap->data()->cellToCoord(xIndex, yIndex, &x, &y);
            double xx = qSin(x + 5 * t);
            int s = sgn(xx);
            z = qPow(qAbs(xx), t/10.0) * qPow(qAbs(qSin(y)), t/10.0);
            colorMap->data()->setCell(xIndex, yIndex, z);
          }
        }

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
            QString("%1 FPS")
            .arg(frameCount/(key-lastFpsKey), 0, 'f', 0)
            , 0);
      lastFpsKey = key;
      frameCount = 0;
    }
}

void MainWindow::updatesurf(double t, double **data)
{

    //qDebug() << "updating surf t = " << t;

    static int simcount = 0;
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
              ui->customPlot->setData(xIndex, yIndex, data[xIndex][yIndex]);
              total += data[xIndex][yIndex];
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
            QString("%1 FPS, %2 Sim Updates/sec, Average Intensity: %3")
            .arg(frameCount/(key-lastFpsKey), 0, 'f', 2)
            .arg(simcount/(key-lastFpsKey), 0, 'f', 2)
            .arg(total, 2, 'f', 4)
            , 0);
      lastFpsKey = key;
      frameCount = 0;
      simcount = 0;
    }
    fpsBound = true;
}

void MainWindow::fpsUnbound()
{
    fpsBound = false;
}
