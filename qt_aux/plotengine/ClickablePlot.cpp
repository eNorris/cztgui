#include "ClickablePlot.h"

ClickablePlot::ClickablePlot(QWidget *parent) : QCustomPlot(parent), activetrack(false)
{
    axisRect()->setupFullAxesBox(true);
    xAxis->setLabel("x");
    yAxis->setLabel("y");
    addGraph();

    colorMap = new QCPColorMap(xAxis, yAxis);
    addPlottable(colorMap);
    setColorMap(colorMap);

    colorScale = new QCPColorScale(this);
    marginGroup = new QCPMarginGroup(this);


}

void ClickablePlot::setDims(const int x, const int y)
{
    // set up the QCPColorMap:
    nx = x;
    ny = y;


    colorMap->data()->setSize(nx, ny); // we want the color map to have nx * ny data points
    colorMap->data()->setRange(QCPRange(0, 1), QCPRange(0, 1)); // and span the coordinate range -4..4 in both key (x) and value (y) dimensions
    // now we assign some data, by accessing the QCPColorMapData instance of the color map:


    plotLayout()->addElement(0, 1, colorScale); // add it to the right of the main axis rect
    colorScale->setType(QCPAxis::atRight); // scale shall be vertical bar with tick/axis labels right (actually atRight is already the default)
    colorMap->setColorScale(colorScale); // associate the color map with the color scale
    colorScale->axis()->setLabel("Temperature");

    // make sure the axis rect and color scale synchronize their bottom and top margins (so they line up):
    //QCPMarginGroup *marginGroup = new QCPMarginGroup(ui->customPlot);
    axisRect()->setMarginGroup(QCP::msBottom|QCP::msTop, marginGroup);
    colorScale->setMarginGroup(QCP::msBottom|QCP::msTop, marginGroup);

    // set the color gradient of the color map to one of the presets:
    //colorMap->setGradient(QCPColorGradient::gpPolar);
    colorMap->setGradient(QCPColorGradient::gpJet);
    //colorMap->rescaleDataRange();  // Put this in the real time loop to auto scale
    colorMap->setDataRange(QCPRange(0.0, 1.0));

    // rescale the key (x) and value (y) axes so the whole color map is visible:
    rescaleAxes();

    //connect(&dataTimer, SIGNAL(timeout()), this, SLOT(rt2DDataSlot()));
    //dataTimer.start(30); // Interval 0 means to refresh as fast as possible
    //connect()


}

void ClickablePlot::setData(const int ix, const int iy, const double v)
{
    colorMap->data()->setCell(ix, iy, v);
}

ClickablePlot::~ClickablePlot()
{
    //QCustomPlot::~QCustomPlot();
}

void ClickablePlot::mousePressEvent(QMouseEvent *event)
{
    activetrack = true;
    clicktype = event->button();
    //qDebug() << "clicked the mouse!";
}

void ClickablePlot::mouseMoveEvent(QMouseEvent *event)
{
    if(activetrack)
    {
        qreal mx = event->localPos().x(),
              my = event->localPos().y();
        //qDebug() << "Mouse coord: " << mx << "   " << my;

        int mtop = this->axisRect()->top(),
            mbot = this->axisRect()->bottom(),
            mlft = this->axisRect()->left(),
            mrht = this->axisRect()->right();
        //qDebug() << "mouse vals: " << mtop << "   " << mbot << "   "
        //         << mlft << "   " << mrht;

        double dtop = colorMap->data()->keyRange().upper,
               dbot = colorMap->data()->keyRange().lower,
               dlft = colorMap->data()->valueRange().lower,
               drht = colorMap->data()->valueRange().upper;
        //qDebug() << "disp vals: " << dtop << "   " << dbot << "   "
        //         << dlft << "   " << drht;

        double dx = (mx - mlft) * (drht - dlft) / (mrht - mlft) + dlft,
               dy = (my - mtop)* (dtop - dbot) / (mtop - mbot) + dtop;
        //qDebug() << "Disp coord: " << dx << "   " << dy;

        int xi, yi;
        colorMap->data()->coordToCell(dx, dy, &xi, &yi);
        //qDebug() << "Plot coord: " << xi << "   " << yi;

        if(clicktype == Qt::LeftButton)
            emit addmousemoved(xi, yi);
        else
            emit submousemoved(xi, yi);
    }
}

void ClickablePlot::mouseReleaseEvent(QMouseEvent *event)
{
    activetrack = false;
}

void ClickablePlot::wheelEvent(QWheelEvent* event)
{
    emit wheelscroll(event->delta());
}
