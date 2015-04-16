#include "clickableplot.h"

ClickablePlot::ClickablePlot(QWidget *parent) : QCustomPlot(parent), activetrack(false)
{
    colorScale = new QCPColorScale(this);
    marginGroup = new QCPMarginGroup(this);

    axisRect()->setupFullAxesBox(true);
    xAxis->setLabel("x");
    yAxis->setLabel("y");
    addGraph();

    // set up the QCPColorMap:
    colorMap = new QCPColorMap(xAxis, yAxis);
    addPlottable(colorMap);
    setColorMap(colorMap);
}

ClickablePlot::~ClickablePlot()
{
    //QCustomPlot::~QCustomPlot();
}

void ClickablePlot::setColorMap(QCPColorMap *map)
{
    colorMap = map;
}

void ClickablePlot::setValue(const int x, const int y, const float v)
{
    colorMap->data()->setCell(x, y, v);
}

void ClickablePlot::setDims(const int nx, const int ny)
{
    colorMap->data()->setSize(nx, ny); // we want the color map to have nx * ny data points
    colorMap->data()->setRange(QCPRange(0, nx), QCPRange(0, ny)); // and span the coordinate range -4..4 in both key (x) and value (y) dimensions
    // now we assign some data, by accessing the QCPColorMapData instance of the color map:

    plotLayout()->addElement(0, 1, colorScale); // add it to the right of the main axis rect
    colorScale->setType(QCPAxis::atRight); // scale shall be vertical bar with tick/axis labels right (actually atRight is already the default)
    colorMap->setColorScale(colorScale); // associate the color map with the color scale
    colorScale->axis()->setLabel("Intensity [counts]");

    // make sure the axis rect and color scale synchronize their bottom and top margins (so they line up):
    axisRect()->setMarginGroup(QCP::msBottom|QCP::msTop, marginGroup);
    colorScale->setMarginGroup(QCP::msBottom|QCP::msTop, marginGroup);

    // set the color gradient of the color map to one of the presets:
    colorMap->setGradient(QCPColorGradient::gpJet);
    colorMap->setDataRange(QCPRange(0.0, 1.0));

    // rescale the key (x) and value (y) axes so the whole color map is visible:
    rescaleAxes();
}

void ClickablePlot::rescale()
{
    colorMap->rescaleDataRange(true);
    //colorScale->rescaleDataRange(nx);
}

void ClickablePlot::mousePressEvent(QMouseEvent *event)
{
    activetrack = true;
    clicktype = event->button();
}

void ClickablePlot::mouseMoveEvent(QMouseEvent *event)
{
    if(activetrack)
    {
        qreal mx = event->localPos().x(),
              my = event->localPos().y();

        int mtop = this->axisRect()->top(),
            mbot = this->axisRect()->bottom(),
            mlft = this->axisRect()->left(),
            mrht = this->axisRect()->right();

        double dtop = colorMap->data()->keyRange().upper,
               dbot = colorMap->data()->keyRange().lower,
               dlft = colorMap->data()->valueRange().lower,
               drht = colorMap->data()->valueRange().upper;

        double dx = (mx - mlft) * (drht - dlft) / (mrht - mlft) + dlft,
               dy = (my - mtop)* (dtop - dbot) / (mtop - mbot) + dtop;

        int xi, yi;
        colorMap->data()->coordToCell(dx, dy, &xi, &yi);

        if(clicktype == Qt::LeftButton)
            emit addmousemoved(xi, yi);
        else
            emit submousemoved(xi, yi);
    }
}

void ClickablePlot::mouseReleaseEvent(QMouseEvent *)
{
    activetrack = false;
}

void ClickablePlot::wheelEvent(QWheelEvent* event)
{
    emit wheelscroll(event->delta());
}
