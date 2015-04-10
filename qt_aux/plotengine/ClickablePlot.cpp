#include "ClickablePlot.h"

ClickablePlot::ClickablePlot(QWidget *parent) : QCustomPlot(parent), activetrack(false)
{

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
