#ifndef CLICKABLEPLOT_H
#define CLICKABLEPLOT_H

#include "qcustomplot.h"


class ClickablePlot : public QCustomPlot
{

    Q_OBJECT

public:
    ClickablePlot(QWidget *parent = 0);
    virtual ~ClickablePlot();

    void setColorMap(QCPColorMap *map){colorMap = map;}

    virtual void mousePressEvent(QMouseEvent *event);
    virtual void mouseMoveEvent(QMouseEvent *event);
    virtual void mouseReleaseEvent(QMouseEvent *event);
    virtual void wheelEvent(QWheelEvent* event);

protected:
    bool activetrack;
    int clicktype;
    QCPColorMap *colorMap;

signals:
    void addmousemoved(int x, int y);
    void submousemoved(int x, int y);
    void wheelscroll(int);
};

#endif // CLICKABLEPLOT_H
