#ifndef CLICKABLEPLOT_H
#define CLICKABLEPLOT_H

#include "qcustomplot.h"

class ClickablePlot : public QCustomPlot
{

    Q_OBJECT

protected:
    QCPColorScale *colorScale;
    QCPMarginGroup *marginGroup;

public:
    ClickablePlot(QWidget *parent = 0);
    virtual ~ClickablePlot();

    void setColorMap(QCPColorMap *map);
    void setDims(const int nx, const int ny);
    void setValue(const int x, const int y, const float v);
    void rescale();

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
