#ifndef VERTICALAUTOSLIDER_H
#define VERTICALAUTOSLIDER_H

#include <QSlider>

class VerticalAutoSlider : public QSlider
{

    Q_OBJECT

public:
    VerticalAutoSlider(QWidget *parent = 0);
    virtual ~VerticalAutoSlider() {}

protected:
    const static float SCALE_FACTOR = 1.10;
    const static int ADD_FACTOR = 10;

signals:
    void newpressure(double);

private slots:
    void autoscroll(int);
    void autoscrollscale(int);
};

#endif // VERTICALAUTOSLIDER_H
