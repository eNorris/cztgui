#include "VerticalAutoslider.h"

VerticalAutoSlider::VerticalAutoSlider(QWidget *parent) : QSlider(parent)
{

}

void VerticalAutoSlider::autoscrollscale(int x)
{
    if(x > 0)
        setValue(value() * SCALE_FACTOR);
    else
        setValue(value() / SCALE_FACTOR);
    emit newpressure(value() / 100.0);
}

void VerticalAutoSlider::autoscroll(int x)
{
    if(x > 0)
        setValue(value() + ADD_FACTOR);
    else
        setValue(value() - ADD_FACTOR);
    emit newpressure(value() / 100.0);
}
