
#include "util.h"

void delay(int millisecs)
{
    QTime dieTime = QTime::currentTime().addMSecs(millisecs);
    while(QTime::currentTime() < dieTime);
        QCoreApplication::processEvents(QEventLoop::AllEvents, 100);
}
