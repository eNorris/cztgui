
#include "globals.h"
#include <QDateTime>

QString createTimestampedStr(const std::string &a_Str)
{
    QString l_TimestampedStr = QDateTime::currentDateTime().toString(timeStampFormat);
    l_TimestampedStr += " - ";
    l_TimestampedStr += a_Str.c_str();

    return l_TimestampedStr;
}
