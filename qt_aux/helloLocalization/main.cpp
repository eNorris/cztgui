#include "mainwindow.h"
#include <QApplication>
#include <QTranslator>
#include <QLibraryInfo>
#include <QDebug>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    //QTranslator qtTranslator;
    //qtTranslator.load("qt_" + QLocale::system().name(),
    //        QLibraryInfo::location(QLibraryInfo::TranslationsPath));
    //app.installTranslator(&qtTranslator);

    //QTranslator myappTranslator;
    //myappTranslator.load("myapp_" + QLocale::system().name());
    //app.installTranslator(&myappTranslator);

    //QLocale::setDefault(QLocale(QLocale::Chinese, QLocale::China));
    //qDebug() << QLocale::system().language();

    MainWindow w;
    w.show();

    return app.exec();
}
