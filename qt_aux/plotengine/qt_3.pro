#-------------------------------------------------
#
# Project created by QtCreator 2015-02-20T08:53:15
#
#-------------------------------------------------

QT       += core gui

# Manually add printsupport to QCustomPlot packages
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets printsupport

TARGET = qt_3
TEMPLATE = app

SOURCES += main.cpp \
    MainWindow.cpp \
    qcustomplot.cpp \
    SimEngine.cpp \
    ClickablePlot.cpp \
    VerticalAutoslider.cpp

HEADERS  += MainWindow.h \
    qcustomplot.h \
    utils.h \
    SimEngine.h \
    ClickablePlot.h \
    VerticalAutoslider.h

FORMS    += ui_MainWindow.ui
