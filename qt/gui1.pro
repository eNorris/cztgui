#-------------------------------------------------
#
# Project created by QtCreator 2015-03-03T19:20:24
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = gui1
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    patientinfoform.cpp \
    studyinfoform.cpp \
    otherinfoform.cpp \
    imagetransferform.cpp \
    transferlogform.cpp \
    systemform.cpp \
    connectdialog.cpp \
    fpgadialog.cpp \
    globalconfigdialog.cpp \
    anodedialog.cpp \
    cathodedialog.cpp

HEADERS  += mainwindow.h \
    patientinfoform.h \
    studyinfoform.h \
    otherinfoform.h \
    imagetransferform.h \
    transferlogform.h \
    systemform.h \
    connectdialog.h \
    fpgadialog.h \
    globalconfigdialog.h \
    anodedialog.h \
    cathodedialog.h

FORMS    += mainwindow.ui \
    patientinfoform.ui \
    studyinfoform.ui \
    otherinfoform.ui \
    imagetransferform.ui \
    transferlogform.ui \
    systemform.ui \
    connectdialog.ui \
    fpgadialog.ui \
    globalconfigdialog.ui \
    anodedialog.ui \
    cathodedialog.ui
