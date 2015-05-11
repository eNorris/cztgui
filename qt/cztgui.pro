#-------------------------------------------------
#
# Project created by QtCreator 2015-03-03T19:20:24
#
#-------------------------------------------------

QT       += core gui sql
QTPLUGIN += qsqlmysql

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets printsupport

TARGET = cztgui
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
    cathodedialog.cpp \
    globals.cpp \
    systemconfigdialog.cpp \
    ClickablePlot.cpp \
    qcustomplot.cpp \
    SimEngine.cpp \
    util.cpp \
    protocoldialog.cpp \
    patientdata.cpp

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
    cathodedialog.h \
    globals.h \
    common_types.h \
    spectdmsharedlib_global.h \
    SpectDM_types.h \
    H3DASIC_types.h \
    spectdmdll.h \
    systemconfigdialog.h \
    qcustomplot.h \
    SimEngine.h \
    clickableplot.h \
    util.h \
    protocoldialog.h \
    patientdata.h

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
    cathodedialog.ui \
    systemconfigdialog.ui \
    protocoldialog.ui

# Add the library file for 64-bit Linux, links to BOTH libSpectDMSharedLib.so and libSpectDMSharedLib.so.42
# The two files are identical, but because of naming conventions, they have to both be there.
win32:CONFIG(release, debug|release): LIBS += -L$$PWD/release/ -lSpectDMSharedLib
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/debug/ -lSpectDMSharedLib
else:unix: LIBS += -L$$PWD/ -lSpectDMSharedLib

INCLUDEPATH += $$PWD/
DEPENDPATH += $$PWD/
