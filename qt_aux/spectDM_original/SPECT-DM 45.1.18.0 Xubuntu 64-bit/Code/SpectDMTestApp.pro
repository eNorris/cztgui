#-------------------------------------------------
#
# Project created by QtCreator 2014-03-03T14:26:54
#
#-------------------------------------------------

QT       += core gui
QT       += network

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = SpectDMTestApp
TEMPLATE = app

VERSION = 43.1.21.46

DEFINES += APP_VERSION=\\\"$$VERSION\\\"
# prevents the resulting file from having major version in it's name
TARGET_EXT = .exe


SOURCES += main.cpp\
        mainwindow.cpp

HEADERS  += mainwindow.h \
    spectdmdll.h

FORMS    += mainwindow.ui

#unix|win32: LIBS += -L$$PWD/ -lSpectDM42

#win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../API/SpectDM/release/ -lSpectDM42
#else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../evDetectorCommon/debug/ -levDetectorCommon
#else:unix: LIBS += -L$$PWD/../evDetectorCommon/ -levDetectorCommon

#INCLUDEPATH += $$PWD/
#DEPENDPATH += $$PWD/

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../API/SpectDMSharedLib/release/ -lSpectDMSharedLib42
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../API/SpectDMSharedLib/debug/ -lSpectDMSharedLib42
else:unix: LIBS += -L$$PWD/../../API/SpectDMSharedLib/ -L$$PWD -lSpectDMSharedLib

INCLUDEPATH += $$PWD/../../API/SpectDMSharedLib/debug
DEPENDPATH += $$PWD/../../API/SpectDMSharedLib/debug
