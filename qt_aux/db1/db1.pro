#-------------------------------------------------
#
# Project created by QtCreator 2015-05-06T16:31:13
#
#-------------------------------------------------

QT       += core gui sql
QTPLUGIN += qsqlmysql

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = db1
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp

HEADERS  += mainwindow.h

FORMS    += mainwindow.ui
