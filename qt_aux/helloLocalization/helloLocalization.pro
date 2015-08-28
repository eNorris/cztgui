#-------------------------------------------------
#
# Project created by QtCreator 2015-08-26T15:13:42
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = helloLocalization
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp

HEADERS  += mainwindow.h

FORMS    += mainwindow.ui

TRANSLATIONS = app_sp.ts \
               app_fr.ts

CODECFORTR = ISO-8859-5
CODECFORSRC = UTF-8
