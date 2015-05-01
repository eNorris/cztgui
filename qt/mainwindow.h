#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QStandardItemModel>
#include <QDateTime>
#include <QVector>

#include "patientdata.h"

class ProtocolDialog;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

    void buildPatientDataBase();

protected:
    ProtocolDialog *protocolDialog;

    QVector<PatientData*> patientVector;

private:
    Ui::MainWindow *ui;

signals:
    void emitUpdateChildren(QModelIndex, QVector<PatientData*>&);

public slots:
    void updateChildren(QModelIndex indx);
    void on_acquireProtocolButton_clicked();
};

#endif // MAINWINDOW_H
