#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QStandardItemModel>
#include <QDateTime>
#include <QVector>

#include <QtSql/QSql>
#include <QtSql/QSqlDatabase>
#include <QtSql/QSqlDriver>
#include <QtSql/QSqlQuery>
#include <QtSql/QSqlError>

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

    //void buildPatientDataBase();

protected:
    ProtocolDialog *protocolDialog;
    QSqlDatabase *db;
    QStandardItemModel *model;

    QVector<PatientData*> patientVector;

    QSqlDatabase* db_connect(QString dbname);
    void dbFetchPatientInfo();
    void buildModel();
    void updateSheet();


public:
    Ui::MainWindow *ui;

signals:
    void emitUpdateChildren(QModelIndex, QVector<PatientData*>&);

public slots:
    void updateChildren(QModelIndex indx);
    void on_acquireProtocolButton_clicked();
};

template <typename T>
struct PtrLess // public std::binary_function<bool, const T*, const T*>
{
  bool operator()(const T* a, const T* b) const
  {
    // may want to check that the pointers aren't zero...
    return *a < *b;
  }
};

#endif // MAINWINDOW_H
