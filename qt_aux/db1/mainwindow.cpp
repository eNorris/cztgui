#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    createConnection();
}

MainWindow::~MainWindow()
{
    delete ui;
}

bool MainWindow::createConnection()
{
    QSqlDatabase db = QSqlDatabase::addDatabase("QSQLITE"); //QSqlDatabase::addDatabase("QMYSQL");
    db.setHostName("localhost");
    db.setDatabaseName("/media/Storage/cztgui/qt_aux/db1/patientdata.db");
    //db.setDatabaseName("/media/Storage/catgui/qt_aux/db1/patientdfgdfgdata.db");
    db.setUserName("root");
    db.setPassword("rootpassword");

    bool ok = db.open();
    qDebug() << "ok = " << ok;
    qDebug("%s.", qPrintable(db.lastError().text()));

    QSqlQuery query(db);
    bool qgood = query.exec("select firstname from patient where gender == 'f'");

    if(qgood)
    {
        qDebug() << "Query returned success";
        qDebug("%s.", qPrintable(db.lastError().text()));
    }
    else
    {
        qDebug() << "Qeury failed";
        qDebug("%s.", qPrintable(db.lastError().text()));
    }

}
