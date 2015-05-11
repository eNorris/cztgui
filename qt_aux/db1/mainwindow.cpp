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
    bool qgood = query.exec("select firstname from patient where gender == 'm'");

    if(qgood)
    {
        qDebug() << "Query returned success";
        qDebug("%s.", qPrintable(db.lastError().text()));

        int numrows = -1;
        if(db.driver()->hasFeature(QSqlDriver::QuerySize))
            numrows = query.size();
        else
        {
            query.last();
            numrows = query.at() + 1;
        }
        query.first();
        query.previous();

        qDebug() << "Results: " << numrows;

        while(query.next())
        {
            QString name = query.value(0).toString();
            qDebug() << name;
        }

    }
    else
    {
        qDebug() << "Qeury failed";
        qDebug("%s.", qPrintable(db.lastError().text()));
    }

    // Add a new row
    QSqlQuery q2;
    q2.prepare("INSERT INTO patient VALUES(?, ?, ?, ?, ?, ?, ?, ?)");
    q2.addBindValue("Alex");
    q2.addBindValue("Q.");
    q2.addBindValue("Dzurick");
    q2.addBindValue(556);
    q2.addBindValue("m");
    q2.addBindValue("9/8/1989");
    q2.addBindValue(445.3);
    q2.addBindValue(211.1);
    if(!q2.exec())
    {
        qDebug() << "q2 Query failed!";
    }

    QSqlQuery q3;
    if(!q3.exec("UPDATE patient SET firstname = 'Manish' WHERE id = 1"))
    {
        qDebug() << "q3 Query failed!";
    }
}
