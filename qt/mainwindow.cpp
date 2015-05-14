#include "mainwindow.h"
#include "ui_mainwindow.h"

#include "protocoldialog.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow), db(NULL), model(NULL)
{
    //QLocale::setDefault(QLocale(QLocale::English, QLocale::UnitedStates));
    QLocale::setDefault(QLocale(QLocale::Chinese, QLocale::China));

    ui->setupUi(this);

    protocolDialog = new ProtocolDialog(this);

    //buildPatientDataBase();
    db = db_connect("/media/Storage/cztgui/qt/patientdata.db");
    dbFetchPatientInfo();
    buildModel();
    updateSheet();

    connect(ui->tableView, SIGNAL(clicked(QModelIndex)), this, SLOT(updateChildren(QModelIndex)));

    connect(ui->tableView, SIGNAL(activated(QModelIndex)), this, SLOT(updateChildren(QModelIndex)));

    connect(this, SIGNAL(emitUpdateChildren(QModelIndex, QVector<PatientData*>&)), ui->patientInfoForm, SLOT(updateChildren(QModelIndex, QVector<PatientData*>&)));

    //connect(ui->tableView, SIGNAL(clicked(QModelIndex)), pat, SLOT(updateChildren(QModelIndex)));

    //ui->columnView->setModel(&model);

    //    cview->setModel(&model);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::updateChildren(QModelIndex indx)
{
    emit emitUpdateChildren(indx, patientVector);
}

void MainWindow::on_acquireProtocolButton_clicked()
{
    int index = ui->otherInfoForm->getProtocolType();

    QString tempstr = ui->otherInfoForm->getProtocolName(); //ui->protocolDropBox->itemText(ci);
    protocolDialog->setProtocolText(tempstr);

    switch(index)
    {
    case 0:
        protocolDialog->setProtocolTime(5.0);
        break;
    case 1:
        protocolDialog->setProtocolTime(30.0);
        break;
    case 2:
        protocolDialog->setProtocolTime(55.5);
        break;
    }

    protocolDialog->exec();
}

QSqlDatabase* MainWindow::db_connect(QString dbname)
{
    if(db != NULL)
    {
        delete db;
        db = NULL;
    }

    //qDebug() << "before creation: " << db;
    db = new QSqlDatabase;
    //qDebug() << "after creation: " << db;

    *db = QSqlDatabase::addDatabase("QSQLITE");
    db->setHostName("localhost");
    db->setDatabaseName(dbname);
    db->setUserName("root");
    db->setPassword("rootpassword");

    bool ok = db->open();
    //qDebug() << "ok = " << ok;
    //qDebug("%s.", qPrintable(db->lastError().text()));

    return db;
}

void MainWindow::dbFetchPatientInfo()
{
    if(db == NULL)
        qDebug() << "NULL Database!";

    QSqlQuery query(*db);
    bool qgood = query.exec("select * from patient");

    if(qgood)
    {

        while(query.next())
        {
            PatientData *p = new PatientData();
            p->firstName = query.value(0).toString();
            p->middleName = query.value(1).toString();
            p->lastName = query.value(2).toString();
            p->patientId = query.value(3).toInt();
            p->gender = query.value(4).toString();
            p->birthdate = QDate::fromString(query.value(5).toString(), "M/d/yyyy");
            p->weight = query.value(6).toFloat();
            p->height = query.value(7).toFloat();

            patientVector.append(p);
        }
    }
    else
    {
        qDebug() << "Query returned failure!";
        qDebug("%s.", qPrintable(db->lastError().text()));

    }
}

void MainWindow::buildModel()
{
    model = new QStandardItemModel(patientVector.size(),6,this); //2 Rows and 3 Columns
    model->setHorizontalHeaderItem(0, new QStandardItem(QString(tr("Patient Name"))));
    model->setHorizontalHeaderItem(1, new QStandardItem(QString("Patient Id")));
    model->setHorizontalHeaderItem(2, new QStandardItem(QString("Study Name")));
    model->setHorizontalHeaderItem(3, new QStandardItem(QString("Start Date/Time")));
    model->setHorizontalHeaderItem(4, new QStandardItem(QString("Status")));
    model->setHorizontalHeaderItem(5, new QStandardItem(QString("Accession Number")));

    ui->tableView->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);


}

void MainWindow::updateSheet()
{

    qSort(patientVector.begin(), patientVector.end(), PtrLess<PatientData>());

    for(int i = 0; i < patientVector.size(); i++)
    {
        model->setItem(i, 0, new QStandardItem(patientVector[i]->firstName + " " + patientVector[i]->lastName));
        model->setItem(i, 1, new QStandardItem(QString::number(patientVector[i]->patientId)));
        model->setItem(i, 2, new QStandardItem("Experiment 123"));
        model->setItem(i, 3, new QStandardItem(patientVector[i]->birthdate.toString()));
        model->setItem(i, 4, new QStandardItem("Ongoing"));
        model->setItem(i, 5, new QStandardItem("???"));
    }

    ui->tableView->setModel(model);

}

