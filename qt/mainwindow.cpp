#include "mainwindow.h"
#include "ui_mainwindow.h"

#include "protocoldialog.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    //QLocale::setDefault(QLocale(QLocale::English, QLocale::UnitedStates));
    QLocale::setDefault(QLocale(QLocale::Chinese, QLocale::China));

    ui->setupUi(this);

    protocolDialog = new ProtocolDialog(this);

    // Set the headers
    QStandardItemModel *model = new QStandardItemModel(2,6,this); //2 Rows and 3 Columns
    model->setHorizontalHeaderItem(0, new QStandardItem(QString(tr("Patient Name"))));
    model->setHorizontalHeaderItem(1, new QStandardItem(QString("Patient Id")));
    model->setHorizontalHeaderItem(2, new QStandardItem(QString("Study Name")));
    model->setHorizontalHeaderItem(3, new QStandardItem(QString("Start Date/Time")));
    model->setHorizontalHeaderItem(4, new QStandardItem(QString("Status")));
    model->setHorizontalHeaderItem(5, new QStandardItem(QString("Accession Number")));
    ui->tableView->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);

    buildPatientDataBase();

    for(int i = 0; i < patientVector.size(); i++)
    {
        model->setItem(i, 0, new QStandardItem(patientVector[i]->firstName + " " + patientVector[i]->lastName));
        model->setItem(i, 1, new QStandardItem(QString::number(patientVector[i]->patientId)));
        model->setItem(i, 2, new QStandardItem("Experiment 123"));
        model->setItem(i, 3, new QStandardItem(patientVector[i]->birthdate.toString()));
        model->setItem(i, 4, new QStandardItem("Ongoing"));
        model->setItem(i, 5, new QStandardItem("???"));
    }

    //model->setVerticalHeaderItem(0, new QStandardItem(QString("Fish!")));
    //QString s = QDateTime::currentDateTime().toString();
    //model->setItem(0, 0, new QStandardItem("Bob"));
    //model->setItem(0, 1, new QStandardItem("314159"));
    //model->setItem(0, 2, new QStandardItem("Experiment 429"));
    //model->setItem(0, 3, new QStandardItem(QDateTime::currentDateTime().toString()));
    //model->setItem(0, 4, new QStandardItem("Ongoing"));
    //model->setItem(0, 5, new QStandardItem("???"));
    //ui->tableView->setModel(model);
    ui->tableView->setModel(model);

    connect(ui->tableView, SIGNAL(clicked(QModelIndex)), this, SLOT(updateChildren(QModelIndex)));

    connect(this, SIGNAL(emitUpdateChildren(QModelIndex, QVector<PatientData*>&)), ui->patientInfoForm, SLOT(updateChildren(QModelIndex, QVector<PatientData*>&)));

    //connect(ui->tableView, SIGNAL(clicked(QModelIndex)), pat, SLOT(updateChildren(QModelIndex)));

    //ui->columnView->setModel(&model);

    //    cview->setModel(&model);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::buildPatientDataBase()
{
    PatientData *manish = new PatientData();
    manish->build("Manish", "", "Sharma", 1, "m", QDate(2020, 1, 1), 146.3, 201.2);
    PatientData *ashish = new PatientData();
    ashish->build("Ashish", "", "Avachat", 2, "m", QDate(1991, 2, 28), 146.3, 201.2);
    PatientData *lukas = new PatientData();
    lukas->build("Lukas", "", "Tucker", 3, "m", QDate(1988, 3, 15), 146.3, 201.2);
    PatientData *jonathan = new PatientData();
    jonathan->build("Jonathan", "", "Scott", 234, "m", QDate(1991, 5, 2), 146.3, 201.2);
    PatientData *edward = new PatientData();
    edward->build("Edward", "", "Norris", 501, "m", QDate(1990, 6, 30), 146.3, 201.2);
    PatientData *liu = new PatientData();
    liu->build("Xin", "", "Liu", 661, "m", QDate(1977, 5, 4), 146.3, 201.2);
    PatientData *mueller = new PatientData();
    mueller->build("Gary", "", "Mueller", 71, "m", QDate(1870, 8, 1), 146.3, 201.2);
    PatientData *castano = new PatientData();
    castano->build("Carlos", "", "Castano", 8, "m", QDate(1950, 9, 15), 146.3, 201.2);
    PatientData *alajo = new PatientData();
    alajo->build("Ayodegi", "", "Alajo", 9, "m", QDate(1941, 10, 31), 146.3, 201.2);
    PatientData *lee = new PatientData();
    lee->build("Hank", "", "Lee", 91, "m", QDate(1999, 11, 30), 146.3, 201.2);
    PatientData *erika = new PatientData();
    erika->build("Erika", "", "Tucker", 4, "f", QDate(2013, 12, 31), 146.3, 201.2);

    patientVector.append(manish);
    patientVector.append(ashish);
    patientVector.append(lukas);
    patientVector.append(jonathan);
    patientVector.append(edward);
    patientVector.append(liu);
    patientVector.append(mueller);
    patientVector.append(castano);
    patientVector.append(alajo);
    patientVector.append(lee);
    patientVector.append(erika);
}

void MainWindow::updateChildren(QModelIndex indx)
{
    emit emitUpdateChildren(indx, patientVector);
    //indx.data();
    //QStandardItemModel *model = static_cast<QStandardItemModel*>(ui->tableView->model());
    //qDebug() << "Did it!";
    //model->
    //emit update
}

void MainWindow::on_acquireProtocolButton_clicked()
{
    //int ci = ui->protocolDropBox->currentIndex();
    //

    //int index = ui->protocolDropBox->currentIndex();
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



