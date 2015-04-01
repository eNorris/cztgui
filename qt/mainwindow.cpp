#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    //QLocale::setDefault(QLocale(QLocale::English, QLocale::UnitedStates));
    QLocale::setDefault(QLocale(QLocale::Chinese, QLocale::China));

    ui->setupUi(this);

    /*
    QStandardItemModel model;
    for (int groupnum = 0; groupnum < 3 ; ++groupnum)
    {
        // Create the phone groups as QStandardItems
        QStandardItem *group = new QStandardItem(QString("Group %1").arg(groupnum));

        // Append to each group 5 person as children
        for (int personnum = 0; personnum < 5 ; ++personnum)
        {
            QStandardItem *child = new QStandardItem(QString("Person %1 (group %2)").arg(personnum).arg(groupnum));
            // the appendRow function appends the child as new row
            group->appendRow(child);
        }
        // append group as new row to the model. model takes the ownership of the item
        model.appendRow(group);
    }
    */

    QStandardItemModel *model = new QStandardItemModel(2,6,this); //2 Rows and 3 Columns
    model->setHorizontalHeaderItem(0, new QStandardItem(QString(tr("Patient Name"))));
    model->setHorizontalHeaderItem(1, new QStandardItem(QString("Patient Id")));
    model->setHorizontalHeaderItem(2, new QStandardItem(QString("Study Name")));
    model->setHorizontalHeaderItem(3, new QStandardItem(QString("Start Date/Time")));
    model->setHorizontalHeaderItem(4, new QStandardItem(QString("Status")));
    model->setHorizontalHeaderItem(5, new QStandardItem(QString("Accession Number")));
    ui->tableView->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);

    //for (int c = 0; c < ui->tableView->horizontalHeader()->count(); ++c)
    //{
    //    ui->tableView->horizontalHeader()->setSectionResizeMode(
    //        c, QHeaderView::Stretch);
    //



    //model->setVerticalHeaderItem(0, new QStandardItem(QString("Fish!")));
    QString s = QDateTime::currentDateTime().toString();
    model->setItem(0, 0, new QStandardItem("Bob"));
    model->setItem(0, 1, new QStandardItem("314159"));
    model->setItem(0, 2, new QStandardItem("Experiment 429"));
    model->setItem(0, 3, new QStandardItem(QDateTime::currentDateTime().toString()));
    model->setItem(0, 4, new QStandardItem("Ongoing"));
    model->setItem(0, 5, new QStandardItem("???"));
    //ui->tableView->setModel(model);
    ui->tableView->setModel(model);

    //ui->columnView->setModel(&model);

    //    cview->setModel(&model);
}

MainWindow::~MainWindow()
{
    delete ui;
}
