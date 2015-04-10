#include "systemform.h"
#include "ui_systemform.h"

#include "connectdialog.h"
#include "fpgadialog.h"
#include "globalconfigdialog.h"
#include "anodedialog.h"
#include "cathodedialog.h"
#include "systemconfigdialog.h"

SystemForm::SystemForm(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::SystemForm)
{
    ui->setupUi(this);

    connectDialog = new ConnectDialog(this);
    fpgaDialog = new FpgaDialog(this);
    configDialog = new GlobalConfigDialog(this);
    anodeDialog = new AnodeDialog(this);
    cathodeDialog = new CathodeDialog(this);
    systemConfigDialog = new SystemConfigDialog(this);

    //connect(ui->browseButton, SIGNAL(clicked()), this, SLOT(on_browseClicked()));
    //connect(ui->connectButton, SIGNAL(clicked()), this, SLOT(on_connectClicked()));
    //connect(ui->fpgaButton, SIGNAL(clicked()), this, SLOT(on_fpgaClicked()));
    //connect(ui->configButton, SIGNAL(clicked()), this, SLOT(on_globalClicked()));
    //connect(ui->anodeButton, SIGNAL(clicked()), this, SLOT(on_anodeClicked()));
    //connect(ui->cathodeButton, SIGNAL(clicked()), this, SLOT(on_cathodeClicked()));

    ui->lcdNumber->display(28);

    //QString dir = QFileDialog::getExistingDirectory(this, tr("Open Directory"),
    //                                             "/home",
    //                                             QFileDialog::ShowDirsOnly
    //                                             | QFileDialog::DontResolveSymlinks);
}

SystemForm::~SystemForm()
{
    delete ui;

    delete connectDialog;
    delete fpgaDialog;
    delete configDialog;
    delete anodeDialog;
    delete cathodeDialog;
    delete systemConfigDialog;
}

void SystemForm::on_browseButton_clicked()
{
    QString dirname = QFileDialog::getExistingDirectory(this, tr("Open Directory"),
                                                        "/home",
                                                        QFileDialog::ShowDirsOnly
                                                        | QFileDialog::DontResolveSymlinks);
    if(dirname.length() != 0)
        ui->browseLineEdit->setText(dirname);
    qDebug() << dirname;
}

void SystemForm::on_connectButton_clicked()
{
    connectDialog->exec();
}

void SystemForm::on_fpgaButton_clicked()
{
    fpgaDialog->exec();
}

void SystemForm::on_globalConfigButton_clicked()
{
    configDialog->exec();
}

void SystemForm::on_anodeButton_clicked()
{
    anodeDialog->exec();
}

void SystemForm::on_cathodeButton_clicked()
{
    cathodeDialog->exec();
}

void SystemForm::on_systemConfigButton_clicked()
{
    systemConfigDialog->exec();
}
