#include "systemform.h"
#include "ui_systemform.h"

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

    connect(ui->browseButton, SIGNAL(clicked()), this, SLOT(on_browseClicked()));
    connect(ui->connectButton, SIGNAL(clicked()), this, SLOT(on_connectClicked()));
    connect(ui->fpgaButton, SIGNAL(clicked()), this, SLOT(on_fpgaClicked()));
    connect(ui->configButton, SIGNAL(clicked()), this, SLOT(on_globalClicked()));
    connect(ui->anodeButton, SIGNAL(clicked()), this, SLOT(on_anodeClicked()));
    connect(ui->cathodeButton, SIGNAL(clicked()), this, SLOT(on_cathodeClicked()));

    ui->lcdNumber->display(28);

    //QString dir = QFileDialog::getExistingDirectory(this, tr("Open Directory"),
    //                                             "/home",
    //                                             QFileDialog::ShowDirsOnly
    //                                             | QFileDialog::DontResolveSymlinks);
}

SystemForm::~SystemForm()
{
    delete ui;
}

void SystemForm::on_browseClicked()
{
    QString dirname = QFileDialog::getExistingDirectory(this, tr("Open Directory"),
                                                        "/home",
                                                        QFileDialog::ShowDirsOnly
                                                        | QFileDialog::DontResolveSymlinks);
    if(dirname.length() != 0)
        ui->browseLineEdit->setText(dirname);
    qDebug() << dirname;
}

void SystemForm::on_connectClicked()
{
    connectDialog->exec();
}

void SystemForm::on_fpgaClicked()
{
    fpgaDialog->exec();
}

void SystemForm::on_globalClicked()
{
    configDialog->exec();
}

void SystemForm::on_anodeClicked()
{
    anodeDialog->exec();
}

void SystemForm::on_cathodeClicked()
{
    cathodeDialog->exec();
}
