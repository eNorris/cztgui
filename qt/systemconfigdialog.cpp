#include "systemconfigdialog.h"
#include "ui_systemconfigdialog.h"

SystemConfigDialog::SystemConfigDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::SystemConfigDialog)
{
    ui->setupUi(this);
}

SystemConfigDialog::~SystemConfigDialog()
{
    delete ui;
}


