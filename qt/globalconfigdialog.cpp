#include "globalconfigdialog.h"
#include "ui_globalconfigdialog.h"

GlobalConfigDialog::GlobalConfigDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::GlobalConfigDialog)
{
    ui->setupUi(this);
}

GlobalConfigDialog::~GlobalConfigDialog()
{
    delete ui;
}
