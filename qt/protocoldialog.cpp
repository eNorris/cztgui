#include "protocoldialog.h"
#include "ui_protocoldialog.h"

ProtocolDialog::ProtocolDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::ProtocolDialog)
{
    ui->setupUi(this);
}

ProtocolDialog::~ProtocolDialog()
{
    delete ui;
}

void ProtocolDialog::setProtocolText(QString str)
{
    ui->protocolLabel->setText(str);
}

void ProtocolDialog::setProtocolTime(const double t)
{
    ui->protocolTimeSpinBox->setValue(t);
}
