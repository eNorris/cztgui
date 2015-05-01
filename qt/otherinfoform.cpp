#include "otherinfoform.h"
#include "ui_otherinfoform.h"

#include "protocoldialog.h"

OtherInfoForm::OtherInfoForm(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::OtherInfoForm)
{
    ui->setupUi(this);

    //protocolDialog = new ProtocolDialog(this);


}

OtherInfoForm::~OtherInfoForm()
{
    delete ui;
}

int OtherInfoForm::getProtocolType()
{
    return ui->protocolDropBox->currentIndex();
}

QString OtherInfoForm::getProtocolName()
{
    return ui->protocolDropBox->itemText(getProtocolType());
}

/*
void OtherInfoForm::on_startButton_clicked()
{
    int ci = ui->protocolDropBox->currentIndex();
    QString tempstr = ui->protocolDropBox->itemText(ci);
    protocolDialog->setProtocolText(tempstr);

    int index = ui->protocolDropBox->currentIndex();
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
*/


