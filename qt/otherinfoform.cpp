#include "otherinfoform.h"
#include "ui_otherinfoform.h"

OtherInfoForm::OtherInfoForm(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::OtherInfoForm)
{
    ui->setupUi(this);
}

OtherInfoForm::~OtherInfoForm()
{
    delete ui;
}
