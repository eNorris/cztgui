#include "patientinfoform.h"
#include "ui_patientinfoform.h"

PatientInfoForm::PatientInfoForm(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::PatientInfoForm)
{
    ui->setupUi(this);
}

PatientInfoForm::~PatientInfoForm()
{
    delete ui;
}
