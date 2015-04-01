#include "studyinfoform.h"
#include "ui_studyinfoform.h"

StudyInfoForm::StudyInfoForm(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::StudyInfoForm)
{
    ui->setupUi(this);
}

StudyInfoForm::~StudyInfoForm()
{
    delete ui;
}
