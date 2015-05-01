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

void PatientInfoForm::updateChildren(QModelIndex idx, QVector<PatientData*> &data)
{
    //qDebug() << "updating the kids!";

    int row = idx.row();
    ui->firstNameLineEdit->setText(data[row]->firstName);
    ui->lastNameLineEdit->setText(data[row]->lastName);
    ui->idLineEdit->setText(QString::number(data[row]->patientId));

    if(data[row]->gender == "m")
        ui->genderComboBox->setCurrentIndex(0);
    else
        ui->genderComboBox->setCurrentIndex(1);

    ui->birthdateDateEdit->setDate(data[row]->birthdate);
    //ui->ageSpinBox->setValue((QDate::currentDate() - data[row]->birthdate).year());

    ui->weightLineEdit->setText(QString::number(data[row]->weight));
    ui->heightLineEdit->setText(QString::number(data[row]->height));

    //const QStandardItemModel *model = static_cast<const QStandardItemModel*>(idx.model());
    //QString s = model->data(idx).toString();
    //qDebug() << "s = " << s;
}
