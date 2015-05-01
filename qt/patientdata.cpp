#include "patientdata.h"

PatientData::PatientData(QObject *parent) : QObject(parent)
{

}

PatientData::~PatientData()
{

}

void PatientData::build(QString firstname, QString middlename, QString lastname, int patientid, QString gender, QDate birthdate, float wt, float ht)
{
    this->firstName = firstname;
    this->middleName = middlename;
    this->lastName = lastname;
    this->patientId = patientid;
    this->gender = gender;
    this->birthdate = birthdate;
    this->weight = wt;
    this->height = ht;
}
