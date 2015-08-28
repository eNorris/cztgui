#include "patientdata.h"

PatientData::PatientData(QObject *parent) : QObject(parent)
{

}

PatientData::~PatientData()
{

}

bool PatientData::operator<(const PatientData& other) const
{
    if(lastName.toLower() < other.lastName.toLower())
        return true;
    else if(lastName.toLower() > other.lastName.toLower())
        return false;

    if(middleName.toLower() < other.middleName.toLower())
        return true;
    else if(middleName.toLower() > other.middleName.toLower())
        return false;

    if(firstName.toLower() < other.firstName.toLower())
        return true;
    else if(firstName.toLower() > other.firstName.toLower())
        return false;

    return patientId < other.patientId;
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

