#ifndef PATIENTDATA_H
#define PATIENTDATA_H

#include <QObject>
#include <QString>
#include <QDate>

class PatientData : public QObject
{
    Q_OBJECT
public:
    explicit PatientData(QObject *parent = 0);
    //PatientData(const PatientObject &src);
    ~PatientData();

    void build(QString firstname, QString middlename, QString lastname, int patientid, QString gender, QDate birthdate, float wt, float ht);

    QString firstName;
    QString middleName;
    QString lastName;

    int patientId;
    QString gender;
    QDate birthdate;
    float weight;  // kg
    float height;  // cm

signals:

public slots:
};

#endif // PATIENTDATA_H
