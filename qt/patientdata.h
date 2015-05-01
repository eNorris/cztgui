#ifndef PATIENTDATA_H
#define PATIENTDATA_H

#include <QObject>

class PatientData : public QObject
{
    Q_OBJECT
public:
    explicit PatientData(QObject *parent = 0);
    ~PatientData();

signals:

public slots:
};

#endif // PATIENTDATA_H
