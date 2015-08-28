#ifndef PATIENTINFOFORM_H
#define PATIENTINFOFORM_H

#include <QWidget>
#include <QDebug>
#include <QModelIndex>
#include <QStandardItemModel>
#include <QDate>

#include "patientdata.h"

namespace Ui {
class PatientInfoForm;
}

class PatientInfoForm : public QWidget
{
    Q_OBJECT

public:
    explicit PatientInfoForm(QWidget *parent = 0);
    ~PatientInfoForm();

private:
    Ui::PatientInfoForm *ui;

protected:
    //liu added
    // this event is called, when a new translator is loaded or the system language is changed
    void changeEvent(QEvent*); //liu added



public slots:
    void updateChildren(QModelIndex idx, QVector<PatientData*> &data);
    void updateAge(QDate date);
};

#endif // PATIENTINFOFORM_H
