#ifndef PATIENTINFOFORM_H
#define PATIENTINFOFORM_H

#include <QWidget>

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
};

#endif // PATIENTINFOFORM_H
