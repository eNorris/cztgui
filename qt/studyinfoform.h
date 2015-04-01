#ifndef STUDYINFOFORM_H
#define STUDYINFOFORM_H

#include <QWidget>

namespace Ui {
class StudyInfoForm;
}

class StudyInfoForm : public QWidget
{
    Q_OBJECT

public:
    explicit StudyInfoForm(QWidget *parent = 0);
    ~StudyInfoForm();

private:
    Ui::StudyInfoForm *ui;

//protected slots:
//    void on_
};

#endif // STUDYINFOFORM_H
