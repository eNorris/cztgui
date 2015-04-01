#ifndef OTHERINFOFORM_H
#define OTHERINFOFORM_H

#include <QWidget>

namespace Ui {
class OtherInfoForm;
}

class OtherInfoForm : public QWidget
{
    Q_OBJECT

public:
    explicit OtherInfoForm(QWidget *parent = 0);
    ~OtherInfoForm();

private:
    Ui::OtherInfoForm *ui;
};

#endif // OTHERINFOFORM_H
