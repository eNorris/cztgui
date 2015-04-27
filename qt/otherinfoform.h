#ifndef OTHERINFOFORM_H
#define OTHERINFOFORM_H

#include <QWidget>

class ProtocolDialog;

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

protected:
    ProtocolDialog *protocolDialog;

protected slots:
    void on_startButton_clicked();


};

#endif // OTHERINFOFORM_H
