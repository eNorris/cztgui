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

public:
    int getProtocolType();
    QString getProtocolName();

protected:
    //ProtocolDialog *protocolDialog;
    //liu added
    // this event is called, when a new translator is loaded or the system language is changed
    void changeEvent(QEvent*); //liu added


protected slots:
    //void on_startButton_clicked();


};

#endif // OTHERINFOFORM_H
