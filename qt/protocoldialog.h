#ifndef PROTOCOLDIALOG_H
#define PROTOCOLDIALOG_H

#include <QDialog>

namespace Ui {
class ProtocolDialog;
}

class ProtocolDialog : public QDialog
{
    Q_OBJECT

public:
    explicit ProtocolDialog(QWidget *parent = 0);
    ~ProtocolDialog();

private:
    Ui::ProtocolDialog *ui;

public slots:
    void setProtocolText(QString str);
    void setProtocolTime(const double t);
};

#endif // PROTOCOLDIALOG_H
