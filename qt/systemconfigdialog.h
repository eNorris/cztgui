#ifndef SYSTEMCONFIGDIALOG_H
#define SYSTEMCONFIGDIALOG_H

#include <QDialog>

namespace Ui {
    class SystemConfigDialog;
}

class SystemConfigDialog : public QDialog
{
    Q_OBJECT

public:
    explicit SystemConfigDialog(QWidget *parent = 0);
    ~SystemConfigDialog();

private:
    Ui::SystemConfigDialog *ui;
};

#endif // SYSTEMCONFIGDIALOG_H
