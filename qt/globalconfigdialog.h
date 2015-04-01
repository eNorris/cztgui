#ifndef GLOBALCONFIGDIALOG_H
#define GLOBALCONFIGDIALOG_H

#include <QDialog>

namespace Ui {
class GlobalConfigDialog;
}

class GlobalConfigDialog : public QDialog
{
    Q_OBJECT

public:
    explicit GlobalConfigDialog(QWidget *parent = 0);
    ~GlobalConfigDialog();

private:
    Ui::GlobalConfigDialog *ui;
};

#endif // GLOBALCONFIGDIALOG_H
