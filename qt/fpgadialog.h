#ifndef FPGADIALOG_H
#define FPGADIALOG_H

#include <QDialog>

namespace Ui {
class FpgaDialog;
}

class FpgaDialog : public QDialog
{
    Q_OBJECT

public:
    explicit FpgaDialog(QWidget *parent = 0);
    ~FpgaDialog();

private:
    Ui::FpgaDialog *ui;
};

#endif // FPGADIALOG_H
