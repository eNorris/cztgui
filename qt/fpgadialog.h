#ifndef FPGADIALOG_H
#define FPGADIALOG_H

#include "globals.h"

#include <QDialog>
#include <QMessageBox>

#include "spectdmdll.h"

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

protected slots:
    void on_UpdateGMButton_clicked();
};

#endif // FPGADIALOG_H
