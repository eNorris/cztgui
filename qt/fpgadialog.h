#ifndef FPGADIALOG_H
#define FPGADIALOG_H

#include "globals.h"

#include <QDialog>
#include <QMessageBox>
#include <QDebug>

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
    void loadDefaults();

private:
    Ui::FpgaDialog *ui;

protected:
    //liu added
    // this event is called, when a new translator is loaded or the system language is changed
    void changeEvent(QEvent*); //liu added


protected slots:
    void on_UpdateGMButton_clicked();
};

#endif // FPGADIALOG_H
