#ifndef SYSTEMFORM_H
#define SYSTEMFORM_H

#include <QWidget>
#include <QFileDialog>
#include <QDebug>

#include "connectdialog.h"
#include "fpgadialog.h"
#include "globalconfigdialog.h"
#include "anodedialog.h"
#include "cathodedialog.h"

namespace Ui {
class SystemForm;
}

class SystemForm : public QWidget
{
    Q_OBJECT

public:
    explicit SystemForm(QWidget *parent = 0);
    ~SystemForm();

private:
    Ui::SystemForm *ui;

    ConnectDialog *connectDialog;
    FpgaDialog *fpgaDialog;
    GlobalConfigDialog *configDialog;
    AnodeDialog *anodeDialog;
    CathodeDialog *cathodeDialog;

protected slots:
    void on_browseClicked();
    void on_connectClicked();
    void on_fpgaClicked();
    void on_globalClicked();
    void on_anodeClicked();
    void on_cathodeClicked();
};

#endif // SYSTEMFORM_H
