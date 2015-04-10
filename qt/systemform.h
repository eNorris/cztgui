#ifndef SYSTEMFORM_H
#define SYSTEMFORM_H

#include <QWidget>
#include <QFileDialog>
#include <QDebug>

/*
#include "connectdialog.h"
#include "fpgadialog.h"
#include "globalconfigdialog.h"
#include "anodedialog.h"
#include "cathodedialog.h"
#include "systemconfigdialog.h"
*/

class ConnectDialog;
class FpgaDialog;
class GlobalConfigDialog;
class AnodeDialog;
class CathodeDialog;
class SystemConfigDialog;

namespace Ui {
    class SystemForm;
}

class SystemForm : public QWidget
{
    Q_OBJECT

public:
    explicit SystemForm(QWidget *parent = 0);
    ~SystemForm();

    ConnectDialog *connectDialog;
    FpgaDialog *fpgaDialog;
    GlobalConfigDialog *configDialog;
    AnodeDialog *anodeDialog;
    CathodeDialog *cathodeDialog;
    SystemConfigDialog *systemConfigDialog;

private:
    Ui::SystemForm *ui;



protected slots:
    void on_browseButton_clicked();
    void on_connectButton_clicked();
    void on_fpgaButton_clicked();
    void on_globalConfigButton_clicked();
    void on_anodeButton_clicked();
    void on_cathodeButton_clicked();
    void on_systemConfigButton_clicked();
};

#endif // SYSTEMFORM_H
