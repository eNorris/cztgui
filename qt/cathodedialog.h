#ifndef CATHODEDIALOG_H
#define CATHODEDIALOG_H

#include "globals.h"
#include "spectdmdll.h"

#include <QDialog>
#include <QMessageBox>

namespace Ui {
    class CathodeDialog;
}

class CathodeDialog : public QDialog
{
    Q_OBJECT

public:
    explicit CathodeDialog(QWidget *parent = 0);
    ~CathodeDialog();

public:
    Ui::CathodeDialog *ui;

public:
    void UpdateASICCathodeItems();
    void UpdateHGAffectedWidgets(bool a_HGSet);
};

#endif // CATHODEDIALOG_H
