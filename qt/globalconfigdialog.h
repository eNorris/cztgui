#ifndef GLOBALCONFIGDIALOG_H
#define GLOBALCONFIGDIALOG_H

#include "globals.h"

#include <QDialog>
#include <QMessageBox>

#include <spectdmdll.h>

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

signals:
    void updateCathodeItems();
    void updateAnodeItems();
    void updateAnodeCathodeHG(bool);

protected:
    void UpdateHGAffectedWidgets(bool a_HGSet);

    void UpdateASICAnodeItems();

protected slots:
    void on_ASICGlobal_UpdateASICButton_clicked();
};

#endif // GLOBALCONFIGDIALOG_H
