#ifndef ANODEDIALOG_H
#define ANODEDIALOG_H

#include "globals.h"
#include "spectdmdll.h"

#include <QDialog>
#include <QMessageBox>

namespace Ui {
class AnodeDialog;
}

class AnodeDialog : public QDialog
{
    Q_OBJECT

public:
    explicit AnodeDialog(QWidget *parent = 0);
    ~AnodeDialog();

public:
    void UpdateASICAnodeItems();
    void UpdateHGAffectedWidgets(bool a_HGSet);

public:
    Ui::AnodeDialog *ui;

protected slots:
    void on_ASICAnode_UpdateASICButton_clicked();
    void on_updateAnodeChannelButton_clicked();
};

#endif // ANODEDIALOG_H
