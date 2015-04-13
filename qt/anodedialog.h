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

private:
    Ui::AnodeDialog *ui;
};

#endif // ANODEDIALOG_H
