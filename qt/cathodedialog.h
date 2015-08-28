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

    void loadDefaults();

public:
    void UpdateASICCathodeItems();
    void UpdateHGAffectedWidgets(bool a_HGSet);

protected:
    //liu added
    // this event is called, when a new translator is loaded or the system language is changed
    void changeEvent(QEvent*); //liu added


public slots:
    void on_cathode_UpdateASICButton_clicked();
    void on_updateCathodeChannelButton_clicked();
};

#endif // CATHODEDIALOG_H
