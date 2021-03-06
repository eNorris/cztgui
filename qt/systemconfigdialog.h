#ifndef SYSTEMCONFIGDIALOG_H
#define SYSTEMCONFIGDIALOG_H

#include <QDialog>
#include <QMessageBox>

#include "globals.h"
#include "spectdmdll.h"

#include "qdebug.h"

namespace Ui {
    class SystemConfigDialog;
}

class SystemConfigDialog : public QDialog
{
    Q_OBJECT

public:
    explicit SystemConfigDialog(QWidget *parent = 0);
    ~SystemConfigDialog();

    static SystemConfigDialog* m_pInstance;
    static SystemConfigDialog* GetInstance();

    static void SysStatusCallbackFunc(const char *a_Data);
    void AddSysStatusEntry(const std::string &a_Status);

    void UpdateGMPixelMapCheckbox();

    void loadDefaults();

private:
    Ui::SystemConfigDialog *ui;

protected:
    //liu added
    // this event is called, when a new translator is loaded or the system language is changed
    void changeEvent(QEvent*); //liu added


public slots:
    void UpdateSysConfigItems();

    void on_startCollectBtn_clicked();
    void on_stopCollectBtn_clicked();

    void on_stopSysBtn_clicked();
    void on_startSysBtn_clicked();

    void on_sendConfigButton_clicked();
};

#endif // SYSTEMCONFIGDIALOG_H
