#ifndef CONNECTDIALOG_H
#define CONNECTDIALOG_H

#include "globals.h"
#include "util.h"

#include <QDialog>

#include "spectdmdll.h"
//#include "systemform.h"

namespace Ui {
class ConnectDialog;
}

class ConnectDialog : public QDialog
{
    Q_OBJECT

public:
    explicit ConnectDialog(QWidget *parent = 0);
    ~ConnectDialog();

    static ConnectDialog* m_pInstance;
    static ConnectDialog* GetInstance();
    //static void DeleteInstance();

    // These seem kinda useless, are they really necessary?
    static void ConnectStatusCallbackFunc(const char* a_Data);
    //static void SysStatusCallbackFunc(const char* a_Data);
    static void ToolbarStatusCallbackFunc(const char* a_Data);

    static void OperationErrorCallbackFunc(const char* a_Data);
    static void OperationCompleteCallbackFunc();
    static void OperationProgressCallbackFunc(int a_Progress);

    void AddConnectStatusEntry(const std::string &a_Status);
    //void UpdateSysConfigItems();
    //void AddSysStatusEntry(const std::string &a_Status);  // Should be on another class

signals:
    void connected();

private slots:
    void on_connectButton_clicked();
    void on_disconnectButton_clicked();

private:
    Ui::ConnectDialog *ui;
    static bool m_Connected;

protected:
    //liu added
    // this event is called, when a new translator is loaded or the system language is changed
    void changeEvent(QEvent*); //liu added

};

#endif // CONNECTDIALOG_H
