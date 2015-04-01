#ifndef CONNECTDIALOG_H
#define CONNECTDIALOG_H

#include <QDialog>

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
    static void DeleteInstance();

    static void ConnectStatusCallbackFunc(const char* a_Data);
    static void SysStatusCallbackFunc(const char* a_Data);
    static void ToolbarStatusCallbackFunc(const char* a_Data);

    static void OperationErrorCallbackFunc(const char* a_Data);
    static void OperationCompleteCallbackFunc();
    static void OperationProgressCallbackFunc(int a_Progress);

private:
    Ui::ConnectDialog *ui;
};

#endif // CONNECTDIALOG_H
