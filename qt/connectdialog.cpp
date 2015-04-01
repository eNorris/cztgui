#include "connectdialog.h"
#include "ui_connectdialog.h"

ConnectDialog::ConnectDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::ConnectDialog)
{
    ui->setupUi(this);
}

ConnectDialog::~ConnectDialog()
{
    delete ui;
}

void ConnectDialog::ConnectStatusCallbackFunc(
    const char* a_Data
)
{
    //GetInstance()->AddConnectStatusEntry(a_Data);
}

void ConnectDialog::SysStatusCallbackFunc(
    const char *a_Data
)
{
    //GetInstance()->AddSysStatusEntry(a_Data);
}

void ConnectDialog::ToolbarStatusCallbackFunc(
    const char *a_Data
)
{
    //GetInstance()->ui->statusBar->showMessage(a_Data);
}

void ConnectDialog::OperationErrorCallbackFunc(
    const char* a_Data
)
{
    //GetInstance()->CloseProgressDlg();
    //GetInstance()->ReportWarning(a_Data);
}

void ConnectDialog::OperationCompleteCallbackFunc()
{
    //GetInstance()->CloseProgressDlg();
}

void ConnectDialog::OperationProgressCallbackFunc(
    int a_Progress
)
{
    //emit GetInstance()->progressUpdate(a_Progress);
}

ConnectDialog *ConnectDialog::GetInstance()
{
    /*
    if(!m_pInstance)
    {
        m_pInstance = new ConnectDialog();
    }

    return m_pInstance;
    */
    return 0;
}

void ConnectDialog::DeleteInstance()
{
    /*
    if(m_pInstance)
    {
        delete m_pInstance;
        m_pInstance = 0;
    }
    */
}
