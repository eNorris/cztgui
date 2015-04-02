#include "connectdialog.h"
#include "ui_connectdialog.h"

ConnectDialog* ConnectDialog::m_pInstance = NULL;
bool ConnectDialog::m_Connected = false;

ConnectDialog::ConnectDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::ConnectDialog)
{
    ui->setupUi(this);
    m_pInstance = this;

    SpectDMDll::SetConnectStatusFunction(ConnectStatusCallbackFunc);
    //SpectDMDll::SetConnectStatusFunction(ConnectStatusCallbackFunc);
    //SpectDMDll::SetSysStatusFunction(SysStatusCallbackFunc);
    //SpectDMDll::SetToolbarStatusFunction(ToolbarStatusCallbackFunc);

    // setup operation callbacks
    //SpectDMDll::SetOperationErrorFunction(OperationErrorCallbackFunc);
    //SpectDMDll::SetOperationCompleteFunction(OperationCompleteCallbackFunc);
    //SpectDMDll::SetOperationProgressFunction(OperationProgressCallbackFunc);

    connect(ui->clearButton, SIGNAL(clicked()), ui->statusTextEdit, SLOT(clear()));
}

ConnectDialog::~ConnectDialog()
{
    delete ui;
}

void ConnectDialog::ConnectStatusCallbackFunc(const char* a_Data)
{
    GetInstance()->AddConnectStatusEntry(a_Data);
}

// Moved
//void ConnectDialog::SysStatusCallbackFunc(const char *a_Data)
//{
//    GetInstance()->AddSysStatusEntry(a_Data);
//}

void ConnectDialog::ToolbarStatusCallbackFunc(const char *a_Data)
{
    //GetInstance()->ui->statusBar->showMessage(a_Data);
}

void ConnectDialog::OperationErrorCallbackFunc(const char* a_Data)
{
    //GetInstance()->CloseProgressDlg();
    //GetInstance()->ReportWarning(a_Data);
}

void ConnectDialog::OperationCompleteCallbackFunc()
{
    //GetInstance()->CloseProgressDlg();
}

void ConnectDialog::OperationProgressCallbackFunc(int a_Progress)
{
    //emit GetInstance()->progressUpdate(a_Progress);
}

ConnectDialog *ConnectDialog::GetInstance()
{
    //if(!m_pInstance)
    //{
    //    m_pInstance = new ConnectDialog();
    //}

    return m_pInstance;
}

//void ConnectDialog::DeleteInstance()
//{
//    if(m_pInstance)
//    {
//        delete m_pInstance;
//        m_pInstance = NULL;
//    }
//}

void ConnectDialog::AddConnectStatusEntry(const std::string &a_Status)
{
    ui->statusTextEdit->append(createTimestampedStr(a_Status));
}

//void ConnectDialog::AddSysStatusEntry(const std::string &a_Status)
//{
//    ui->sysStatusTextArea->append(createTimestampedStr(a_Status));
//}

void ConnectDialog::on_connectButton_clicked()
{
    if(!ui->hostLineEdit->text().isEmpty() && !ui->cameraLineEdit->text().isEmpty())
    {
        //SpectDMDll::SetHostIPAddress(ui->hostLineEdit->text().toStdString().c_str());
        //SpectDMDll::SetCameraIPAddress(ui->cameraLineEdit->text().toStdString().c_str());
        SpectDMDll::SetHostIPAddress(ui->hostLineEdit->text().toUtf8().constData());
        SpectDMDll::SetCameraIPAddress(ui->cameraLineEdit->text().toUtf8().constData());

        if(SpectDMDll::Initialize())
        {
            m_Connected = true;
            ui->connectButton->setEnabled(false);
            ui->disconnectButton->setEnabled(true);
            //ui->loadFromFileBtn->setEnabled(false);
            //EnableNonConnectTabs(true);
            UpdateSysConfigItems();
        }
        else
        {
            AddConnectStatusEntry(SpectDMDll::GetLastError());
        }
    }
    else
    {
        AddConnectStatusEntry("Connect requires Local IP and Remote IP addresses");
    }
}

void ConnectDialog::on_disconnectButton_clicked()
{
    if(m_Connected)
    {
        SpectDMDll::Disconnect();

        ui->connectButton->setEnabled(true);
        ui->disconnectButton->setEnabled(false);
        //ui->loadFromFileBtn->setEnabled(true);

        //EnableNonConnectTabs(false);
    }
}

void ConnectDialog::UpdateSysConfigItems()
{
    // TODO
    // set mb multiplexer addr type
    //MBMultiplexerAddrType l_MBMultiplexAddrType = SpectDMDll::GetMBMultiplexerAddressType();
    //ui->mbMultiplexerAddrTypeCombo->setCurrentIndex(static_cast<int>(l_MBMultiplexAddrType));

    // TODO - Maybe this should be moved to another class and called as a signal?
    // token type
    //SysTokenType l_TokenType = SpectDMDll::GetSysTokenType();
    //QRadioButton* l_TokenTypeToCheck = 0;
    //
    //switch(l_TokenType)
    //{
    //    case SysTokenType_DaisyChain:
    //        l_TokenTypeToCheck = ui->tokenDaisyChainradio;
    //        break;
    //    case SysTokenType_GM3Only:
    //        l_TokenTypeToCheck = ui->tokenGM3radio;
    //        break;
    //    default:
    //        break;
    //}
    //
    //if(l_TokenTypeToCheck)
    //{
    //    l_TokenTypeToCheck->setChecked(true);
    //}

    /*
    // Pixel Mapping Mode
    PixelMappingMode l_PixelMappingMode = SpectDMDll::GetPixelMappingMode();
    QRadioButton* l_PixelMapModeToCheck = 0;

    switch(l_PixelMappingMode)
    {
        case PixelMappingMode_GMBased:
        {
            l_PixelMapModeToCheck = ui->pixelMapGMBasedRadio;
            break;
        }
        case PixelMappingMode_Global:
        {
            l_PixelMapModeToCheck = ui->pixelMapGlobalRadio;
            break;
        }
        default:
        {
            break;
        }
    }

    if(l_PixelMapModeToCheck)
    {
        l_PixelMapModeToCheck->setChecked(true);
        // update GM enable pixel mapping enable state after updating pixel mapping mode
        UpdateGMPixelMapCheckbox();
    }

    // packet transfer rate
    ui->pktTransferSpinner->setValue(SpectDMDll::GetPacketTransferRate());

    // debug mode
    // need to check if debug mode is set.
    ui->debugModeCheckBox->setChecked(SpectDMDll::IsDebugModeActive());
    */

}




