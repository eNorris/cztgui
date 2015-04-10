#include "systemconfigdialog.h"
#include "ui_systemconfigdialog.h"

SystemConfigDialog* SystemConfigDialog::m_pInstance = NULL;

SystemConfigDialog::SystemConfigDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::SystemConfigDialog)
{
    ui->setupUi(this);

    m_pInstance = this;

    SpectDMDll::SetSysStatusFunction(SysStatusCallbackFunc);
}

SystemConfigDialog::~SystemConfigDialog()
{
    delete ui;
}

void SystemConfigDialog::SysStatusCallbackFunc(const char *a_Data)
{
    GetInstance()->AddSysStatusEntry(a_Data);
}

SystemConfigDialog *SystemConfigDialog::GetInstance()
{
    return m_pInstance;
}

//void SystemConfigDialog::SysStatusCallbackFunc(const char *a_Data)
//{
//    GetInstance()->AddSysStatusEntry(a_Data);
//

void SystemConfigDialog::AddSysStatusEntry(const std::string &a_Status)
{
    ui->sysStatusTextArea->append(createTimestampedStr(a_Status));
}

void SystemConfigDialog::UpdateSysConfigItems()
{
    // TODO
    // set mb multiplexer addr type
    MBMultiplexerAddrType l_MBMultiplexAddrType = SpectDMDll::GetMBMultiplexerAddressType();
    ui->mbMultiplexerAddrTypeCombo->setCurrentIndex(static_cast<int>(l_MBMultiplexAddrType));

    // TODO - Maybe this should be moved to another class and called as a signal?
    // token type
    SysTokenType l_TokenType = SpectDMDll::GetSysTokenType();
    QRadioButton* l_TokenTypeToCheck = 0;

    switch(l_TokenType)
    {
        case SysTokenType_DaisyChain:
            l_TokenTypeToCheck = ui->tokenDaisyChainradio;
            break;
        case SysTokenType_GM3Only:
            l_TokenTypeToCheck = ui->tokenGM3radio;
            break;
        default:
            break;
    }

    if(l_TokenTypeToCheck)
        l_TokenTypeToCheck->setChecked(true);


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
}

void SystemConfigDialog::UpdateGMPixelMapCheckbox()
{
    // TODO - I just commented this out, because I didn't include this in the UI file.
    // this function is called when the pixel map mode is changed on the sys config tab
    // if the user has set GM-based then the GM enable pixel mapping checkbox should be enabled
    // if the user has selected global then the GM enable pixel mapping checkbox should be disabled
    // as its value is ignored here anyhow.
    //ui->GMPixelMapCheck->setEnabled(ui->pixelMapGMBasedRadio->isChecked());
}

void SystemConfigDialog::on_startCollectBtn_clicked()
{
    if(SpectDMDll::StartPhotonCollection())
    {
        ui->stopSysBtn->setEnabled(false);
        ui->startCollectBtn->setEnabled(false);
        ui->stopCollectBtn->setEnabled(true);
    }
    else
    {
        QMessageBox::warning(this, "SpectDM", SpectDMDll::GetLastError().c_str());
    }
}

void SystemConfigDialog::on_stopCollectBtn_clicked()
{
    if(SpectDMDll::StopPhotonCollection())
    {
        ui->stopSysBtn->setEnabled(true);
        ui->stopCollectBtn->setEnabled(false);
        ui->startCollectBtn->setEnabled(true);
    }
    else
    {
        QMessageBox::warning(this, "SpectDM", SpectDMDll::GetLastError().c_str());
    }
}

void SystemConfigDialog::on_startSysBtn_clicked()
{
    qDebug() << "system started";
    if(SpectDMDll::StartSys())
    {
        ui->startSysBtn->setEnabled(false);
        ui->stopSysBtn->setEnabled(true);
        ui->startCollectBtn->setEnabled(true);
    }
    else
    {
        QMessageBox::warning(this, "SpectDM", SpectDMDll::GetLastError().c_str());
    }
}

void SystemConfigDialog::on_stopSysBtn_clicked()
{
    qDebug() << "system stopped";
    if(SpectDMDll::StopSys())
    {
        ui->stopSysBtn->setEnabled(false);
        ui->startSysBtn->setEnabled(true);
        ui->startCollectBtn->setEnabled(false);
    }
    else
    {
        QMessageBox::warning(this, "SpectDM", SpectDMDll::GetLastError().c_str());
    }
}
