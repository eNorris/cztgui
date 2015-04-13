#include "fpgadialog.h"
#include "ui_fpgadialog.h"

FpgaDialog::FpgaDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::FpgaDialog)
{
    ui->setupUi(this);
}

FpgaDialog::~FpgaDialog()
{
    delete ui;
}

void FpgaDialog::on_UpdateGMButton_clicked()
{
    bool l_Success = true;

    if(l_Success)
    {
        GMOption l_GMOptions = 0;

        if(ui->GMDisPhotonCollectCheck->isChecked())
        {
            l_GMOptions |= GMOption_DisablePhotonCollect;
        }

        if(ui->GMDebugCheck->isChecked())
        {
            l_GMOptions |= GMOption_DebugMode;
        }

        if(ui->GMChannel1TestModeCheck->isChecked())
        {
            l_GMOptions |= GMOption_Channel1TestMode;
        }

        if(ui->GMPixelMapCheck->isChecked())
        {
            l_GMOptions |= GMOption_EnablePixelMap;
        }

        if(!SpectDMDll::SetGMOptions(l_GMOptions))
        {
            l_Success = false;
        }
    }

    if(l_Success)
    {
        if(!SpectDMDll::SetDelayTime(ui->delayTimeSpinner->value()))
        {
            l_Success = false;
        }
    }

    if(l_Success)
    {
        if(!SpectDMDll::SetTimestampResolution(ui->timestampResSpinner->value()))
        {
            l_Success = false;
        }
    }

    if(l_Success)
    {
        if(!SpectDMDll::SetGMADC1Channel(ui->ADC1ChannelCombo->currentText().toInt()))
        {
            l_Success = false;
        }
    }

    if(l_Success)
    {
        if(!SpectDMDll::SetGMADC2Channel(ui->ADC2ChannelCombo->currentText().toInt()))
        {
            l_Success = false;
        }
    }

    if(l_Success)
    {
        QString l_DefaultGMStr = ui->defaultGMCombo->currentText();
        if(l_DefaultGMStr != "-")
        {
            if(!SpectDMDll::SetDefaultGM(l_DefaultGMStr.toInt()))
            {
                l_Success = false;
            }
        }
    }

    if(l_Success)
    {
        // GM Readout options test
        GMReadoutOption l_ReadoutOpt = 0;

        if(ui->negEnergyCheck->isChecked())
        {
            l_ReadoutOpt |= GMReadoutOpt_NegativeEnergy;
        }

        if(ui->timeDetectCheck->isChecked())
        {
            l_ReadoutOpt |= GMReadoutOpt_TimeDetect;
        }

        if(!SpectDMDll::SetGMReadoutOptions(l_ReadoutOpt))
        {
            l_Success = false;
        }
    }

    if(l_Success)
    {
        // GM ReadoutMode test
        if(ui->GMReadoutModeBtnGroup->checkedButton() != 0)
        {
            GMReadoutMode l_Mode = GMReadout_Undefined;
            // a button is checked
            if(ui->readAllRadio->isChecked())
            {
                l_Mode = GMReadout_ReadAll;
            }
            else if(ui->sparsifiedRadio->isChecked())
            {
                l_Mode = GMReadout_SparsifiedMode;
            }

            if(!SpectDMDll::SetGMReadoutMode(l_Mode))
            {
                l_Success = false;
            }
        }
    }

    if(l_Success)
    {
        // GM Cathode mode test
        if(ui->GMCathodeModeBtnGroup->checkedButton() != 0)
        {
            GMCathodeMode l_CathMode = GMCathMode_Undefined;

            if(ui->unipolarRadio->isChecked())
            {
                l_CathMode = GMCathMode_Unipolar;
            }
            else if(ui->multiThreshRadio->isChecked())
            {
                l_CathMode = GMCathMode_MultiThreshold;
            }
            else if(ui->bipolarRadio->isChecked())
            {
                l_CathMode = GMCathMode_Bipolar;
            }

            if(!SpectDMDll::SetGMCathodeMode(l_CathMode))
            {
                l_Success = false;
            }
        }
    }

    if(l_Success)
    {
        // pulser frequency test
        if(ui->GMPulserFreqBtnGroup->checkedButton())
        {
            GMPulserFrequency l_Freq = GMPulserFreq_Undefined;

            if(ui->hundredHzRadio->isChecked())
            {
                l_Freq = GMPulserFreq_100Hz;
            }
            else if(ui->oneKHzRadio->isChecked())
            {
                l_Freq = GMPulserFreq_1kHz;
            }
            else if(ui->tenKHz->isChecked())
            {
                l_Freq = GMPulserFreq_10kHz;
            }
            else if(ui->hundredKHz->isChecked())
            {
                l_Freq = GMPulserFreq_100kHz;
            }

            if(!SpectDMDll::SetGMPulserFrequency(l_Freq))
            {
                l_Success = false;
            }
        }
    }

    if(l_Success)
    {
        // pulser options
        GMPulserOption l_PulserOpt = 0;

        if(ui->anodePulserCheck->isChecked())
        {
            l_PulserOpt |= GMPulserOpt_Anode;
        }

        if(ui->cathodePulserCheck->isChecked())
        {
            l_PulserOpt |= GMPulserOpt_Cathode;
        }

        if(!SpectDMDll::SetGMPulserOptions(l_PulserOpt))
        {
            l_Success = false;
        }
    }

    if(l_Success)
    {
        QString l_NoOfPulsesTxt = ui->noOfPulsesCombo->currentText();

        if(!l_NoOfPulsesTxt.isEmpty())
        {
            if(!SpectDMDll::SetNoOfPulses(l_NoOfPulsesTxt.toInt()))
            {
                l_Success = false;
            }
        }
    }

    // If camera update mode is auto then functions above would have sent data down to device
    if(l_Success && SpectDMDll::GetCameraUpdateMode() == CameraUpdateMode_Manual)
    {
        l_Success = SpectDMDll::SendGMData();
    }

    if(!l_Success)
    {
        QMessageBox::warning(this, "SpectDM", SpectDMDll::GetLastError().c_str());
    }
}
