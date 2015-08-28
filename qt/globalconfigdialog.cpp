#include "globalconfigdialog.h"
#include "ui_globalconfigdialog.h"

GlobalConfigDialog::GlobalConfigDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::GlobalConfigDialog)
{
    ui->setupUi(this);

    for(int i = 0; i < 128; i++)
    {
        //ui->anodeChannelNoCombo->addItem(QString("%1").arg(i + 1));
        ui->anodeChannelMonitorCombo->addItem(QString("%1").arg(i + 1));
        //ui->anodeChannelNoCombo->addItem(QString("%1").arg(i + 1));
        //ui->anodeChannelMonitorCombo->addItem(QString("%1").arg(i + 1));

    }

    ui->DACScombo->setEnabled(false);
    ui->anodeChannelMonitorCombo->setEnabled(false);
    ui->cathEnergyTimingCombo->setEnabled(false);


}

GlobalConfigDialog::~GlobalConfigDialog()
{
    delete ui;
}

void GlobalConfigDialog::loadDefaults()
{
    ui->disableMultiResetAcqModeCheck->setChecked(true);
    ui->singleEventModeCheck->setChecked(true);
    ui->channelGain_60Radio->setChecked(true);
    on_ASICGlobal_UpdateASICButton_clicked();
}

void GlobalConfigDialog::on_ASICGlobal_UpdateASICButton_clicked()
{
    bool l_Success = true;

    ASICGlobalOptions l_GlobalOpt = ASICGlobal_None;

    if(ui->routeMonitorToPinTDOCheck->isChecked())
    {
        l_GlobalOpt |= ASICGlobal_RouteMonitorToPinTDO;
    }

    if(ui->routeTempMonPinAXPK62Check->isChecked())
    {
        l_GlobalOpt |= ASICGlobal_RouteTempMonitorToAXPK62;
    }

    if(ui->disableMultiResetAcqModeCheck->isChecked())
    {
        l_GlobalOpt |= ASICGlobal_DisableMultipleResetAcquisitionMode;
    }

    if(ui->timingMFSCheck->isChecked())
    {
        l_GlobalOpt |= ASICGlobal_TimingMultipleFiringSuppressor;
    }

    if(ui->energyMFSCheck->isChecked())
    {
        l_GlobalOpt |= ASICGlobal_EnergyMultipleFiringSuppressor;
    }

    if(ui->highGainCheck->isChecked())
    {
        l_GlobalOpt |= ASICGlobal_HighGain;
    }

    if(ui->monitorOutputsCheck->isChecked())
    {
        l_GlobalOpt |= ASICGlobal_MonitorOutputs;
    }

    if(ui->singleEventModeCheck->isChecked())
    {
        l_GlobalOpt |= ASICGlobal_SingleEventMode;
    }

    if(ui->validationCheck->isChecked())
    {
        l_GlobalOpt |= ASICGlobal_Validation;
    }

    if(ui->chnl62PreAmpMonitorOutputCheck->isChecked())
    {
        l_GlobalOpt |= ASICGlobal_BufferChnl62PreAmplifierMonitorOutput;
    }

    if(ui->chnl63PreAmpMonitorOutputCheck->isChecked())
    {
        l_GlobalOpt |= ASICGlobal_BufferChnl63PreAmplifierMonitorOutput;
    }

    if(ui->peakAndTimeDetectorOutputCheck->isChecked())
    {
        l_GlobalOpt |= ASICGlobal_BufferPeakAndTimeDetectorOutputs;
    }

    if(ui->auxMonitorOutputCheck->isChecked())
    {
        l_GlobalOpt |= ASICGlobal_BufferAuxMonitorOutput;
    }

    if(SpectDMDll::SetASICGlobalOptions(l_GlobalOpt))
    {
        // update the HG affected widgets before we update ASIC anode and cathode
        // items otherwise value we want to set may not be in combo box.
        UpdateHGAffectedWidgets((l_GlobalOpt & ASICGlobal_HighGain) != 0);

        // User may have ticked / unticked High Gain (HG) and if this is the case
        // then we need to update the HG-affected variables on Anode and Cathode tabs
        // .. but only if an active GM is set, if we are broadcasting, which ones would
        // we use?
        if(SpectDMDll::GetGMUpdateType() == GMUpdateType_SingleGM)
        {
            emit updateAnodeItems();
            emit updateCathodeItems();
            //UpdateASICAnodeItems();
            //UpdateASICCathodeItems();
        }
    }
    else
    {
        l_Success = false;
    }

    if(l_Success)
    {
        if(ui->ASICTimingUnipolarGainBtnGroup->checkedButton())
        {
            TimingChannelUnipolarGain l_UniPolarGain = TimingChannelUnipolarGain_Undefined;

            if(ui->unipolarGain_27radio->isChecked())
            {
                l_UniPolarGain = TimingChannelUnipolarGain_27mV;
            }
            else if(ui->unipolarGain_81radio->isChecked())
            {
                l_UniPolarGain = TimingChannelUnipolarGain_81mV;
            }

            if(!SpectDMDll::SetTimingChannelUnipolarGain(l_UniPolarGain))
            {
                l_Success = false;
            }
        }
    }

    if(l_Success)
    {
        if(ui->ASICTimingBipolarGainBtnGroup->checkedButton())
        {
            TimingChannelBipolarGain l_BiPolarGain = TimingChannelBipolarGain_Undefined;

            if(ui->timingChannelBiPolarGain_21mV_radio->isChecked())
            {
                l_BiPolarGain = TimingChannelBipolarGain_21mV;
            }
            else if(ui->timingChannelBiPolarGain_55mV_radio->isChecked())
            {
                l_BiPolarGain = TimingChannelBipolarGain_55mV;
            }
            else if(ui->timingChannelBiPolarGain_63mV_radio->isChecked())
            {
                l_BiPolarGain = TimingChannelBipolarGain_63mV;
            }
            else if(ui->timingChannelBiPolarGain_164mV_radio->isChecked())
            {
                l_BiPolarGain = TimingChannelBipolarGain_164mV;
            }

            if(!SpectDMDll::SetTimingChannelBiPolarGain(l_BiPolarGain))
            {
                l_Success = false;
            }
        }
    }

    if(l_Success)
    {
        if(ui->ASICReadoutModeBtnGroup->checkedButton())
        {
            GMASICReadoutMode l_ReadoutMode = GMASICReadout_Undefined;

            if(ui->ASICNormalSparsifiedRadio->isChecked())
            {
                l_ReadoutMode = GMASICReadout_NormalSparsified;
            }
            else if(ui->ASICEnhancedSparsifiedRadio->isChecked())
            {
                l_ReadoutMode = GMASICReadout_EnhancedSparsified;
            }

            if(!SpectDMDll::SetASICReadoutMode(l_ReadoutMode))
            {
                l_Success = false;
            }
        }
    }

    if(l_Success)
    {
        if(ui->ASICChannelGainBtnGroup->checkedButton())
        {
            ChannelGain l_ChannelGain = ChannelGain_Undefined;

            if(ui->channelGain_20radio->isChecked())
            {
                l_ChannelGain = ChannelGain_20mV;
            }
            else if(ui->channelGain_60Radio->isChecked())
            {
                l_ChannelGain = ChannelGain_60mV;
            }

            if(!SpectDMDll::SetChannelGain(l_ChannelGain))
            {
                l_Success = false;
            }
        }
    }

    if(l_Success)
    {
        if(ui->ASICInternalLeakageCurrGenBtnGroup->checkedButton())
        {
            InternalLeakageCurrentGenerator l_CurrGen = InternalLeakageCurrentGenerator_Undefined;

            if(ui->internalLeakageCurrGen_0A_radio->isChecked())
            {
                l_CurrGen = InternalLeakageCurrentGenerator_0A;
            }
            else if(ui->internalLeakageCurrGen_60pA_radio->isChecked())
            {
                l_CurrGen = InternalLeakageCurrentGenerator_60pA;
            }

            if(!SpectDMDll::SetInternalLeakageCurrentGenerator(l_CurrGen))
            {
                l_Success = false;
            }
        }
    }

    if(l_Success)
    {
        if(ui->ASICMultipleFireSuppressBtnGroup->checkedButton())
        {
            MultipleFiringSuppressionTime l_SuppressTime = MultipleFiringSuppressionTime_Undefined;

            if(ui->multipleFiringSuppress_62_5_radio->isChecked())
            {
                l_SuppressTime = MultipleFiringSuppressionTime_62_5nS;
            }
            else if(ui->multipleFiringSuppress_125_radio->isChecked())
            {
                l_SuppressTime = MultipleFiringSuppressionTime_125nS;
            }
            else if(ui->multipleFiringSuppress_250_radio->isChecked())
            {
                l_SuppressTime = MultipleFiringSuppressionTime_250nS;
            }
            else if(ui->multipleFiringSuppress_600_radio->isChecked())
            {
                l_SuppressTime = MultipleFiringSuppressionTime_600nS;
            }

            if(!SpectDMDll::SetMultipleFiringSuppressTime(l_SuppressTime))
            {
                l_Success = false;
            }
        }
    }

    // Analog output to monitor setting
    if(l_Success)
    {
        if(ui->noFuncRadio->isChecked())
        {
            l_Success &= SpectDMDll::SetAnalogOutputToMonitor(AnalogOutput_NoFunction);
        }
        else if(ui->baselineRadio->isChecked())
        {
            l_Success &= SpectDMDll::SetAnalogOutputToMonitor(AnalogOutput_Baseline);
        }
        else if(ui->temperatureRadio->isChecked())
        {
            l_Success &= SpectDMDll::SetAnalogOutputToMonitor(AnalogOutput_Temperature);
        }
        else if(ui->DACSradio->isChecked())
        {
            // need to additionally check combo here
            DACS l_DACSelected = static_cast<DACS>(ui->DACScombo->currentIndex() + 1);

            l_Success &= SpectDMDll::SetDACToMonitor(l_DACSelected);
        }
        else if(ui->cathodeEnergyTimingRadio->isChecked())
        {
            // need to additionally check combo here
            CathodeEnergyTiming l_CathEnergyTimingSelected
                = static_cast<CathodeEnergyTiming>(ui->cathEnergyTimingCombo->currentIndex() + 1);

            l_Success &= SpectDMDll::SetCathodeEnergyTimingToMonitor(l_CathEnergyTimingSelected);
        }
        else if(ui->anodeChannelRadio->isChecked())
        {
            // need to additionally check combo here
            l_Success &= SpectDMDll::SetAnodeChannelToMonitor(ui->anodeChannelMonitorCombo->currentText().toInt());
        }
        else
        {
            // error?
        }
    }

    // If camera update mode is auto then functions above would have sent data down to device
    if(l_Success && SpectDMDll::GetCameraUpdateMode() == CameraUpdateMode_Manual)
    {
        l_Success = SpectDMDll::SendASICGlobalData();
    }

    if(!l_Success)
    {
        QMessageBox::warning(this, "SpectDM", SpectDMDll::GetLastError().c_str());
    }
}

void GlobalConfigDialog::UpdateHGAffectedWidgets(bool aHGSet)
{
    emit updateAnodeCathodeHG(aHGSet);
    /*
    // first and foremost, empty affected combo boxes
    ui->anodePeakDetectTimoutCombo->clear();
    ui->cathodePeakDetectTimeoutCombo->clear();
    ui->anodeTimeDetectRampLengthCombo->clear();
    ui->cathodeTimeDetectRampLengthCombo->clear();
    ui->anodePeakingTimeCombo->clear();
    ui->cathodePeakingTimeCombo->clear();

    // setup peak detect timeout combos on anode and cathode
    for(int i = 0; i < 4; i++)
    {
        int l_Value = a_HGSet ? peakDetectTimoutVals[i] * peakDetectTimeoutValHGFactor : peakDetectTimoutVals[i];
        QString l_ValStr = QString("%1").arg(l_Value);
        ui->anodePeakDetectTimoutCombo->addItem(l_ValStr);
        ui->cathodePeakDetectTimeoutCombo->addItem(l_ValStr);
    }

    // setup time detect ramp length combos
    for(int i = 0; i < 4; i++)
    {
        int l_Value = a_HGSet ? timeDetectRampLengthVals[i] * timeDetectRampLengthValHGFactor : timeDetectRampLengthVals[i];
        QString l_ValStr = QString("%1").arg(l_Value);
        ui->anodeTimeDetectRampLengthCombo->addItem(l_ValStr);
        ui->cathodeTimeDetectRampLengthCombo->addItem(l_ValStr);
    }

    // setup peaking time
    for(int i = 0; i < 4; i++)
    {
        double l_Value = a_HGSet ? peakingTimeVals[i] * peakingTimeValHGFactor : peakingTimeVals[i];
        QString l_ValStr = QString("%1").arg(l_Value);
        ui->anodePeakingTimeCombo->addItem(l_ValStr);
        ui->cathodePeakingTimeCombo->addItem(l_ValStr);
    }
    */
}


// liu added
void GlobalConfigDialog::changeEvent(QEvent* event)
{
    if(0 != event)
    {
        switch(event->type())
        {
        // this event is send if a translator is loaded
            case QEvent::LanguageChange:
            ui->retranslateUi(this);
            break;

        default:
            // Do nothing
            break;

        // this event is send, if the system, language changes
        //  case QEvent::LocaleChange:
        //  {
        //    QString locale = QLocale::system().name();
        //    locale.truncate(locale.lastIndexOf('_'));
        //    loadLanguage(locale);
        //  }
        //  break;
        }
    }
    QWidget::changeEvent(event);
}
//liu added end


