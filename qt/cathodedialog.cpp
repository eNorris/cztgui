#include "cathodedialog.h"
#include "ui_cathodedialog.h"

CathodeDialog::CathodeDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::CathodeDialog)
{
    ui->setupUi(this);

    for(int i = 0; i < 2; i++)
    {
        ui->cathodeChannelNoCombo->addItem(QString("%2").arg((i + 1)));
    }
    UpdateHGAffectedWidgets(false);

    ui->individualCathodeRadio->setChecked(true);


}

CathodeDialog::~CathodeDialog()
{
    delete ui;
}

void CathodeDialog::UpdateASICCathodeItems()
{
    // could stick top two in a function or use for loop
    CathodeChannel::InternalLeakageCurrentGenerator l_CathChnl1InternalLeak = CathodeChannel::InternalLeakageCurrentGenerator_Undefined;

    if(SpectDMDll::GetCathodeChannelInternalLeakageCurrentGenerator(1, &l_CathChnl1InternalLeak))
    {
        switch(l_CathChnl1InternalLeak)
        {
            case CathodeChannel::InternalLeakageCurrentGenerator_350pA:
            {
                ui->cathChnl1InternalLeak_350p_radio->setChecked(true);
                break;
            }
            case CathodeChannel::InternalLeakageCurrentGenerator_2nA:
            {
                ui->cathChnl1InternalLeak_2nA_radio->setChecked(true);
                break;
            }
            default:
            {
                break;
            }
        }
    }
    else
    {
        QMessageBox::warning(this, "ASIC cathode error", SpectDMDll::GetLastError().c_str());
    }

    CathodeChannel::InternalLeakageCurrentGenerator l_CathChnl2InternalLeak = CathodeChannel::InternalLeakageCurrentGenerator_Undefined;

    if(SpectDMDll::GetCathodeChannelInternalLeakageCurrentGenerator(2, &l_CathChnl2InternalLeak))
    {
        switch(l_CathChnl2InternalLeak)
        {
            case CathodeChannel::InternalLeakageCurrentGenerator_350pA:
            {
                ui->cathChnl2InternalLeak_350p_radio->setChecked(true);
                break;
            }
            case CathodeChannel::InternalLeakageCurrentGenerator_2nA:
            {
                ui->cathChnl2InternalLeak_2nA_radio->setChecked(true);
                break;
            }
            default:
            {
                break;
            }
        }
    }
    else
    {
        QMessageBox::warning(this, "ASIC cathode error", SpectDMDll::GetLastError().c_str());
    }

    TestModeInput l_TestModeInput = TestModeInput_Undefined;

    if(SpectDMDll::GetCathodeTestModeInput(&l_TestModeInput))
    {
        switch(l_TestModeInput)
        {
            case TestModeInput_Step:
            {
                ui->cathTestMode_stepradio->setChecked(true);
                break;
            }
            case TestModeInput_Ramp:
            {
                ui->cathTestMode_rampradio->setChecked(true);
                break;
            }
            default:
            {
                break;
            }
        }
    }
    else
    {
        QMessageBox::warning(this, "ASIC cathode error", SpectDMDll::GetLastError().c_str());
    }

    TimingChannelsShaperPeakingTime l_ShaperPeakingTime = TimingChannelsShaperPeakingTime_Undefined;

    if(SpectDMDll::GetCathodeTimingChannelsShaperPeakingTime(&l_ShaperPeakingTime))
    {
        switch(l_ShaperPeakingTime)
        {
            case TimingChannelsShaperPeakingTime_100nS:
            {
                ui->shaperPeakingTime_100_radio->setChecked(true);
                break;
            }
            case TimingChannelsShaperPeakingTime_200nS:
            {
                ui->shaperPeakingTime_200_radio->setChecked(true);
                break;
            }
            case TimingChannelsShaperPeakingTime_400nS:
            {
                ui->shaperPeakingTime_400_radio->setChecked(true);
                break;
            }
            case TimingChannelsShaperPeakingTime_800nS:
            {
                ui->shaperPeakingTime_800_radio->setChecked(true);
                break;
            }
            default:
            {
                break;
            }
        }
    }
    else
    {
        QMessageBox::warning(this, "ASIC cathode error", SpectDMDll::GetLastError().c_str());
    }

    CathodeTestClockType l_TestClockType = CathodeTestClockType_Undefined;

    if(SpectDMDll::GetCathodeTestClockType(&l_TestClockType))
    {
        switch(l_TestClockType)
        {
            case CathodeTestClockType_ArrivesOnSDI_NSDI:
            {
                ui->cathTestClock_SDI->setChecked(true);
                break;
            }
            case CathodeTestClockType_CopyAnodeTestClock:
            {
                ui->cathTestClock_AnodeCopy->setChecked(true);
                break;
            }
            default:
            {
                break;
            }
        }
    }
    else
    {
        QMessageBox::warning(this, "ASIC cathode error", SpectDMDll::GetLastError().c_str());
    }

    int l_CathMultiThreshDisplaceStep = 0;
    if(SpectDMDll::GetCathodeTimingChannelsSecondaryMultiThresholdsDisplacementStep(&l_CathMultiThreshDisplaceStep))
    {
        ui->secondaryMultiThreshDispSpinner->setValue(l_CathMultiThreshDisplaceStep);
    }
    else
    {
        QMessageBox::warning(this, "ASIC cathode error", SpectDMDll::GetLastError().c_str());
    }

    int l_PeakDetectTimeout = 0;
    if(SpectDMDll::GetPeakDetectTimeout(ChannelType_Cathode, &l_PeakDetectTimeout))
    {
        ui->cathodePeakDetectTimeoutCombo->setCurrentText(QString("%1").arg(l_PeakDetectTimeout));
    }
    else
    {
        QMessageBox::warning(this, "ASIC cathode error", SpectDMDll::GetLastError().c_str());
    }

    int l_TimeDetectRampLength = 0;
    if(SpectDMDll::GetTimeDetectRampLength(ChannelType_Cathode, &l_TimeDetectRampLength))
    {
        ui->cathodeTimeDetectRampLengthCombo->setCurrentText(QString("%1").arg(l_TimeDetectRampLength));
    }
    else
    {
        QMessageBox::warning(this, "ASIC cathode error", SpectDMDll::GetLastError().c_str());
    }

    float l_PeakingTime = 0.0f;
    if(SpectDMDll::GetPeakingTime(ChannelType_Cathode, &l_PeakingTime))
    {
        ui->cathodePeakingTimeCombo->setCurrentText(QString("%1").arg(l_PeakingTime));
    }
    else
    {
        QMessageBox::warning(this, "ASIC cathode error", SpectDMDll::GetLastError().c_str());
    }

    int l_TestPulseStep = 0;
    if(SpectDMDll::GetTestPulseStep(ChannelType_Cathode, &l_TestPulseStep))
    {
        ui->cathodeTestPulseSpinner->setValue(l_TestPulseStep);
    }
    else
    {
        QMessageBox::warning(this, "ASIC cathode error", SpectDMDll::GetLastError().c_str());
    }

    int l_CathEnergyThreshStep = 0;
    if(SpectDMDll::GetChannelThresholdStep(ChannelThresholdType_CathodeEnergy, &l_CathEnergyThreshStep))
    {
        ui->cathodeEnergyThreshSpinner->setValue(l_CathEnergyThreshStep);
    }
    else
    {
        QMessageBox::warning(this, "ASIC cathode error", SpectDMDll::GetLastError().c_str());
    }

    int l_CathPrimaryMultiStep = 0;
    if(SpectDMDll::GetChannelThresholdStep(ChannelThresholdType_CathodeTimingPrimaryMultiThresholdBiPolar, &l_CathPrimaryMultiStep))
    {
        ui->cathodeTimingPrimaryThreshSpinner->setValue(l_CathPrimaryMultiStep);
    }
    else
    {
        QMessageBox::warning(this, "ASIC cathode error", SpectDMDll::GetLastError().c_str());
    }

    int l_CathUnipolarThreshStep = 0;
    if(SpectDMDll::GetChannelThresholdStep(ChannelThresholdType_CathodeTimingUnipolar, &l_CathUnipolarThreshStep))
    {
        ui->cathodeTimingUnipolarThreshSpinner->setValue(l_CathUnipolarThreshStep);
    }
    else
    {
        QMessageBox::warning(this, "ASIC cathode error", SpectDMDll::GetLastError().c_str());
    }
}

void CathodeDialog::UpdateHGAffectedWidgets(bool a_HGSet)
{
    // first and foremost, empty affected combo boxes
    //ui->anodePeakDetectTimoutCombo->clear();
    ui->cathodePeakDetectTimeoutCombo->clear();
    //ui->anodeTimeDetectRampLengthCombo->clear();
    ui->cathodeTimeDetectRampLengthCombo->clear();
    //ui->anodePeakingTimeCombo->clear();
    ui->cathodePeakingTimeCombo->clear();

    // setup peak detect timeout combos on anode and cathode
    for(int i = 0; i < 4; i++)
    {
        int l_Value = a_HGSet ? peakDetectTimoutVals[i] * peakDetectTimeoutValHGFactor : peakDetectTimoutVals[i];
        QString l_ValStr = QString("%1").arg(l_Value);
        //ui->anodePeakDetectTimoutCombo->addItem(l_ValStr);
        ui->cathodePeakDetectTimeoutCombo->addItem(l_ValStr);
    }

    // setup time detect ramp length combos
    for(int i = 0; i < 4; i++)
    {
        int l_Value = a_HGSet ? timeDetectRampLengthVals[i] * timeDetectRampLengthValHGFactor : timeDetectRampLengthVals[i];
        QString l_ValStr = QString("%1").arg(l_Value);
        //ui->anodeTimeDetectRampLengthCombo->addItem(l_ValStr);
        ui->cathodeTimeDetectRampLengthCombo->addItem(l_ValStr);
    }

    // setup peaking time
    for(int i = 0; i < 4; i++)
    {
        double l_Value = a_HGSet ? peakingTimeVals[i] * peakingTimeValHGFactor : peakingTimeVals[i];
        QString l_ValStr = QString("%1").arg(l_Value);
        //ui->anodePeakingTimeCombo->addItem(l_ValStr);
        ui->cathodePeakingTimeCombo->addItem(l_ValStr);
    }
}

void CathodeDialog::on_cathode_UpdateASICButton_clicked()
{
    bool l_Success = true;

    if(ui->ASICCathChnl1InternalLeakageBtnGroup->checkedButton())
    {
        CathodeChannel::InternalLeakageCurrentGenerator l_CathInternalLeak = CathodeChannel::InternalLeakageCurrentGenerator_Undefined;

        if(ui->cathChnl1InternalLeak_350p_radio->isChecked())
        {
            l_CathInternalLeak = CathodeChannel::InternalLeakageCurrentGenerator_350pA;
        }
        else if(ui->cathChnl1InternalLeak_2nA_radio->isChecked())
        {
            l_CathInternalLeak = CathodeChannel::InternalLeakageCurrentGenerator_2nA;
        }

        if(!SpectDMDll::SetCathodeChannelInternalLeakageCurrentGenerator(1, l_CathInternalLeak))
        {
            l_Success = false;
        }
    }

    if(l_Success)
    {
        if(ui->ASICCathChnl2InternalLeakageBtnGroup->checkedButton())
        {
            CathodeChannel::InternalLeakageCurrentGenerator l_CathInternalLeak = CathodeChannel::InternalLeakageCurrentGenerator_Undefined;

            if(ui->cathChnl2InternalLeak_350p_radio->isChecked())
            {
                l_CathInternalLeak = CathodeChannel::InternalLeakageCurrentGenerator_350pA;
            }
            else if(ui->cathChnl2InternalLeak_2nA_radio->isChecked())
            {
                l_CathInternalLeak = CathodeChannel::InternalLeakageCurrentGenerator_2nA;
            }

            if(!SpectDMDll::SetCathodeChannelInternalLeakageCurrentGenerator(2, l_CathInternalLeak))
            {
                l_Success = false;
            }
        }
    }

    if(l_Success)
    {
        if(ui->ASICCathTestModeBtnGroup->checkedButton())
        {
            TestModeInput l_TestMode = TestModeInput_Undefined;

            if(ui->cathTestMode_stepradio->isChecked())
            {
                l_TestMode = TestModeInput_Step;
            }
            else if(ui->cathTestMode_rampradio->isChecked())
            {
                l_TestMode = TestModeInput_Ramp;
            }

            if(!SpectDMDll::SetCathodeTestModeInput(l_TestMode))
            {
                l_Success = false;
            }
        }
    }

    if(l_Success)
    {
        if(ui->ASICCathShaperPeakingBtnGroup->checkedButton())
        {
            TimingChannelsShaperPeakingTime l_ShaperPeakingTime = TimingChannelsShaperPeakingTime_Undefined;

            if(ui->shaperPeakingTime_100_radio->isChecked())
            {
                l_ShaperPeakingTime = TimingChannelsShaperPeakingTime_100nS;
            }
            else if(ui->shaperPeakingTime_200_radio->isChecked())
            {
                l_ShaperPeakingTime = TimingChannelsShaperPeakingTime_200nS;
            }
            else if(ui->shaperPeakingTime_400_radio->isChecked())
            {
                l_ShaperPeakingTime = TimingChannelsShaperPeakingTime_400nS;
            }
            else if(ui->shaperPeakingTime_800_radio->isChecked())
            {
                l_ShaperPeakingTime = TimingChannelsShaperPeakingTime_800nS;
            }

            if(!SpectDMDll::SetCathodeTimingChannelsShaperPeakingTime(l_ShaperPeakingTime))
            {
                l_Success = false;
            }
        }
    }

    if(l_Success)
    {
        if(!SpectDMDll::SetCathodeTimingChannelsSecondaryMultiThresholdsDisplacementStep(ui->secondaryMultiThreshDispSpinner->value()))
        {
            l_Success = false;
        }
    }

    if(l_Success)
    {
        if(ui->ASICCathodeTestClockTypeBtnGroup->checkedButton())
        {
            CathodeTestClockType l_TestClockType = CathodeTestClockType_Undefined;

            if(ui->cathTestClock_SDI->isChecked())
            {
                l_TestClockType = CathodeTestClockType_ArrivesOnSDI_NSDI;
            }
            else if(ui->cathTestClock_AnodeCopy->isChecked())
            {
                l_TestClockType = CathodeTestClockType_CopyAnodeTestClock;
            }

            if(!SpectDMDll::SetCathodeTestClockType(l_TestClockType))
            {
                l_Success = false;
            }
        }
    }

    if(l_Success)
    {
        if(!SpectDMDll::SetPeakDetectTimeout(ChannelType_Cathode, ui->cathodePeakDetectTimeoutCombo->currentText().toInt()))
        {
            l_Success = false;
        }
    }

    if(l_Success)
    {
        if(!SpectDMDll::SetTimeDetectRampLength(ChannelType_Cathode, ui->cathodeTimeDetectRampLengthCombo->currentText().toInt()))
        {
            l_Success = false;
        }
    }

    if(l_Success)
    {
        if(!SpectDMDll::SetPeakingTime(ChannelType_Cathode, ui->cathodePeakingTimeCombo->currentText().toDouble()))
        {
            l_Success = false;
        }
    }

    if(l_Success)
    {
        if(!SpectDMDll::SetTestPulseStep(ChannelType_Cathode, ui->cathodeTestPulseSpinner->value()))
        {
            l_Success = false;
        }
    }

    // thresholds

    if(l_Success)
    {
        if(!SpectDMDll::SetChannelThresholdStep(ChannelThresholdType_CathodeEnergy, ui->cathodeEnergyThreshSpinner->value()))
        {
            l_Success = false;
        }
    }

    if(l_Success)
    {
        if(!SpectDMDll::SetChannelThresholdStep(ChannelThresholdType_CathodeTimingPrimaryMultiThresholdBiPolar
                                          , ui->cathodeTimingPrimaryThreshSpinner->value()))
        {
            l_Success = false;
        }
    }

    if(l_Success)
    {
        if(!SpectDMDll::SetChannelThresholdStep(ChannelThresholdType_CathodeTimingUnipolar
                                          , ui->cathodeTimingUnipolarThreshSpinner->value()))
        {
            l_Success = false;
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

void CathodeDialog::on_updateCathodeChannelButton_clicked()
{
    bool l_Success = true;

    SpectDMDll::SetActiveChannelType(ChannelType_Cathode);

    if(ui->allCathodesRadio->isChecked())
    {
        SpectDMDll::SetChannelUpdateType(ChannelUpdateType_Broadcast);
    }
    else
    {
        // individual channel selected.
        SpectDMDll::SetChannelUpdateType(ChannelUpdateType_SingleChannel);
        SpectDMDll::SetActiveChannel(ui->cathodeChannelNoCombo->currentText().toInt());
    }

    if(!SpectDMDll::MaskChannel(ui->cathodeChannelMaskCheck->isChecked()))
    {
        l_Success = false;
    }

    if(l_Success)
    {
        if(!SpectDMDll::EnableChannelTestCapacitor(ui->cathodeEnableTestCapacitorCheck->isChecked()))
        {
            l_Success = false;
        }
    }

    if(l_Success)
    {
        if(ui->ASICCathChannelShapedSignalBtnGroup->checkedButton())
        {
            CathodeChannel::ShapedTimingSignal l_ShapedSignal = CathodeChannel::ShapedTimingSignal_Undefined;

            if(ui->cathodeShapedTimingSignalUnipolar_radio->isChecked())
            {
                l_ShapedSignal = CathodeChannel::ShapedTimingSignal_Unipolar;
            }
            else if(ui->shapedTimingSignalBipolar_radio->isChecked())
            {
                l_ShapedSignal = CathodeChannel::ShapedTimingSignal_Bipolar;
            }

            if(!SpectDMDll::SetCathodeChannelShapedTimingSignal(l_ShapedSignal))
            {
                l_Success = false;
            }
        }
    }

    if(l_Success)
    {
        if(!SpectDMDll::SetChannelPositivePulseThresholdTrimStep(ui->cathodeChnlPosTrimSpinner->value()))
        {
            l_Success = false;
        }
    }

    if(l_Success)
    {
        if(!SpectDMDll::SetCathodeTimingChannelTrimStep(CathodeTimingChannelType_Unipolar
                                                      , ui->cathodeChnlUnipolarTrimSpinner->value()))
        {
            l_Success = false;
        }
    }

    if(l_Success)
    {
        if(!SpectDMDll::SetCathodeTimingChannelTrimStep(CathodeTimingChannelType_FirstMultiThresholdBiPolar
                                                      , ui->cathodeChnlFirstMultiTrimSpinner->value()))
        {
            l_Success = false;
        }
    }

    if(l_Success)
    {
        if(!SpectDMDll::SetCathodeTimingChannelTrimStep(CathodeTimingChannelType_SecondMultiThreshold
                                                      , ui->cathodeChnlSecondMultiTrimSpinner->value()))
        {
            l_Success = false;
        }
    }

    if(l_Success)
    {
        if(!SpectDMDll::SetCathodeTimingChannelTrimStep(CathodeTimingChannelType_ThirdMultiThreshold
                                                  , ui->cathodeChnlThirdMultiTrimSpinner->value()))
        {
            l_Success = false;
        }
    }

    if(l_Success)
    {
        if(ui->ASICCathodeTimingModeBtnGroup->checkedButton())
        {
            CathodeChannel::TimingMode l_CathTimingMode = CathodeChannel::TimingMode_Undefined;

            if(ui->cathodeTimingModeUnipolar_radio->isChecked())
            {
                l_CathTimingMode = CathodeChannel::TimingMode_Unipolar;
            }
            else if(ui->cathodeTimingModeMultiUnipolar_radio->isChecked())
            {
                l_CathTimingMode = CathodeChannel::TimingMode_MultiThreshold_Unipolar;
            }
            else if(ui->cathodeChnlTimingModeBiPolarUniPolar_radio->isChecked())
            {
                l_CathTimingMode = CathodeChannel::TimingMode_BiPolar_Unipolar;
            }

            if(!SpectDMDll::SetCathodeChannelTimingMode(l_CathTimingMode))
            {
                l_Success = false;
            }
        }
    }

    // If camera update mode is auto then functions above would have sent data down to device
    if(l_Success && SpectDMDll::GetCameraUpdateMode() == CameraUpdateMode_Manual)
    {
        l_Success = SpectDMDll::SendASICChannelData();
    }

    if(!l_Success)
    {
        QMessageBox::warning(this, "SpectDM", SpectDMDll::GetLastError().c_str());
    }
}
