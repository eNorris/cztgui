#include "cathodedialog.h"
#include "ui_cathodedialog.h"

CathodeDialog::CathodeDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::CathodeDialog)
{
    ui->setupUi(this);
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
