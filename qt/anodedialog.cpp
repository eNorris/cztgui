#include "anodedialog.h"
#include "ui_anodedialog.h"

AnodeDialog::AnodeDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::AnodeDialog)
{
    ui->setupUi(this);
}

AnodeDialog::~AnodeDialog()
{
    delete ui;
}

void AnodeDialog::UpdateASICAnodeItems()
{
    TestPulseEdge l_TestPulseEdge = TestPulseEdge_Undefined;

    if(SpectDMDll::GetAnodeTestPulseEdge(&l_TestPulseEdge))
    {
        switch(l_TestPulseEdge)
        {
            case TestPulseEdge_InjectNegCharge:
            {
                ui->anodeEdge_negCharge_radio->setChecked(true);
                break;
            }
            case TestPulseEdge_InjectPosAndNegCharge:
            {
                ui->anodeTestPulse_PosAndNeg_radio->setChecked(true);
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
        QMessageBox::warning(this, "ASIC anode error", SpectDMDll::GetLastError().c_str());
    }

    int l_PeakDetectTimeout = 0;
    if(SpectDMDll::GetPeakDetectTimeout(ChannelType_Anode, &l_PeakDetectTimeout))
    {
        ui->anodePeakDetectTimoutCombo->setCurrentText(QString("%1").arg(l_PeakDetectTimeout));
    }
    else
    {
        QMessageBox::warning(this, "ASIC anode error", SpectDMDll::GetLastError().c_str());
    }

    int l_TimeDetectRampLength = 0;
    if(SpectDMDll::GetTimeDetectRampLength(ChannelType_Anode, &l_TimeDetectRampLength))
    {
        ui->anodeTimeDetectRampLengthCombo->setCurrentText(QString("%1").arg(l_TimeDetectRampLength));
    }
    else
    {
        QMessageBox::warning(this, "ASIC anode error", SpectDMDll::GetLastError().c_str());
    }

    float l_PeakingTime = 0.0f;
    if(SpectDMDll::GetPeakingTime(ChannelType_Anode, &l_PeakingTime))
    {
        ui->anodePeakingTimeCombo->setCurrentText(QString("%1").arg(l_PeakingTime));
    }
    else
    {
        QMessageBox::warning(this, "ASIC anode error", SpectDMDll::GetLastError().c_str());
    }

    int l_TestPulseStep = 0;
    if(SpectDMDll::GetTestPulseStep(ChannelType_Anode, &l_TestPulseStep))
    {
        ui->anodeTestPulseSpinner->setValue(l_TestPulseStep);
    }
    else
    {
        QMessageBox::warning(this, "ASIC anode error", SpectDMDll::GetLastError().c_str());
    }

    int l_AnodePosEnergyThreshStep = 0;
    if(SpectDMDll::GetChannelThresholdStep(ChannelThresholdType_AnodePositiveEnergy, &l_AnodePosEnergyThreshStep))
    {
        ui->anodePosEnergyThreshSpinner->setValue(l_AnodePosEnergyThreshStep);
    }
    else
    {
        QMessageBox::warning(this, "ASIC anode error", SpectDMDll::GetLastError().c_str());
    }

    int l_AnodeNegEnergyThreshStep = 0;
    if(SpectDMDll::GetChannelThresholdStep(ChannelThresholdType_AnodeNegativeEnergy, &l_AnodeNegEnergyThreshStep))
    {
        ui->anodeNegEnergyThreshSpinner->setValue(l_AnodeNegEnergyThreshStep);
    }
    else
    {
        QMessageBox::warning(this, "ASIC anode error", SpectDMDll::GetLastError().c_str());
    }
}

void AnodeDialog::UpdateHGAffectedWidgets(bool a_HGSet)
{
    // first and foremost, empty affected combo boxes
    ui->anodePeakDetectTimoutCombo->clear();
    //ui->cathodePeakDetectTimeoutCombo->clear();
    ui->anodeTimeDetectRampLengthCombo->clear();
    //ui->cathodeTimeDetectRampLengthCombo->clear();
    ui->anodePeakingTimeCombo->clear();
    //ui->cathodePeakingTimeCombo->clear();

    // setup peak detect timeout combos on anode and cathode
    for(int i = 0; i < 4; i++)
    {
        int l_Value = a_HGSet ? peakDetectTimoutVals[i] * peakDetectTimeoutValHGFactor : peakDetectTimoutVals[i];
        QString l_ValStr = QString("%1").arg(l_Value);
        ui->anodePeakDetectTimoutCombo->addItem(l_ValStr);
        //ui->cathodePeakDetectTimeoutCombo->addItem(l_ValStr);
    }

    // setup time detect ramp length combos
    for(int i = 0; i < 4; i++)
    {
        int l_Value = a_HGSet ? timeDetectRampLengthVals[i] * timeDetectRampLengthValHGFactor : timeDetectRampLengthVals[i];
        QString l_ValStr = QString("%1").arg(l_Value);
        ui->anodeTimeDetectRampLengthCombo->addItem(l_ValStr);
        //ui->cathodeTimeDetectRampLengthCombo->addItem(l_ValStr);
    }

    // setup peaking time
    for(int i = 0; i < 4; i++)
    {
        double l_Value = a_HGSet ? peakingTimeVals[i] * peakingTimeValHGFactor : peakingTimeVals[i];
        QString l_ValStr = QString("%1").arg(l_Value);
        ui->anodePeakingTimeCombo->addItem(l_ValStr);
        //ui->cathodePeakingTimeCombo->addItem(l_ValStr);
    }
}
