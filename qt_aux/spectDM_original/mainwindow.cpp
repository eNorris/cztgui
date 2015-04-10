#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "spectdmdll.h"
#include <QDebug>
#include <QDateTime>
#include <QFileDialog>
#include <QKeyEvent>
#include <QMessageBox>
#include <QScrollArea>

MainWindow* MainWindow::m_pInstance = NULL;

const QString timeStampFormat = "yyyy/MM/dd:hh:mm:ss.zzz";

const int minRegDirectVal = 0;
const int maxRegDirectVal = 255;

// tab index consts
const int connectTabIndex = 0;
const int sysConfigTabIndex = 1;
const int GMConfigTabIndex = 2;
const int ASICTabIndex = 3;
const int simulatorTabIndex = 4;
const int packetsTabIndex_NoSimulator = 4;
const int packetsTabIndex_SimulatorActive = 5;

// ASIC sub-tab indices
const int ASICGlobalTabIndex = 0;
const int ASICAnodeTabIndex = 1;
const int ASICCathodeTabIndex = 2;

// Int to Hex consts
const int IntToHexFieldWidth = 2;
const int base16 = 16;

// data sets for Peak Detect Timeout, Time detect ramp length and peaking time
const int peakDetectTimoutVals [4] = {1, 2, 4, 8};
const int timeDetectRampLengthVals [4] = {1, 2, 3, 4};
const double peakingTimeVals [4] = {0.25, 0.5, 1.0, 2.0};

const int peakDetectTimeoutValHGFactor = 4;
const int timeDetectRampLengthValHGFactor = 4;
const int peakingTimeValHGFactor = 6;

const double testPulseStep = 1.808406647116325;
const double thresholdStep = 1.808406647116325;

const double thresholdTrimStep = 3.5;
const double thresholdTrimMin = -52.5;

const int pulseThresholdTrimStep = 2;
const int pulseThresholdTrimMin = -62;

const double multiThreshDisplacementStep = 6.25;
const double threshDisplacementMin = -100.0;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    m_Connected(false),
    m_PacketCount(0)
{
    ui->setupUi(this);
    SpectDMDll::SetConnectStatusFunction(ConnectStatusCallbackFunc);
    SpectDMDll::SetSysStatusFunction(SysStatusCallbackFunc);
    SpectDMDll::SetToolbarStatusFunction(ToolbarStatusCallbackFunc);

    // setup operation callbacks
    SpectDMDll::SetOperationErrorFunction(OperationErrorCallbackFunc);
    SpectDMDll::SetOperationCompleteFunction(OperationCompleteCallbackFunc);
    SpectDMDll::SetOperationProgressFunction(OperationProgressCallbackFunc);

    SetupSpectUI();
}

void MainWindow::ConnectStatusCallbackFunc(
    const char* a_Data
)
{
    GetInstance()->AddConnectStatusEntry(a_Data);
}

void MainWindow::SysStatusCallbackFunc(
    const char *a_Data
)
{
    GetInstance()->AddSysStatusEntry(a_Data);
}

void MainWindow::ToolbarStatusCallbackFunc(
    const char *a_Data
)
{
    GetInstance()->ui->statusBar->showMessage(a_Data);
}

void MainWindow::OperationErrorCallbackFunc(
    const char* a_Data
)
{
    GetInstance()->CloseProgressDlg();
    GetInstance()->ReportWarning(a_Data);
}

void MainWindow::OperationCompleteCallbackFunc()
{
    GetInstance()->CloseProgressDlg();
}

void MainWindow::OperationProgressCallbackFunc(
    int a_Progress
)
{
    emit GetInstance()->progressUpdate(a_Progress);
}

MainWindow *MainWindow::GetInstance()
{
    if(!m_pInstance)
    {
        m_pInstance = new MainWindow();
    }

    return m_pInstance;
}

void MainWindow::DeleteInstance()
{
    if(m_pInstance)
    {
        delete m_pInstance;
        m_pInstance = 0;
    }
}

void MainWindow::AddConnectStatusEntry(
    const std::string &a_Status
)
{
    ui->statusEdit->append(CreateTimestampedStr(a_Status));
}

void MainWindow::AddSysStatusEntry(
    const std::string &a_Status
)
{
    ui->sysStatusTextArea->append(CreateTimestampedStr(a_Status));
}

void MainWindow::keyPressEvent(
    QKeyEvent *a_KeyEvent
)
{
    if(a_KeyEvent->key() == Qt::Key_F1)
    {
        StartHelp();
    }
}

MainWindow::~MainWindow()
{
    SpectDMDll::Close();
    delete ui;
}

void MainWindow::SetupSpectUI()
{
    for(int i = 0; i < 128; i++)
    {
        ui->anodeChannelNoCombo->addItem(QString("%1").arg(i + 1));
        ui->anodeChannelMonitorCombo->addItem(QString("%1").arg(i + 1));
    }

    for(int i = 0; i < 2; i++)
    {
        ui->cathodeChannelNoCombo->addItem(QString("%2").arg((i + 1)));
    }

    UpdateHGAffectedWidgets(false);

    ui->individualAnodeRadio->setChecked(true);
    ui->individualCathodeRadio->setChecked(true);

    ui->disconnectButton->setEnabled(false);
    ui->stopSysBtn->setEnabled(false);
    ui->startCollectBtn->setEnabled(false);
    ui->stopCollectBtn->setEnabled(false);

#ifdef INTERNAL_BUILD
    ui->stopSimBtn->setEnabled(false);
#endif

    ui->DACScombo->setEnabled(false);
    ui->cathEnergyTimingCombo->setEnabled(false);
    ui->anodeChannelMonitorCombo->setEnabled(false);

    connect(ui->actionExit, SIGNAL(triggered()), this, SLOT(close()));

    QScrollArea* l_ScrollArea = new QScrollArea(this);
    l_ScrollArea->setWidget(ui->tabWidget);

    setCentralWidget(l_ScrollArea);

#ifndef INTERNAL_BUILD
    // remove simulator tab and packets tab, we remove index 3 twice as packets tab
    // becomes index 3 after we remove the simulator one.
    ui->tabWidget->removeTab(3);
    ui->tabWidget->removeTab(3);
#endif

    // disable all non-connect tabs until we connect
    EnableNonConnectTabs(false);
}

void MainWindow::on_connectButton_clicked()
{
    if(!ui->hostIPEdit->text().isEmpty() && !ui->cameraIPEdit->text().isEmpty())
    {
        SpectDMDll::SetHostIPAddress(ui->hostIPEdit->text().toStdString().c_str());
        SpectDMDll::SetCameraIPAddress(ui->cameraIPEdit->text().toStdString().c_str());

        if(SpectDMDll::Initialize())
        {
            m_Connected = true;
            ui->connectButton->setEnabled(false);
            ui->disconnectButton->setEnabled(true);
            ui->loadFromFileBtn->setEnabled(false);
            EnableNonConnectTabs(true);
            UpdateSysConfigItems();

#ifdef INTERNAL_BUILD
            UpdateSimulatorItems();
#endif
        }
        else
        {
            AddConnectStatusEntry(SpectDMDll::GetLastError().c_str());
        }
    }
    else
    {
        AddConnectStatusEntry("Connect requires Local IP and Remote IP addresses");
    }
}

void MainWindow::on_disconnectButton_clicked()
{
    if(m_Connected)
    {
        SpectDMDll::Disconnect();

        ui->connectButton->setEnabled(true);
        ui->disconnectButton->setEnabled(false);
        ui->loadFromFileBtn->setEnabled(true);

        EnableNonConnectTabs(false);
    }
}

void MainWindow::on_sendConfigButton_clicked()
{
    // check our controls and set things
    bool l_Success = true;

    if(l_Success)
    {
        if(!SpectDMDll::SetMBMultiplexerAddressType(static_cast<MBMultiplexerAddrType>(ui->mbMultiplexerAddrTypeCombo->currentIndex())))
        {
            l_Success = false;
        }
    }

    if(l_Success)
    {
        if(!SpectDMDll::SetPacketTransferRate(ui->pktTransferSpinner->value()))
        {
            l_Success = false;
        }
    }

    if(l_Success)
    {
        l_Success = SpectDMDll::SetDebugMode(ui->debugModeCheckBox->isChecked());
    }

    if(l_Success)
    {
        if(ui->SysTokenTypeBtnGroup->checkedButton())
        {
            SysTokenType l_TokenType = SysTokenType_Undefined;

            if(ui->tokenDaisyChainradio->isChecked())
            {
                l_TokenType = SysTokenType_DaisyChain;
            }
            else if(ui->tokenGM3radio->isChecked())
            {
                l_TokenType = SysTokenType_GM3Only;
            }

            if(!SpectDMDll::SetSysTokenType(l_TokenType))
            {
                l_Success = false;
            }
        }
    }

    if(l_Success)
    {
        PixelMappingMode l_PixelMappingMode = PixelMappingMode_Undefined;

        if(ui->pixelMapGMBasedRadio->isChecked())
        {
            l_PixelMappingMode = PixelMappingMode_GMBased;
        }
        else if(ui->pixelMapGlobalRadio->isChecked())
        {
            l_PixelMappingMode = PixelMappingMode_Global;
        }

        if(SpectDMDll::SetPixelMappingMode(l_PixelMappingMode))
        {
            // update GM enable pixel mapping enable state after updating pixel mapping mode
            UpdateGMPixelMapCheckbox();
        }
        else
        {
            l_Success = false;
        }
    }

//    if(l_Success)
//    {
//        if(ui->gmUpdateBtnGroup->checkedButton())
//        {
//            GMUpdateType l_Type = GMUpdateType_Undefined;

//            if(ui->updateSingleGMRadio->isChecked())
//            {
//                l_Type = GMUpdateType_SingleGM;
//            }
//            else if(ui->updateBroadcastRadio->isChecked())
//            {
//                l_Type = GMUpdateType_Broadcast;
//            }

//            if(!SpectDMDll::SetGMUpdateType(l_Type))
//            {
//                l_Success = false;
//            }
//        }
//    }

    // If camera update mode is auto then functions above would have sent data down to device
    if(l_Success && SpectDMDll::GetCameraUpdateMode() == CameraUpdateMode_Manual)
    {
        l_Success = SpectDMDll::SendGBEControlData();
    }

    if(!l_Success)
    {
        QMessageBox::warning(this, "SpectDM", SpectDMDll::GetLastError().c_str());
    }
}

void MainWindow::on_UpdateGMButton_clicked()
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

//void MainWindow::on_gmRegReadButton_clicked()
//{
//    QString l_GMRegNo = ui->gmRegEdit->text();

//    if(!l_GMRegNo.isEmpty())
//    {
//        unsigned char l_Data;
//        if(SpectDMDll::GMRegDirectRead(l_GMRegNo.toInt(), &l_Data))
//        {
//            //ui->gmDataEdit->clear();
//            int l_Val = static_cast<int>(l_Data);
//            ui->gmDataEdit->setText(QString("%1").arg(l_Val));
//        }
//        else
//        {
//            QMessageBox::warning(this, "GM Error", SpectDMDll::GetLastError().c_str());
//        }
//    }
//    else
//    {
//        QMessageBox::warning(this, "GM Error", "GM Reg No not set");
//    }
//}

//void MainWindow::on_asicTestBtn_clicked()
//{
    //SpectDMDll::SetActiveGM(1);

    // Cathode test clock
//    SpectDMDll::SetCathodeTestClockType(CathodeTestClockType_CopyAnodeTestClock);
//    CathodeTestClockType l_TestClockType = CathodeTestClockType_Undefined;
//    if(SpectDMDll::GetCathodeTestClockType(&l_TestClockType))
//    {
//        qDebug() << QString("Cathode Test Clock : %1").arg(l_TestClockType);
//    }
//    else
//    {
//        qDebug() << "Failed to get test clock";
//    }

//    // enable single event mode
//    SpectDMDll::SetASICGlobalOptions(ASICGlobal_SingleEventMode);
//    ASICGlobalOptions l_GlobOpt;

//    if(SpectDMDll::GetASICGlobalOptions(&l_GlobOpt))
//    {
//        qDebug() << QString("ASIC global options (SSE) : %1").arg(l_GlobOpt);
//    }
//    else
//    {
//        qDebug() << "Failed to get asic global options (SSE)";
//    }

//    // add more global options
//    l_GlobOpt |= (ASICGlobal_EnergyMultipleFiringSuppressor | ASICGlobal_Validation);

//    SpectDMDll::SetASICGlobalOptions(l_GlobOpt);

//    if(SpectDMDll::GetASICGlobalOptions(&l_GlobOpt))
//    {
//        qDebug() << QString("ASIC global options (SSP | SSE | EVALID) : %1").arg(l_GlobOpt);
//    }
//    else
//    {
//        qDebug() << "Failed to get asic global options (SSP | SSE | EVALID)";
//    }

//    SpectDMDll::SetPeakDetectTimeout(ChannelType_Anode, 8);

//    int l_AnodePeakDetectTimeout = 0;

//    if(SpectDMDll::GetPeakDetectTimeout(ChannelType_Anode, &l_AnodePeakDetectTimeout))
//    {
//        qDebug() << QString("Anode Peak Detect timeout: %1").arg(l_AnodePeakDetectTimeout);
//    }
//    else
//    {
//        qDebug() << "Failed to get anode peak detect timeout";
//    }

//    SpectDMDll::SetTimeDetectRampLength(ChannelType_Anode, 3);

//    int l_AnodeRampLength = 0;

//    if(SpectDMDll::GetTimeDetectRampLength(ChannelType_Anode, &l_AnodeRampLength))
//    {
//        qDebug() << QString("Anode time detect ramp length: %1").arg(l_AnodeRampLength);
//    }
//    else
//    {
//        qDebug() << "Failed to get ramp length";
//    }

//    SpectDMDll::SetASICGlobalOptions(ASICGlobal_MonitorOutputs);

//    if(SpectDMDll::GetASICGlobalOptions(&l_GlobOpt))
//    {
//        qDebug() << QString("ASIC global options (SAUXi) : %1").arg(l_GlobOpt);
//    }
//    else
//    {
//        qDebug() << "Failed to get asic global options (SAUXi)";
//    }

//    // could try sticking different ones in here (we should maybe check for undefined
//    SpectDMDll::SetAnalogOutputToMonitor(AnalogOutput_CathodeChannel1Energy);

//    AnalogOutput l_AnalogOutput = AnalogOutput_Undefined;

//    if(SpectDMDll::GetAnalogOutputMonitored(&l_AnalogOutput))
//    {
//        qDebug() << QString("Analog output monitored: %1").arg(l_AnalogOutput);
//    }
//    else
//    {
//        qDebug() << "Failed to get analog output monitored";
//    }

//    SpectDMDll::SetTimingChannelUnipolarGain(TimingChannelUnipolarGain_81mV);

//    TimingChannelUnipolarGain l_UnipolarGain = TimingChannelUnipolarGain_27mV; // needs an undefined option

//    if(SpectDMDll::GetTimingChannelUnipolarGain(&l_UnipolarGain))
//    {
//        qDebug() << QString("Timing channel unipolar gain: %1").arg(l_UnipolarGain);
//    }
//    else
//    {
//        qDebug() << "Failed to get unipolar gain";
//    }

//    SpectDMDll::SetTimingChannelBiPolarGain(TimingChannelBipolarGain_164mV);

//    TimingChannelBipolarGain l_BiPolarGain;

//    if(SpectDMDll::GetTimingChannelBiPolarGain(&l_BiPolarGain))
//    {
//        qDebug() << QString("Timing channel bipolar gain: %1").arg(l_BiPolarGain);
//    }
//    else
//    {
//        qDebug() << "Failed to get bipolar gain";
//    }

//    SpectDMDll::SetASICReadoutMode(GMASICReadout_EnhancedSparsified);

//    GMASICReadoutMode l_ReadoutMode = GMASICReadout_Undefined;

//    if(SpectDMDll::GetASICReadoutMode(&l_ReadoutMode))
//    {
//        qDebug() << QString("ASIC Readout mode: %1").arg(l_ReadoutMode);
//    }

//    SpectDMDll::SetASICGlobalOptions(ASICGlobal_RouteTempMonitorToAXPK62);

//    if(SpectDMDll::GetASICGlobalOptions(&l_GlobOpt))
//    {
//        qDebug() << QString("ASIC global options (STMPM) : %1").arg(l_GlobOpt);
//    }
//    else
//    {
//        qDebug() << "Failed to get asic global options (STMPM)";
//    }

//    SpectDMDll::SetPeakingTime(ChannelType_Anode, 0.5);

//    double l_AnodePeakingTime = 0;

//    if(SpectDMDll::GetPeakingTime(ChannelType_Anode, &l_AnodePeakingTime))
//    {
//        qDebug() << QString("Anode Peaking time : %1").arg(l_AnodePeakingTime);
//    }
//    else
//    {
//        qDebug() << "Failed to get anode peaking time";
//    }

//    SpectDMDll::SetASICGlobalOptions(ASICGlobal_TimingMultipleFiringSuppressor
//                                   | ASICGlobal_DisableMultipleResetAcquisitionMode
//                                   | ASICGlobal_RouteMonitorToPinTDO);

//    if(SpectDMDll::GetASICGlobalOptions(&l_GlobOpt))
//    {
//        qDebug() << QString("ASIC global options (SSET | DISMRS | SAUXTD) : %1").arg(l_GlobOpt);
//    }
//    else
//    {
//        qDebug() << "Failed to get asic global options (SSET | DISMRS | SAUXTD)";
//    }

//    SpectDMDll::SetChannelGain(ChannelGain_60mV);

//    ChannelGain l_ChannelGain = ChannelGain_20mV;

//    if(SpectDMDll::GetChannelGain(&l_ChannelGain))
//    {
//        qDebug() << QString("Channel Gain : %1").arg(l_ChannelGain);
//    }
//    else
//    {
//        qDebug() << "Failed to get channel gain";
//    }

//    SpectDMDll::SetPeakingTime(ChannelType_Cathode, 1);

//    double l_CathodePeakingTime = 0.0;

//    if(SpectDMDll::GetPeakingTime(ChannelType_Cathode, &l_CathodePeakingTime))
//    {
//        qDebug() << QString("Cathode Peaking time: %1").arg(l_CathodePeakingTime);
//    }
//    else
//    {
//        qDebug() <<"Failed to get cathode channels peaking time";
//    }

//    SpectDMDll::SetPeakDetectTimeout(ChannelType_Cathode, 4);

//    int l_CathodePeakDetectTimeout = 0;

//    if(SpectDMDll::GetPeakDetectTimeout(ChannelType_Cathode, &l_CathodePeakDetectTimeout))
//    {
//        qDebug() << QString("Cathode Peak detect timeout: %1").arg(l_CathodePeakDetectTimeout);
//    }
//    else
//    {
//        qDebug() << "Failed to get cathode peak detect timeout";
//    }

//    SpectDMDll::SetTimeDetectRampLength(ChannelType_Cathode, 4);

//    int l_CathodeRampLength = 0;

//    if(SpectDMDll::GetTimeDetectRampLength(ChannelType_Cathode, &l_CathodeRampLength))
//    {
//        qDebug() << QString("Cathode ramp length: %1").arg(l_CathodeRampLength);
//    }
//    else
//    {
//        qDebug() << "Failed to get cathode ramp length";
//    }

//    SpectDMDll::SetCathodeTimingChannelsShaperPeakingTime(TimingChannelsShaperPeakingTime_800nS);

//    TimingChannelsShaperPeakingTime l_ShaperPeakTime = TimingChannelsShaperPeakingTime_100nS; // needs undefined

//    if(SpectDMDll::GetCathodeTimingChannelsShaperPeakingTime(&l_ShaperPeakTime))
//    {
//        qDebug() << QString("Cathode shaper peaking time : %1").arg(l_ShaperPeakTime);
//    }
//    else
//    {
//        qDebug() << "Failed to get cathode shaper peak time";
//    }

//    SpectDMDll::SetCathodeChannelInternalLeakageCurrentGenerator(1, CathodeChannel::InternalLeakageCurrentGenerator_2nA);

//    CathodeChannel::InternalLeakageCurrentGenerator l_CurrGen = CathodeChannel::InternalLeakageCurrentGenerator_350pA;

//    if(SpectDMDll::GetCathodeChannelInternalLeakageCurrentGenerator(1, &l_CurrGen))
//    {
//        qDebug() << QString("Cathode internal leakage channel 1 : %1").arg(l_CurrGen);
//    }
//    else
//    {
//        qDebug() << "Failed to get cathode internal leakage channel 1";
//    }

//    SpectDMDll::SetCathodeChannelInternalLeakageCurrentGenerator(2, CathodeChannel::InternalLeakageCurrentGenerator_2nA);

//    //CathodeChannel::InternalLeakageCurrentGenerator l_CurrGen = CathodeChannel::InternalLeakageCurrentGenerator_350pA;

//    if(SpectDMDll::GetCathodeChannelInternalLeakageCurrentGenerator(2, &l_CurrGen))
//    {
//        qDebug() << QString("Cathode internal leakage channel 2 : %1").arg(l_CurrGen);
//    }
//    else
//    {
//        qDebug() << "Failed to get cathode internal leakage channel 2";
//    }
    //SpectDMDll::MaskChannel(ChannelType_Anode, 1, true);

//    if(!SpectDMDll::Initialize())
//    {
//        qDebug() << SpectDMDll::GetLastError().c_str();
//    }

//}

void MainWindow::on_ASICGlobal_UpdateASICButton_clicked()
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
            UpdateASICAnodeItems();
            UpdateASICCathodeItems();
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

void MainWindow::on_ASICAnode_UpdateASICButton_clicked()
{
    bool l_Success = true;

    if(ui->GMASICAnodeTestPulseEdgeBtnGroup->checkedButton())
    {
        TestPulseEdge l_TestPulseEdge = TestPulseEdge_Undefined;

        if(ui->anodeEdge_negCharge_radio->isChecked())
        {
            l_TestPulseEdge = TestPulseEdge_InjectNegCharge;
        }
        else if(ui->anodeTestPulse_PosAndNeg_radio->isChecked())
        {
            l_TestPulseEdge = TestPulseEdge_InjectPosAndNegCharge;
        }

        if(!SpectDMDll::SetAnodeTestPulseEdge(l_TestPulseEdge))
        {
            l_Success = false;
        }
    }

    if(l_Success)
    {
        if(!SpectDMDll::SetPeakDetectTimeout(ChannelType_Anode, ui->anodePeakDetectTimoutCombo->currentText().toInt()))
        {
            l_Success = false;
        }
    }

    if(l_Success)
    {
        if(!SpectDMDll::SetTimeDetectRampLength(ChannelType_Anode, ui->anodeTimeDetectRampLengthCombo->currentText().toInt()))
        {
            l_Success = false;
        }
    }

    if(l_Success)
    {
        if(!SpectDMDll::SetPeakingTime(ChannelType_Anode, ui->anodePeakingTimeCombo->currentText().toDouble()))
        {
            l_Success = false;
        }
    }

    if(l_Success)
    {
        if(!SpectDMDll::SetTestPulseStep(ChannelType_Anode, ui->anodeTestPulseSpinner->value()))
        {
            l_Success = false;
        }
    }

    if(l_Success)
    {
        if(!SpectDMDll::SetChannelThresholdStep(ChannelThresholdType_AnodePositiveEnergy, ui->anodePosEnergyThreshSpinner->value()))
        {
            l_Success = false;
        }
    }

    if(l_Success)
    {
        if(!SpectDMDll::SetChannelThresholdStep(ChannelThresholdType_AnodeNegativeEnergy, ui->anodeNegEnergyThreshSpinner->value()))
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

void MainWindow::on_updateAnodeChannelButton_clicked()
{
    bool l_Success = true;

    SpectDMDll::SetActiveChannelType(ChannelType_Anode);

    if(ui->allAnodesRadio->isChecked())
    {
        SpectDMDll::SetChannelUpdateType(ChannelUpdateType_Broadcast);
    }
    else
    {
        // individual channel selected.
        SpectDMDll::SetChannelUpdateType(ChannelUpdateType_SingleChannel);
        SpectDMDll::SetActiveChannel(ui->anodeChannelNoCombo->currentText().toInt());
    }

    if(!SpectDMDll::MaskChannel(ui->anodeChannelMaskCheck->isChecked()))
    {
        l_Success = false;
    }

    if(l_Success)
    {
        if(!SpectDMDll::EnableChannelTestCapacitor(ui->enableTestCapacitorCheck->isChecked()))
        {
            l_Success = false;
        }
    }

    if(l_Success)
    {
        if(ui->ASICAnodeChannelMonSignalBtnGroup->checkedButton())
        {
            Signal l_AnodeSignal = Signal_Undefined;

            if(ui->anodeMonitorPosSignalCheck->isChecked())
            {
                l_AnodeSignal = Signal_Positive;
            }
            else if(ui->anodeMonitorNegSignalCheck->isChecked())
            {
                l_AnodeSignal = Signal_Negative;
            }

            if(!SpectDMDll::MonitorAnodeSignal(l_AnodeSignal))
            {
                l_Success = false;
            }
        }
    }

    if(l_Success)
    {
        if(!SpectDMDll::SetChannelPositivePulseThresholdTrimStep(ui->anodeChnlPosTrimSpinner->value()))
        {
            l_Success = false;
        }
    }

    if(l_Success)
    {
        if(!SpectDMDll::SetAnodeChannelNegativePulseThresholdTrimStep(ui->anodeChnlsNegTrimSpinner->value()))
        {
            l_Success = false;
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

void MainWindow::on_anodeChannelNoCombo_currentIndexChanged(const QString & /*arg1*/)
{
    if(SpectDMDll::IsActiveGMSet())
    {
        UpdateSelectedAnodeChannelItems();
    }
}

void MainWindow::on_cathode_UpdateASICButton_clicked()
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

void MainWindow::on_updateCathodeChannelButton_clicked()
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

void MainWindow::on_cathodeChannelNoCombo_currentTextChanged(const QString &)
{
    if(SpectDMDll::IsActiveGMSet())
    {
        UpdateSelectedCathodeChannelItems();
    }
}

void MainWindow::UpdateGMItems()
{
    // GM options
    GMOption l_GMOptions = GMOption_None;

    if(SpectDMDll::GetGMOptions(&l_GMOptions))
    {

            ui->GMDisPhotonCollectCheck->setChecked((l_GMOptions & GMOption_DisablePhotonCollect) != 0);
            ui->GMDebugCheck->setChecked((l_GMOptions & GMOption_DebugMode) != 0);
            ui->GMChannel1TestModeCheck->setChecked((l_GMOptions & GMOption_Channel1TestMode) != 0);
            ui->GMPixelMapCheck->setChecked((l_GMOptions & GMOption_EnablePixelMap) != 0);
    }
    else
    {
        QMessageBox::warning(this, "GM Error", SpectDMDll::GetLastError().c_str());
    }

    // delay time
    int l_DelayTime = 0;

    if(SpectDMDll::GetDelayTime(&l_DelayTime))
    {
        ui->delayTimeSpinner->setValue(l_DelayTime);
    }
    else
    {
        QMessageBox::warning(this, "GM Error", SpectDMDll::GetLastError().c_str());
    }

    // timestamp resolution
    int l_TimestampRes = 0;
    if(SpectDMDll::GetTimestampResolution(&l_TimestampRes))
    {
        ui->timestampResSpinner->setValue(l_TimestampRes);
    }
    else
    {
        QMessageBox::warning(this, "GM Error", SpectDMDll::GetLastError().c_str());
    }

    // ADC channels
    int l_ADCChannel = 0;

    if(SpectDMDll::GetGMADC1Channel(&l_ADCChannel))
    {
        ui->ADC1ChannelCombo->setCurrentText(QString("%1").arg(l_ADCChannel));
    }
    else
    {
        QMessageBox::warning(this, "GM Error", SpectDMDll::GetLastError().c_str());
    }

    if(SpectDMDll::GetGMADC2Channel(&l_ADCChannel))
    {
        ui->ADC2ChannelCombo->setCurrentText(QString("%1").arg(l_ADCChannel));
    }
    else
    {
        QMessageBox::warning(this, "GM Error", SpectDMDll::GetLastError().c_str());
    }

    // Default GM
    int l_DefaultGM = 0;

    if(SpectDMDll::GetDefaultGM(&l_DefaultGM))
    {
        ui->defaultGMCombo->setCurrentText(QString("%1").arg(l_DefaultGM));
    }
    else
    {
        QMessageBox::warning(this, "GM Error", SpectDMDll::GetLastError().c_str());
    }

    // GM readout options
    GMReadoutOption l_ReadoutOpt = GMReadoutOpt_None;

    if(SpectDMDll::GetGMReadoutOptions(&l_ReadoutOpt))
    {
        ui->negEnergyCheck->setChecked((l_ReadoutOpt & GMReadoutOpt_NegativeEnergy) != 0);
        ui->timeDetectCheck->setChecked((l_ReadoutOpt & GMReadoutOpt_TimeDetect) != 0);
    }
    else
    {
        QMessageBox::warning(this, "GM Error", SpectDMDll::GetLastError().c_str());
    }

    // GM readout
    GMReadoutMode l_ReadoutMode = GMReadout_Undefined;

    if(SpectDMDll::GetGMReadoutMode(&l_ReadoutMode))
    {
        switch(l_ReadoutMode)
        {
            case GMReadout_ReadAll:
            {
                ui->readAllRadio->setChecked(true);
                break;
            }
            case GMReadout_SparsifiedMode:
            {
                ui->sparsifiedRadio->setChecked(true);
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
        QMessageBox::warning(this, "GM Error", SpectDMDll::GetLastError().c_str());
    }

    // GM cath mode
    GMCathodeMode l_CathMode = GMCathMode_Undefined;

    if(SpectDMDll::GetGMCathodeMode(&l_CathMode))
    {
        switch(l_CathMode)
        {
            case GMCathMode_Unipolar:
            {
                ui->unipolarRadio->setChecked(true);
                break;
            }
            case GMCathMode_MultiThreshold:
            {
                ui->multiThreshRadio->setChecked(true);
                break;
            }
            case GMCathMode_Bipolar:
            {
                ui->bipolarRadio->setChecked(true);
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
        QMessageBox::warning(this, "GM Error", SpectDMDll::GetLastError().c_str());
    }

    // pulser freq
    GMPulserFrequency l_GMPulserFreq = GMPulserFreq_Undefined;

    if(SpectDMDll::GetGMPulserFrequency(&l_GMPulserFreq))
    {
        switch(l_GMPulserFreq)
        {
            case GMPulserFreq_100Hz:
            {
                ui->hundredHzRadio->setChecked(true);
                break;
            }
            case GMPulserFreq_1kHz:
            {
                ui->oneKHzRadio->setChecked(true);
                break;
            }
            case GMPulserFreq_10kHz:
            {
                ui->tenKHz->setChecked(true);
                break;
            }
            case GMPulserFreq_100kHz:
            {
                ui->hundredKHz->setChecked(true);
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
        QMessageBox::warning(this, "GM Error", SpectDMDll::GetLastError().c_str());
    }

    // pulser options
    GMPulserOption l_PulserOpt = GMPulserOpt_None;

    if(SpectDMDll::GetGMPulserOptions(&l_PulserOpt))
    {
        ui->anodePulserCheck->setChecked((l_PulserOpt & GMPulserOpt_Anode) != 0);
        ui->cathodePulserCheck->setChecked((l_PulserOpt & GMPulserOpt_Cathode) != 0);
    }
    else
    {
        QMessageBox::warning(this, "GM Error", SpectDMDll::GetLastError().c_str());
    }

    // no of pulses
    int l_NoOfPulses = 0;

    if(SpectDMDll::GetNoOfPulses(&l_NoOfPulses))
    {
        ui->noOfPulsesCombo->setCurrentText(QString("%1").arg(l_NoOfPulses));
    }
    else
    {
        QMessageBox::warning(this, "GM Error", SpectDMDll::GetLastError().c_str());
    }
}

void MainWindow::UpdateASICGlobalItems()
{
    // ASIC - GLobal tab
    ASICGlobalOptions l_GlobalOpt = ASICGlobal_None;

    if(SpectDMDll::GetASICGlobalOptions(&l_GlobalOpt))
    {
        ui->routeMonitorToPinTDOCheck->setChecked((l_GlobalOpt & ASICGlobal_RouteMonitorToPinTDO) != 0);
        ui->routeTempMonPinAXPK62Check->setChecked((l_GlobalOpt & ASICGlobal_RouteTempMonitorToAXPK62) != 0);
        ui->disableMultiResetAcqModeCheck->setChecked((l_GlobalOpt & ASICGlobal_DisableMultipleResetAcquisitionMode) != 0);
        ui->timingMFSCheck->setChecked((l_GlobalOpt & ASICGlobal_TimingMultipleFiringSuppressor) != 0);
        ui->energyMFSCheck->setChecked((l_GlobalOpt & ASICGlobal_EnergyMultipleFiringSuppressor) != 0);
        ui->highGainCheck->setChecked((l_GlobalOpt & ASICGlobal_HighGain) != 0);

        UpdateHGAffectedWidgets(ui->highGainCheck->isChecked());

        ui->monitorOutputsCheck->setChecked((l_GlobalOpt & ASICGlobal_MonitorOutputs) != 0);
        ui->singleEventModeCheck->setChecked((l_GlobalOpt & ASICGlobal_SingleEventMode) != 0);
        ui->validationCheck->setChecked((l_GlobalOpt & ASICGlobal_Validation) != 0);

        // buffer enable global options
        ui->chnl62PreAmpMonitorOutputCheck->setChecked((l_GlobalOpt & ASICGlobal_BufferChnl62PreAmplifierMonitorOutput) != 0);
        ui->chnl63PreAmpMonitorOutputCheck->setChecked((l_GlobalOpt & ASICGlobal_BufferChnl63PreAmplifierMonitorOutput) != 0);
        ui->peakAndTimeDetectorOutputCheck->setChecked((l_GlobalOpt & ASICGlobal_BufferPeakAndTimeDetectorOutputs) != 0);
        ui->auxMonitorOutputCheck->setChecked((l_GlobalOpt & ASICGlobal_BufferAuxMonitorOutput) != 0);
    }
    else
    {
        QMessageBox::warning(this, "ASIC error", SpectDMDll::GetLastError().c_str());
    }

    TimingChannelUnipolarGain l_TimingUnipolarGain = TimingChannelUnipolarGain_Undefined;

    if(SpectDMDll::GetTimingChannelUnipolarGain(&l_TimingUnipolarGain))
    {
        switch(l_TimingUnipolarGain)
        {
            case TimingChannelUnipolarGain_27mV:
            {
                ui->unipolarGain_27radio->setChecked(true);
                break;
            }
            case TimingChannelUnipolarGain_81mV:
            {
                ui->unipolarGain_81radio->setChecked(true);
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
        QMessageBox::warning(this, "ASIC error", SpectDMDll::GetLastError().c_str());
    }

    TimingChannelBipolarGain l_TimingBipolarGain = TimingChannelBipolarGain_Undefined;

    if(SpectDMDll::GetTimingChannelBiPolarGain(&l_TimingBipolarGain))
    {
        switch(l_TimingBipolarGain)
        {
            case TimingChannelBipolarGain_21mV:
            {
                ui->timingChannelBiPolarGain_21mV_radio->setChecked(true);
                break;
            }
            case TimingChannelBipolarGain_55mV:
            {
                ui->timingChannelBiPolarGain_55mV_radio->setChecked(true);
                break;
            }
            case TimingChannelBipolarGain_63mV:
            {
                ui->timingChannelBiPolarGain_63mV_radio->setChecked(true);
                break;
            }
            case TimingChannelBipolarGain_164mV:
            {
                ui->timingChannelBiPolarGain_164mV_radio->setChecked(true);
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
        QMessageBox::warning(this, "ASIC error", SpectDMDll::GetLastError().c_str());
    }

    GMASICReadoutMode l_ASICReadoutMode = GMASICReadout_Undefined;

    if(SpectDMDll::GetASICReadoutMode(&l_ASICReadoutMode))
    {
        switch(l_ASICReadoutMode)
        {
            case GMASICReadout_NormalSparsified:
            {
                ui->ASICNormalSparsifiedRadio->setChecked(true);
                break;
            }
            case GMASICReadout_EnhancedSparsified:
            {
                ui->ASICEnhancedSparsifiedRadio->setChecked(true);
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
        QMessageBox::warning(this, "ASIC error", SpectDMDll::GetLastError().c_str());
    }

    ChannelGain l_ChannelGain = ChannelGain_Undefined;

    if(SpectDMDll::GetChannelGain(&l_ChannelGain))
    {
        switch(l_ChannelGain)
        {
            case ChannelGain_20mV:
            {
                ui->channelGain_20radio->setChecked(true);
                break;
            }
            case ChannelGain_60mV:
            {
                ui->channelGain_60Radio->setChecked(true);
            }
            default:
            {
                break;
            }
        }
    }
    else
    {
        QMessageBox::warning(this, "ASIC error", SpectDMDll::GetLastError().c_str());
    }

    InternalLeakageCurrentGenerator l_InternalLeakage = InternalLeakageCurrentGenerator_Undefined;

    if(SpectDMDll::GetInternalLeakageCurrentGenerator(&l_InternalLeakage))
    {
        switch(l_InternalLeakage)
        {
            case InternalLeakageCurrentGenerator_0A:
            {
                ui->internalLeakageCurrGen_0A_radio->setChecked(true);
                break;
            }
            case InternalLeakageCurrentGenerator_60pA:
            {
                ui->internalLeakageCurrGen_60pA_radio->setChecked(true);
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
        QMessageBox::warning(this, "ASIC error", SpectDMDll::GetLastError().c_str());
    }

    MultipleFiringSuppressionTime l_SuppressTime = MultipleFiringSuppressionTime_Undefined;

    if(SpectDMDll::GetMultipleFiringSuppressTime(&l_SuppressTime))
    {
        switch(l_SuppressTime)
        {
            case MultipleFiringSuppressionTime_62_5nS:
            {
                ui->multipleFiringSuppress_62_5_radio->setChecked(true);
                break;
            }
            case MultipleFiringSuppressionTime_125nS:
            {
                ui->multipleFiringSuppress_125_radio->setChecked(true);
                break;
            }
            case MultipleFiringSuppressionTime_250nS:
            {
                ui->multipleFiringSuppress_250_radio->setChecked(true);
                break;
            }
            case MultipleFiringSuppressionTime_600nS:
            {
                ui->multipleFiringSuppress_600_radio->setChecked(true);
            }
            default:
            {
                break;
            }
        }
    }
    else
    {
        QMessageBox::warning(this, "ASIC error", SpectDMDll::GetLastError().c_str());
    }

    AnalogOutput l_AnalogOutput = AnalogOutput_Undefined;

    // do this here and let individual cases do what they need to do
    DisableAnalogOutputCombos();

    if(SpectDMDll::GetAnalogOutputMonitored(&l_AnalogOutput))
    {
        switch(l_AnalogOutput)
        {
            case AnalogOutput_NoFunction:
            {
                ui->noFuncRadio->setChecked(true);
                break;
            }
            case AnalogOutput_Baseline:
            {
                ui->baselineRadio->setChecked(true);
                break;
            }
            case AnalogOutput_Temperature:
            {
                ui->temperatureRadio->setChecked(true);
                break;
            }
            case AnalogOutput_AnodeEnergy:
            {
                ui->anodeChannelRadio->setChecked(true);
                ui->anodeChannelMonitorCombo->setEnabled(true);

                int l_Channel = 0;
                if(SpectDMDll::GetAnodeChannelMonitored(&l_Channel))
                {
                    ui->anodeChannelMonitorCombo->setCurrentText(QString("%1").arg(l_Channel));
                }
                else
                {
                    QMessageBox::warning(this, "ASIC error", SpectDMDll::GetLastError().c_str());
                }
                break;
            }
            case AnalogOutput_CathodeEnergyTiming:
            {
                ui->cathodeEnergyTimingRadio->setChecked(true);
                ui->cathEnergyTimingCombo->setEnabled(true);

                CathodeEnergyTiming l_CathEnergyTiming = CathodeEnergyTiming_Undefined;

                if(SpectDMDll::GetCathodeEnergyTimingMonitored(&l_CathEnergyTiming))
                {
                    int l_Index = static_cast<int>(l_CathEnergyTiming);

                    // the enum starts with undefined but the combo box only contains valid items
                    // so deduct one from casted value to get index we need.
                    l_Index -= 1;

                    ui->cathEnergyTimingCombo->setCurrentIndex(l_Index);
                }
                else
                {
                    QMessageBox::warning(this, "ASIC error", SpectDMDll::GetLastError().c_str());
                }

                break;
            }
            case AnalogOutput_DACS:
            {
                ui->DACSradio->setChecked(true);
                ui->DACScombo->setEnabled(true);

                DACS l_DACMonitored = DACS_Undefined;

                if(SpectDMDll::GetDACMonitored(&l_DACMonitored))
                {
                    int l_Index = static_cast<int>(l_DACMonitored);

                    // see case above for why we do this
                    l_Index -= 1;

                    ui->DACScombo->setCurrentIndex(l_Index);
                }
                else
                {
                    QMessageBox::warning(this, "ASIC error", SpectDMDll::GetLastError().c_str());
                }

                break;
            }
            default:
            {
                // should never get here but add assert
                break;
            }
        }
    }
    else
    {
        QMessageBox::warning(this, "ASIC error", SpectDMDll::GetLastError().c_str());
    }
}

void MainWindow::UpdateASICAnodeItems()
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

void MainWindow::UpdateASICCathodeItems()
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

void MainWindow::UpdateSelectedAnodeChannelItems()
{
    int l_ChannelNo = ui->anodeChannelNoCombo->currentText().toInt();

    bool l_Masked = false;
    if(SpectDMDll::IsChannelMasked(ChannelType_Anode, l_ChannelNo, &l_Masked))
    {
        ui->anodeChannelMaskCheck->setChecked(l_Masked);
    }
    else
    {
        QMessageBox::warning(this, "ASIC anode channel error", SpectDMDll::GetLastError().c_str());
    }

    bool l_TestCapacitorEnabled = false;

    if(SpectDMDll::IsChannelTestCapacitorEnabled(ChannelType_Anode, l_ChannelNo, &l_TestCapacitorEnabled))
    {
        ui->enableTestCapacitorCheck->setChecked(l_TestCapacitorEnabled);
    }
    else
    {
        QMessageBox::warning(this, "ASIC anode channel error", SpectDMDll::GetLastError().c_str());
    }

    Signal l_AnodeSignalMonitored = Signal_Undefined;

    if(SpectDMDll::GetAnodeSignalMonitored(l_ChannelNo, &l_AnodeSignalMonitored))
    {
        switch(l_AnodeSignalMonitored)
        {
            case Signal_Positive:
            {
                ui->anodeMonitorPosSignalCheck->setChecked(true);
                break;
            }
            case Signal_Negative:
            {
                ui->anodeMonitorNegSignalCheck->setChecked(true);
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
        QMessageBox::warning(this, "ASIC anode channel error", SpectDMDll::GetLastError().c_str());
    }

    int l_PosTrimStep = 0;

    if(SpectDMDll::GetChannelPositivePulseThresholdTrimStep(ChannelType_Anode, l_ChannelNo, &l_PosTrimStep))
    {
        ui->anodeChnlPosTrimSpinner->setValue(l_PosTrimStep);
    }
    else
    {
        QMessageBox::warning(this, "ASIC anode channel error", SpectDMDll::GetLastError().c_str());
    }

    int l_NegTrimStep = 0;

    if(SpectDMDll::GetAnodeChannelNegativePulseThresholdTrimStep(l_ChannelNo, &l_NegTrimStep))
    {
        ui->anodeChnlsNegTrimSpinner->setValue(l_NegTrimStep);
    }
    else
    {
        QMessageBox::warning(this, "ASIC anode channel error", SpectDMDll::GetLastError().c_str());
    }
}

void MainWindow::UpdateSelectedCathodeChannelItems()
{
    int l_ChannelNo = ui->cathodeChannelNoCombo->currentText().toInt();

    bool l_Masked = false;
    if(SpectDMDll::IsChannelMasked(ChannelType_Cathode, l_ChannelNo, &l_Masked))
    {
        ui->cathodeChannelMaskCheck->setChecked(l_Masked);
    }
    else
    {
        QMessageBox::warning(this, "ASIC cathode channel error", SpectDMDll::GetLastError().c_str());
    }

    bool l_TestCapacitorEnabled = false;

    if(SpectDMDll::IsChannelTestCapacitorEnabled(ChannelType_Cathode, l_ChannelNo, &l_TestCapacitorEnabled))
    {
        ui->cathodeEnableTestCapacitorCheck->setChecked(l_TestCapacitorEnabled);
    }
    else
    {
        QMessageBox::warning(this, "ASIC cathode channel error", SpectDMDll::GetLastError().c_str());
    }

    CathodeChannel::ShapedTimingSignal l_ShapedTimingSignal = CathodeChannel::ShapedTimingSignal_Undefined;

    if(SpectDMDll::GetCathodeChannelShapedTimingSignal(l_ChannelNo, &l_ShapedTimingSignal))
    {
        switch(l_ShapedTimingSignal)
        {
            case CathodeChannel::ShapedTimingSignal_Unipolar:
            {
                ui->cathodeShapedTimingSignalUnipolar_radio->setChecked(true);
                break;
            }
            case CathodeChannel::ShapedTimingSignal_Bipolar:
            {
                ui->shapedTimingSignalBipolar_radio->setChecked(true);
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
        QMessageBox::warning(this, "ASIC cathode channel error", SpectDMDll::GetLastError().c_str());
    }

    int l_PosTrimStep = 0;

    if(SpectDMDll::GetChannelPositivePulseThresholdTrimStep(ChannelType_Cathode, l_ChannelNo, &l_PosTrimStep))
    {
        ui->cathodeChnlPosTrimSpinner->setValue(l_PosTrimStep);
    }
    else
    {
        QMessageBox::warning(this, "ASIC cathode channel error", SpectDMDll::GetLastError().c_str());
    }

    int l_UnipolarTrimStep = 0;
    if(SpectDMDll::GetCathodeTimingChannelTrimStep(l_ChannelNo, CathodeTimingChannelType_Unipolar, &l_UnipolarTrimStep))
    {
        ui->cathodeChnlUnipolarTrimSpinner->setValue(l_UnipolarTrimStep);
    }
    else
    {
        QMessageBox::warning(this, "ASIC cathode channel error", SpectDMDll::GetLastError().c_str());
    }

    int l_FirstMultiTrimStep = 0;
    if(SpectDMDll::GetCathodeTimingChannelTrimStep(l_ChannelNo, CathodeTimingChannelType_FirstMultiThresholdBiPolar, &l_FirstMultiTrimStep))
    {
        ui->cathodeChnlFirstMultiTrimSpinner->setValue(l_FirstMultiTrimStep);
    }
    else
    {
        QMessageBox::warning(this, "ASIC cathode channel error", SpectDMDll::GetLastError().c_str());
    }

    int l_SecondMultiTrimStep = 0;
    if(SpectDMDll::GetCathodeTimingChannelTrimStep(l_ChannelNo, CathodeTimingChannelType_SecondMultiThreshold, &l_SecondMultiTrimStep))
    {
        ui->cathodeChnlSecondMultiTrimSpinner->setValue(l_SecondMultiTrimStep);
    }
    else
    {
        QMessageBox::warning(this, "ASIC cathode channel error", SpectDMDll::GetLastError().c_str());
    }

    int l_ThirdMultiTrimStep = 0;
    if(SpectDMDll::GetCathodeTimingChannelTrimStep(l_ChannelNo, CathodeTimingChannelType_ThirdMultiThreshold, &l_ThirdMultiTrimStep))
    {
        ui->cathodeChnlThirdMultiTrimSpinner->setValue(l_ThirdMultiTrimStep);
    }
    else
    {
        QMessageBox::warning(this, "ASIC cathode channel error", SpectDMDll::GetLastError().c_str());
    }

    CathodeChannel::TimingMode l_CathTimingMode = CathodeChannel::TimingMode_Undefined;

    if(SpectDMDll::GetCathodeChannelTimingMode(l_ChannelNo, &l_CathTimingMode))
    {
        switch(l_CathTimingMode)
        {
            case CathodeChannel::TimingMode_Unipolar:
            {
                ui->cathodeTimingModeUnipolar_radio->setChecked(true);
                break;
            }
            case CathodeChannel::TimingMode_MultiThreshold_Unipolar:
            {
                ui->cathodeTimingModeMultiUnipolar_radio->setChecked(true);
                break;
            }
            case CathodeChannel::TimingMode_BiPolar_Unipolar:
            {
                ui->cathodeChnlTimingModeBiPolarUniPolar_radio->setChecked(true);
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
        QMessageBox::warning(this, "ASIC cathode channel error", SpectDMDll::GetLastError().c_str());
    }
}

void MainWindow::UpdateSysConfigItems()
{
    // set mb multiplexer addr type
    MBMultiplexerAddrType l_MBMultiplexAddrType = SpectDMDll::GetMBMultiplexerAddressType();
    ui->mbMultiplexerAddrTypeCombo->setCurrentIndex(static_cast<int>(l_MBMultiplexAddrType));

    // token type
    SysTokenType l_TokenType = SpectDMDll::GetSysTokenType();
    QRadioButton* l_TokenTypeToCheck = 0;

    switch(l_TokenType)
    {
        case SysTokenType_DaisyChain:
        {
            l_TokenTypeToCheck = ui->tokenDaisyChainradio;
            break;
        }
        case SysTokenType_GM3Only:
        {
            l_TokenTypeToCheck = ui->tokenGM3radio;
            break;
        }
        default:
        {
            break;
        }
    }

    if(l_TokenTypeToCheck)
    {
        l_TokenTypeToCheck->setChecked(true);
    }

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

#ifdef INTERNAL_BUILD

void MainWindow::UpdateSimulatorItems()
{
    SimulatorOptions l_Options = SpectDMDll::GetSimulatorOptions();

    if(l_Options & SimulatorOption_EnableTimeData)
    {
        ui->simTimeDataCheck->setChecked(true);
    }

    if(l_Options & SimulatorOption_EnableNegativeSignalData)
    {
        ui->simNegSignalData->setChecked(true);
    }

    if(l_Options & SimulatorOption_Cycle)
    {
        ui->simCycleCheck->setChecked(true);
        // disable cycle related vars?
    }

    if(l_Options & SimulatorOption_EnableCathodeData)
    {
        ui->simCathodeGroup->setChecked(true);
    }

    SimulatorPacketFrequency l_SimPktFreq = SpectDMDll::GetSimulatorPacketFrequency();

    QRadioButton* l_ButtonToCheck = 0;

    switch(l_SimPktFreq)
    {
        case SimulatorPacketFrequency_100Hz:
        {
            l_ButtonToCheck = ui->simPktFreq_100Hz_radio;
            break;
        }
        case SimulatorPacketFrequency_1kHz:
        {
            l_ButtonToCheck = ui->simPktFreq_1kHz_radio;
            break;
        }
        case SimulatorPacketFrequency_10kHz:
        {
            l_ButtonToCheck = ui->simPktFreq_10kHz_radio;
            break;
        }
        case SimulatorPacketFrequency_100kHz:
        {
            l_ButtonToCheck = ui->simPktFreq_100kHz_radio;
            break;
        }
        default:
        {
            Q_ASSERT_X(0, "MainWindow::UpdateSimulatorItems()", "Unhandled SimulatorPacketFrequency");
            break;
        }
    }

    if(l_ButtonToCheck)
    {
        l_ButtonToCheck->setChecked(true);
    }

    ui->simNoOfPhotonsCombo->setCurrentText(IntToHexString(SpectDMDll::GetSimulatorPhotonsPerPacket()));
    ui->simGMNoCombo->setCurrentText(IntToHexString(SpectDMDll::GetSimulatorGMNo()));
    ui->simNoOfPktsEdit->setText(IntToHexString(SpectDMDll::GetSimulatorNoOfPackets()));
    ui->simTimeOffsetEdit->setText(IntToHexString(SpectDMDll::GetSimulatorTimeOffset()));

    SimulatorPhotonType l_PhotonType = SpectDMDll::GetSimulatorPhotonType();

    int l_PhotonTypeAsInt = 0;

    switch(l_PhotonType)
    {
        case SimulatedPhotonType_Type0:
        {
            l_PhotonTypeAsInt = 0;
            break;
        }
        case SimulatedPhotonType_Type1:
        {
            l_PhotonTypeAsInt = 1;
            break;
        }
        case SimulatedPhotonType_Type2:
        {
            l_PhotonTypeAsInt  = 2;
            break;
        }
        default:
        {
            Q_ASSERT_X(0, "MainWindow::UpdateSimulatorItems()", "Unhandled SimulatorPhotonType");
            break;
        }
    }

    ui->simPhotonTypeCombo->setCurrentText(IntToHexString(l_PhotonTypeAsInt));

    bool l_Success = false;
    int l_EnergyHolder = 0;
    int l_TimeHolder = 0;

    // E1
    l_Success = SpectDMDll::GetSimulatorEnergy(SimulatorEnergyType_PhotonType0, &l_EnergyHolder);

    if(l_Success)
    {
        ui->simE1Edit->setText(IntToHexString(l_EnergyHolder));
    }

    // T1
    // use && here so if there was a previous failure then we get the error associated with this as we
    // won't make the call to get the next thing here.
    l_Success = l_Success && SpectDMDll::GetSimulatorTime(SimulatorTimeType_PhotonType0, &l_TimeHolder);

    if(l_Success)
    {
        ui->simT1Edit->setText(IntToHexString(l_TimeHolder));
    }

    // E21
    l_Success = l_Success && SpectDMDll::GetSimulatorEnergy(SimulatorEnergyType_PhotonType1_E1, &l_EnergyHolder);

    if(l_Success)
    {
        ui->simE21Edit->setText(IntToHexString(l_EnergyHolder));
    }

    // E22
    l_Success = l_Success && SpectDMDll::GetSimulatorEnergy(SimulatorEnergyType_PhotonType1_E2, &l_EnergyHolder);

    if(l_Success)
    {
        ui->simE22Edit->setText(IntToHexString(l_EnergyHolder));
    }

    // T2
    l_Success = l_Success && SpectDMDll::GetSimulatorTime(SimulatorTimeType_PhotonType1, &l_TimeHolder);

    if(l_Success)
    {
        ui->simT2Edit->setText(IntToHexString(l_TimeHolder));
    }

    // E31
    l_Success = l_Success && SpectDMDll::GetSimulatorEnergy(SimulatorEnergyType_PhotonType2_E1, &l_EnergyHolder);

    if(l_Success)
    {
        ui->simE31Edit->setText(IntToHexString(l_EnergyHolder));
    }

    // E32
    l_Success = l_Success && SpectDMDll::GetSimulatorEnergy(SimulatorEnergyType_PhotonType2_E2, &l_EnergyHolder);

    if(l_Success)
    {
        ui->simE32Edit->setText(IntToHexString(l_EnergyHolder));
    }

    // T3
    l_Success = l_Success && SpectDMDll::GetSimulatorTime(SimulatorTimeType_PhotonType2, &l_TimeHolder);

    if(l_Success)
    {
        ui->simT3Edit->setText(IntToHexString(l_TimeHolder));
    }

    // ECAT
    l_Success = l_Success && SpectDMDll::GetSimulatorEnergy(SimulatorEnergyType_Cathode, &l_EnergyHolder);

    if(l_Success)
    {
        ui->simECATEdit->setText(IntToHexString(l_EnergyHolder));
    }

    // TCAT
    l_Success = l_Success && SpectDMDll::GetSimulatorTime(SimulatorTimeType_Cathode, &l_TimeHolder);

    if(l_Success)
    {
        ui->simTCATEdit->setText(IntToHexString(l_TimeHolder));
    }
}

#endif

void MainWindow::EnablePhotonTypeTimeAndEnergy(
    int a_Type
  , bool a_Enable
)
{
    switch(a_Type)
    {
        case 0:
        {
            ui->simE1Lab->setEnabled(a_Enable);
            ui->simE1Edit->setEnabled(a_Enable);

            ui->simT1Lab->setEnabled(a_Enable);
            ui->simT1Edit->setEnabled(a_Enable);
            break;
        }
        case 1:
        {
            ui->simE21Lab->setEnabled(a_Enable);
            ui->simE21Edit->setEnabled(a_Enable);
            ui->simE22Lab->setEnabled(a_Enable);
            ui->simE22Edit->setEnabled(a_Enable);
            ui->simT2Lab->setEnabled(a_Enable);
            ui->simT2Edit->setEnabled(a_Enable);
            break;
        }
        case 2:
        {
            ui->simE31Lab->setEnabled(a_Enable);
            ui->simE31Edit->setEnabled(a_Enable);
            ui->simE32Lab->setEnabled(a_Enable);
            ui->simE32Edit->setEnabled(a_Enable);
            ui->simT3Lab->setEnabled(a_Enable);
            ui->simT3Edit->setEnabled(a_Enable);
            break;
        }
        default:
        {
            Q_ASSERT_X(0, "MainWindow::EnablePhotonTypeTimeAndEnergy", "Unsupported photon type");
            break;
        }
    }
}

void MainWindow::DisableAnalogOutputCombos()
{
    ui->DACScombo->setEnabled(false);
    ui->cathEnergyTimingCombo->setEnabled(false);
    ui->anodeChannelMonitorCombo->setEnabled(false);
}

QString MainWindow::GetCurrTabIndexHelpPage() const
{
    QString l_HelpPage;

    switch(ui->tabWidget->currentIndex())
    {
        case connectTabIndex:
        {
            l_HelpPage = "Connect.html";
            break;
        }
        case sysConfigTabIndex:
        {
            l_HelpPage = "SysConfig.html";
            break;
        }
        case GMConfigTabIndex:
        {
            l_HelpPage = "GMConfig.html";
            break;
        }
        case ASICTabIndex:
        {
            // if ASIC tab is selected, need to see what sub tab is selected.
            switch(ui->ASICSubTabWidget->currentIndex())
            {
                case ASICGlobalTabIndex:
                {
                    l_HelpPage = "ASICGlobal.html";
                    break;
                }
                case ASICAnodeTabIndex:
                {
                    l_HelpPage = "ASICAnode.html";
                    break;
                }
                case ASICCathodeTabIndex:
                {
                    l_HelpPage = "ASICCathode.html";
                    break;
                }
                default:
                {
                    Q_ASSERT_X(0, "MainWindow::keyPressEvent()", "Unhandled ASIC sub tab type");
                    break;
                }
            }
            break;
        }
        case packetsTabIndex_NoSimulator:
        case packetsTabIndex_SimulatorActive:
        {
            l_HelpPage = "Packets.html";
            break;
        }
        default:
        {
            l_HelpPage = "index.html";
            break;
        }
    }

    return l_HelpPage;
}

void MainWindow::StartHelp()
{
    // Is process running?
    if(m_HelpProcess.state() == QProcess::NotRunning)
    {
        QStringList l_Args;
        l_Args << "-collectionFile";

        #ifdef QT_DEBUG
            l_Args << "../../Package/Help/SpectDMCollection.qhc";
        #else
            l_Args << "help/SpectDMCollection.qhc";
        #endif

        l_Args << "-enableRemoteControl";

        // On XUbuntu, there is more than one assistant program and so we need to explicitly
        // define where the one we want is through the creation of an environment variable
        // called QT_DIR
        QString l_PathToAssistant = QProcessEnvironment::systemEnvironment().value("QT_DIR", "");
        l_PathToAssistant += "/assistant";

        // There is more than one assistant on XUbuntu
        m_HelpProcess.start(l_PathToAssistant, l_Args);

        if(!m_HelpProcess.waitForStarted())
        {
            QMessageBox::warning(this, "SpectDM", "Failed to find Qt Assistant to load help");
            return;
        }
    }

    // get the help page based on the active tab index
    QByteArray l_Data;
    l_Data.append("setSource qthelp://kromek.spectdm.1.0/doc/");
    l_Data.append(GetCurrTabIndexHelpPage());

    // append new line to the end of this, windows works with or without this but linux
    // didn't work without it.
    l_Data.append("\n");

    m_HelpProcess.write(l_Data);
}

void MainWindow::EnableNonConnectTabs(
    bool a_Enable
)
{
    ui->tabWidget->setTabEnabled(sysConfigTabIndex, a_Enable);
    ui->tabWidget->setTabEnabled(GMConfigTabIndex, a_Enable);
    ui->tabWidget->setTabEnabled(ASICTabIndex, a_Enable);

#ifdef INTERNAL_BUILD
    ui->tabWidget->setTabEnabled(simulatorTabIndex, a_Enable);
    ui->tabWidget->setTabEnabled(packetsTabIndex_SimulatorActive, a_Enable);
#else
    ui->tabWidget->setTabEnabled(packetsTabIndex_NoSimulator, a_Enable);
#endif
}

QString MainWindow::IntToHexString(
    int a_No
)
{
    return QString("0x%1").arg(a_No, IntToHexFieldWidth, base16, QChar('0'));
}

int MainWindow::HexStringToInt(
    const QString &a_HexString
)
{
    bool l_Ok = false;
    int l_HexStrAsInt = a_HexString.toInt(&l_Ok, base16);
    Q_ASSERT_X(l_Ok, "MainWindow::HexStringToInt()", "Failed to convert hex string to int");
    return l_HexStrAsInt;
}

void MainWindow::UpdateHGAffectedWidgets(
    bool a_HGSet
)
{
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
}

QString MainWindow::GetTestPulseStr(
    int a_PulseStep
) const
{
    float l_TestPulse = a_PulseStep * testPulseStep;
    return QString("%1mV").arg(QString::number(l_TestPulse, 'f', 2));
}

QString MainWindow::GetThresholdStr(
    int a_ThresholdStep
) const
{
    float l_Threshold = a_ThresholdStep * thresholdStep;
    return QString("%1mV").arg(QString::number(l_Threshold, 'f', 2));
}

QString MainWindow::GetThresholdTrimStr(
    int a_ThresholdTrimStep
) const
{
    double l_ThresholdTrim = thresholdTrimMin + (a_ThresholdTrimStep * thresholdTrimStep);
    return QString("%1mV").arg(QString::number(l_ThresholdTrim, 'f', 1));
}

QString MainWindow::GetPulseThresholdTrimStr(
    int a_PulseThresholdTrimStep
) const
{
    int l_PulseThresholdTrim = pulseThresholdTrimMin + (a_PulseThresholdTrimStep * pulseThresholdTrimStep);
    return QString("%1mV").arg(QString("%1").arg(l_PulseThresholdTrim));
}

QString MainWindow::GetMultiThreshDisplacementStr(
    int a_MultiThreshDispStep
) const
{
    double l_MultiThreshDisp = threshDisplacementMin + (a_MultiThreshDispStep * multiThreshDisplacementStep);
    return QString("%1mV").arg(QString::number(l_MultiThreshDisp, 'f', 2));
}

void MainWindow::EnableSingleGMWidgets(
    bool a_Enable
)
{
    ui->GMCombo->setEnabled(a_Enable);

    // loaded GM widgets
    ui->LoadedGMIDLab->setEnabled(a_Enable);
    ui->loadedGMCombo->setEnabled(a_Enable);
    ui->copyGMConfigBtn->setEnabled(a_Enable);
}

void MainWindow::UpdateGMWidgets()
{
    // update all widgets that relate to a GM
    UpdateGMItems();
    UpdateASICGlobalItems();
    UpdateASICAnodeItems();
    UpdateASICCathodeItems();
    UpdateSelectedAnodeChannelItems();
    UpdateSelectedCathodeChannelItems();
}

void MainWindow::UpdateGMPixelMapCheckbox()
{
    // this function is called when the pixel map mode is changed on the sys config tab
    // if the user has set GM-based then the GM enable pixel mapping checkbox should be enabled
    // if the user has selected global then the GM enable pixel mapping checkbox should be disabled
    // as its value is ignored here anyhow.
    ui->GMPixelMapCheck->setEnabled(ui->pixelMapGMBasedRadio->isChecked());
}

QString MainWindow::CreateTimestampedStr(
    const std::string &a_Str
)
{
    QString l_TimestampedStr = QDateTime::currentDateTime().toString(timeStampFormat);
    l_TimestampedStr += " - ";
    l_TimestampedStr += a_Str.c_str();

    return l_TimestampedStr;
}

void MainWindow::CloseProgressDlg()
{
    // closes progress dialog and deletes current instance
    if(m_ProgressDialog)
    {
        m_ProgressDialog->close();

        delete m_ProgressDialog;
        m_ProgressDialog = 0;
    }
}

void MainWindow::ReportWarning(
    const QString &a_Warning
)
{
    QMessageBox::warning(this, "SpectDM", a_Warning.toStdString().c_str());
}

void MainWindow::SetupProgressDialog(
    const QString &a_Caption
  , const QString &a_CancelText
)
{
    // this assert is untested
    // Q_ASSERT_X(!m_ProgressDialog, "MainWindow::SetupProgressDialog()", "Progess dialog already setup?");
    m_ProgressDialog = new QProgressDialog(a_Caption, a_CancelText, 0, 100, this);

    // we don't want auto hide and auto save as we handle the end of the operation and
    // we'll close before we delete this instance
    m_ProgressDialog->setAutoReset(false);
    m_ProgressDialog->setAutoClose(false);
    m_ProgressDialog->setModal(true);

    // setup signals
    // progressUpdate signal is generic and will always call onto progress dialog setValue
    connect(this, SIGNAL(progressUpdate(int)), m_ProgressDialog, SLOT(setValue(int)));
}

void MainWindow::on_writeRegButton_clicked()
{
    QString l_RegNum = ui->sysControlRegNoCombo->currentText();
    QString l_MSBStr = ui->MSBEdit->text();
    QString l_LSBStr = ui->LSBEdit->text();

    if(l_RegNum.isEmpty() || l_MSBStr.isEmpty() || l_LSBStr.isEmpty())
    {
        QMessageBox::warning(this, "SpectDM", "Reg No, MSB and LSB must be set");
    }
    else
    {
        // we could check vals are between 0 and 255 - this would be users resp
        int l_MSB = l_MSBStr.toInt();
        int l_LSB = l_LSBStr.toInt();

        if(l_MSB >= minRegDirectVal && l_MSB <= maxRegDirectVal)
        {
            if(l_LSB >= minRegDirectVal && l_LSB <= maxRegDirectVal)
            {
                if(!SpectDMDll::RegWrite(l_RegNum.toInt(), l_MSB, l_LSB))
                {
                    QMessageBox::warning(this, "SpectDM", SpectDMDll::GetLastError().c_str());
                }
            }
            else
            {
                QMessageBox::warning(this, "SpectDM"
                                   , QString("LSB must be between %1 and %2").arg(minRegDirectVal)
                                                                             .arg(maxRegDirectVal));
            }
        }
        else
        {
            QMessageBox::warning(this, "SpectDM"
                               , QString("MSB must be between %1 and %2").arg(minRegDirectVal)
                                                                         .arg(maxRegDirectVal));
        }


    }
}

void MainWindow::on_readRegButton_clicked()
{
    QString l_RegNum = ui->sysControlRegNoCombo->currentText();

    if(l_RegNum.isEmpty())
    {
        QMessageBox::warning(this, "SpectDM", "No Reg No set");
    }
    else
    {
        unsigned char l_MSB = 0;
        unsigned char l_LSB = 0;

        if(SpectDMDll::RegRead(l_RegNum.toInt(), &l_MSB, &l_LSB))
        {
            ui->MSBEdit->setText(QString("%1").arg(static_cast<int>(l_MSB)));
            ui->LSBEdit->setText(QString("%1").arg(static_cast<int>(l_LSB)));
        }
        else
        {
            QMessageBox::warning(this, "SpectDM", SpectDMDll::GetLastError().c_str());
        }
    }
}

void MainWindow::on_gmRegReadButton_clicked()
{
    QString l_GMReg = ui->gmRegNoCombo->currentText();

    if(l_GMReg.isEmpty())
    {
        QMessageBox::warning(this, "SpectDM", "No GM Reg num");
    }
    else
    {
        unsigned char l_Data = 0;
        if(SpectDMDll::GMRegRead(l_GMReg.toInt(), &l_Data))
        {
            ui->gmDataEdit->setText(QString("%1").arg(static_cast<int>(l_Data)));
        }
        else
        {
            QMessageBox::warning(this, "SpectDM", SpectDMDll::GetLastError().c_str());
        }
    }
}

void MainWindow::on_gmWriteRegButton_clicked()
{
    QString l_GMReg = ui->gmRegNoCombo->currentText();
    QString l_GMData = ui->gmDataEdit->text();

    if(l_GMReg.isEmpty() || l_GMData.isEmpty())
    {
        QMessageBox::warning(this, "SpectDM", "Set GM reg and data to write to register");
    }
    else
    {
        if(!SpectDMDll::GMRegWrite(l_GMReg.toInt(), l_GMData.toInt()))
        {
            QMessageBox::warning(this, "SpectDM", SpectDMDll::GetLastError().c_str());
        }
    }
}

//void MainWindow::on_saveConfigBtn_clicked()
//{
//    QFileDialog l_SaveDialog(this, "Save Config File", "../", ".xml");
//    l_SaveDialog.setAcceptMode(QFileDialog::AcceptSave);
//    if(l_SaveDialog.exec() == QFileDialog::Accepted)
//    {
//        // check filename isn't blank?
//        l_SaveDialog.getSaveFileName()
//    }
//    if(!SpectDMDll::SaveSysConfig("SysConfig_15_04_14.xml"))
//    {
//        QMessageBox::warning(this, "SpectDM", SpectDMDll::GetLastError().c_str());
//    }
//}

//void MainWindow::on_loadSysConfigBtn_clicked()
//{
//    if(SpectDMDll::LoadSysConfig("SysConfig_15_04_14.xml"))
//    {
//        ui->mbMultiplexEdit->setText(QString("%1").arg(SpectDMDll::GetMBMultiplexerAddress()));
//        ui->pktTransferSpinner->setValue(SpectDMDll::GetPacketTransferRate());
//        //ui->biasVoltageEdit->setText(QString("%1").arg(SpectDMDll::Get));
//        // need a GetBiasVoltage

//        SpectDMDll::
//        ui->debugModeCheckBox
//        ui->gm3TokenCheck
//        // need some sets for the above
//    }
//    else
//    {
//        QMessageBox::warning(this, "SpectDM", SpectDMDll::GetLastError().c_str());
//    }
//}

void MainWindow::on_readNoOfPacketsBtn_clicked()
{
    ui->noOfPacketsEdit->setText(QString("%1").arg(SpectDMDll::GetNoOfCollectedPackets()));
}

void MainWindow::on_readPacketBtn_clicked()
{
    QString l_PacketNo = ui->packetNoEdit->text();

    if(!l_PacketNo.isEmpty())
    {
        bool l_Success = true;
        int l_GMNo = 0;

        // we are assuming user has entered a number here
        if(SpectDMDll::GetPacketData(l_PacketNo.toInt(), PacketData_GMNo, &l_GMNo))
        {
            ui->packetGMEdit->setText(QString("%1").arg(l_GMNo));
        }
        else
        {
            l_Success = false;
        }

        if(l_Success)
        {
            int l_TimeStamp = 0;

            if(SpectDMDll::GetPacketData(l_PacketNo.toInt(), PacketData_Timestamp, &l_TimeStamp))
            {
                ui->packetTimestampEdit->setText(QString("%1").arg(l_TimeStamp));
            }
            else
            {
                l_Success = false;
            }
        }

        if(l_Success)
        {
            int l_PhotonCount = 0;

            if(SpectDMDll::GetPacketData(l_PacketNo.toInt(), PacketData_PhotonCount, &l_PhotonCount))
            {
                ui->packetPhotonCountEdit->setText(QString("%1").arg(l_PhotonCount));
            }
            else
            {
                l_Success = false;
            }
        }

        if(!l_Success)
        {
            QMessageBox::warning(this, "SpectDM", SpectDMDll::GetLastError().c_str());
        }
    }
    else
    {
        QMessageBox::warning(this, "SpectDM", "No packet number specified");
    }
}

void MainWindow::on_readPhotonBtn_clicked()
{
    QString l_PacketNo = ui->packetNoEdit->text();
    QString l_PhotonNo = ui->photonNoEdit->text();

    if(l_PacketNo.isEmpty() || l_PhotonNo.isEmpty())
    {
        QMessageBox::warning(this, "SpectDM", "Action requires packet no and photon no to be set");
    }
    else
    {
        bool l_Success = true;
        int l_PhotonCoord = 0;

        if(SpectDMDll::GetPhotonData(l_PacketNo.toInt(), l_PhotonNo.toInt(), PhotonData_Coordinate, &l_PhotonCoord))
        {
            ui->photonCoordEdit->setText(QString("%1").arg(l_PhotonCoord));
        }
        else
        {
            l_Success = false;
        }

        if(l_Success)
        {
            int l_PhotonEnergy = 0;

            if(SpectDMDll::GetPhotonData(l_PacketNo.toInt(), l_PhotonNo.toInt(), PhotonData_Energy, &l_PhotonEnergy))
            {
                ui->photonEnergyEdit->setText(QString("%1").arg(l_PhotonEnergy));
            }
            else
            {
                l_Success = false;
            }
        }

        if(l_Success)
        {
            int l_PhotonEnergyEvent = 0;

            if(SpectDMDll::GetPhotonData(l_PacketNo.toInt(), l_PhotonNo.toInt(), PhotonData_EnergyPosEvent, &l_PhotonEnergyEvent))
            {
                if(l_PhotonEnergyEvent == 0)
                {
                    ui->photonEnergyNegEventRadio->setChecked(true);
                }
                else
                {
                    ui->photonEnergyPosEventRadio->setChecked(true);
                }
            }
            else
            {
                l_Success = false;
            }
        }

        if(l_Success)
        {
            int l_PhotonTimeDetect = 0;

            if(SpectDMDll::GetPhotonData(l_PacketNo.toInt(), l_PhotonNo.toInt(), PhotonData_TimeDetect, &l_PhotonTimeDetect))
            {
                ui->photonTimeDetectEdit->setText(QString("%1").arg(l_PhotonTimeDetect));
            }
            else
            {
                l_Success = false;
            }
        }

        if(l_Success)
        {
            int l_PhotonTimeDetectEvent = 0;

            if(SpectDMDll::GetPhotonData(l_PacketNo.toInt(), l_PhotonNo.toInt(), PhotonData_TimeDetectPosEvent, &l_PhotonTimeDetectEvent))
            {
                if(l_PhotonTimeDetectEvent == 0)
                {
                    ui->photonTimeDetectNegEventRadio->setChecked(true);
                }
                else
                {
                    ui->photonTimeDetectPosEventRadio->setChecked(true);
                }
            }
            else
            {
                l_Success = false;
            }
        }

        if(!l_Success)
        {
             QMessageBox::warning(this, "SpectDM", SpectDMDll::GetLastError().c_str());
        }
    }
}

void MainWindow::on_startCollectBtn_clicked()
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

void MainWindow::on_stopCollectBtn_clicked()
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

void MainWindow::on_GMStatusBtn_clicked()
{
    // just for test purposes
    GMStatus l_Status = GMStatus_Undefined;

    if(SpectDMDll::GetGMStatus(&l_Status))
    {
        if(l_Status & GMStatus_Idle)
        {
            QFont l_Font = ui->IdleLab->font();
            l_Font.setBold(true);
            ui->IdleLab->setFont(l_Font);
        }

        if(l_Status & GMStatus_ASICLoadError)
        {
            QFont l_Font = ui->IdleLab->font();
            l_Font.setBold(true);
            ui->IdleLab->setFont(l_Font);
        }

        if(l_Status & GMStatus_FIFOFull)
        {
            QFont l_Font = ui->FIFOFullLab->font();
            l_Font.setBold(true);
            ui->FIFOFullLab->setFont(l_Font);
        }
    }
}

void MainWindow::on_ADC1ReadButton_clicked()
{
    int l_ADC1DACS = 0;
    int l_ADC1Volts = 0;
    if(SpectDMDll::ReadGMADC1(&l_ADC1DACS, &l_ADC1Volts))
    {
        ui->ADC1DACSEdit->setText(QString("%1").arg(l_ADC1DACS));
        ui->ADC1VoltsEdit->setText(QString("%1").arg(l_ADC1Volts));
    }
    else
    {
        QMessageBox::warning(this, "SpectDM", SpectDMDll::GetLastError().c_str());
    }
}

void MainWindow::on_ADC2ReadBtn_clicked()
{
    int l_ADC2DACS = 0;
    int l_ADC2Volts = 0;
    if(SpectDMDll::ReadGMADC2(&l_ADC2DACS, &l_ADC2Volts))
    {
        ui->ADC2DACSEdit->setText(QString("%1").arg(l_ADC2DACS));
        ui->ADC2VoltsEdit->setText(QString("%1").arg(l_ADC2Volts));
    }
    else
    {
        QMessageBox::warning(this, "SpectDM", SpectDMDll::GetLastError().c_str());
    }
}

void MainWindow::on_loadFromFileBtn_clicked()
{
    QString l_LocalIP = ui->hostIPEdit->text();

    if(!l_LocalIP.isEmpty())
    {
        SpectDMDll::SetHostIPAddress(l_LocalIP.toStdString().c_str());

        QString l_FileName = QFileDialog::getOpenFileName(this
                                                        , "Load Config"
                                                        , QString()
                                                        , "Binary files (*.dat)");

        if(!l_FileName.isEmpty())
        {
            if(SpectDMDll::LoadConfiguration(l_FileName.toStdString().c_str()))
            {
                ui->cameraIPEdit->setText(QString(SpectDMDll::GetCameraIPAddress().c_str()));
                EnableNonConnectTabs(true);
                UpdateSysConfigItems();
                ui->connectButton->setEnabled(false);
                ui->disconnectButton->setEnabled(true);
                ui->loadFromFileBtn->setEnabled(false);
                ui->startSysBtn->setEnabled(false);
                ui->stopSysBtn->setEnabled(true);
                ui->startCollectBtn->setEnabled(true);
                m_Connected = true;
            }
            else
            {
                QMessageBox::warning(this, "SpectDM", SpectDMDll::GetLastError().c_str());
            }
        }
    }
    else
    {
        QMessageBox::warning(this, "SpectDM", "Local ip must be entered for load config");
    }
}

void MainWindow::on_ReadGBEFirmwareBtn_clicked()
{
    int l_GBEFirmwareVer = 0;

    if(SpectDMDll::GetGBEFirmwareVersion(&l_GBEFirmwareVer))
    {
        ui->GBEFirmwareVerEdit->setText(QString("%1").arg(l_GBEFirmwareVer));
    }
    else
    {
        QMessageBox::warning(this, "SpectDM", SpectDMDll::GetLastError().c_str());
    }
}

void MainWindow::on_ReadGMFirmwareBtn_clicked()
{
    int l_GMFirmwareVer = 0;

    if(SpectDMDll::GetGMFirmwareVersion(&l_GMFirmwareVer))
    {
        ui->GMFirmwareVerEdit->setText(QString("%1").arg(l_GMFirmwareVer));
    }
    else
    {
        QMessageBox::warning(this, "SpectDM", SpectDMDll::GetLastError().c_str());
    }
}

void MainWindow::on_startSysBtn_clicked()
{
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

void MainWindow::on_stopSysBtn_clicked()
{
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

void MainWindow::on_copyGMConfigBtn_clicked()
{
    QString l_LoadedGMIDStr = ui->loadedGMCombo->currentText();

    if(l_LoadedGMIDStr != "-")
    {
        if(SpectDMDll::CopyLoadedGMToGM(l_LoadedGMIDStr.toInt(), SpectDMDll::GetActiveGM()))
        {
            UpdateGMWidgets();
        }
        else
        {
            QMessageBox::warning(this, "SpectDM", SpectDMDll::GetLastError().c_str());
        }
    }
    else
    {
        QMessageBox::warning(this, "SpectDM", "No GM specified to load from");
    }
}

void MainWindow::on_anodeChannelRadio_toggled(bool checked)
{
    ui->anodeChannelMonitorCombo->setEnabled(checked);
}

void MainWindow::on_resetPacketTextBtn_clicked()
{
    ui->packetsText->clear();
    ui->packetLoadProgressBar->setValue(0);
}

void MainWindow::on_readAllPacketsBtn_clicked()
{
    int l_NoOfPackets = SpectDMDll::GetNoOfCollectedPackets();

    if(l_NoOfPackets > 0)
    {
        ui->packetsText->append(QString("No of collected packets: %1\n").arg(l_NoOfPackets));
        for(int i = 0; i < l_NoOfPackets; i++)
        {
            QString l_PacketStr;
            double l_Progress = static_cast<double>(i) / l_NoOfPackets;
            l_Progress *= 100;
            ui->packetLoadProgressBar->setValue(static_cast<int>(l_Progress));

            int l_PacketNo = i + 1;
            l_PacketStr += QString("\nPacket %1\n").arg(l_PacketNo);

            int l_GMNo = 0;
            if(SpectDMDll::GetPacketData(l_PacketNo, PacketData_GMNo, &l_GMNo))
            {
                l_PacketStr += QString("GM No: %1\n").arg(l_GMNo);
            }

            int l_Timestamp = 0;
            if(SpectDMDll::GetPacketData(l_PacketNo, PacketData_Timestamp, &l_Timestamp))
            {
                l_PacketStr += QString("Timestamp: %1\n").arg(l_Timestamp);
            }

            int l_PhotonCount = 0;
            if(SpectDMDll::GetPacketData(l_PacketNo, PacketData_PhotonCount, &l_PhotonCount))
            {
                l_PacketStr += QString("Photon Count: %1\n").arg(l_PhotonCount);
            }

            int l_ActualPhotonCount = 0;
            if(SpectDMDll::GetPacketData(l_PacketNo, PacketData_ActualPhotonCount, &l_ActualPhotonCount))
            {
                l_PacketStr += QString("Actual Photon Count: %1\n").arg(l_ActualPhotonCount);
            }

            for(int j = 0; j < l_ActualPhotonCount; j++)
            {
                int l_PhotonNo = j + 1;

                l_PacketStr += QString("\n\tPhoton %1\n").arg(l_PhotonNo);

                int l_Coord = 0;
                if(SpectDMDll::GetPhotonData(l_PacketNo, l_PhotonNo, PhotonData_Coordinate, &l_Coord))
                {
                    l_PacketStr += QString("\tCoordinate: %1\n").arg(l_Coord);
                }

                int l_Energy = 0;
                if(SpectDMDll::GetPhotonData(l_PacketNo, l_PhotonNo, PhotonData_Energy, &l_Energy))
                {
                    l_PacketStr += QString("\tEnergy: %1\n").arg(l_Energy);
                }

                int l_PosEnergy = 0;
                if(SpectDMDll::GetPhotonData(l_PacketNo, l_PhotonNo, PhotonData_EnergyPosEvent, &l_PosEnergy))
                {
                    l_PacketStr += QString("\tPositive Energy: %1\n").arg(l_PosEnergy);
                }

                int l_TimeDetect = 0;
                if(SpectDMDll::GetPhotonData(l_PacketNo, l_PhotonNo, PhotonData_TimeDetect, &l_TimeDetect))
                {
                    l_PacketStr += QString("\tTime Detect: %1\n").arg(l_TimeDetect);
                }

                int l_PosTimeDetect = 0;
                if(SpectDMDll::GetPhotonData(l_PacketNo, l_PhotonNo, PhotonData_TimeDetectPosEvent, &l_PosTimeDetect))
                {
                    l_PacketStr += QString("\tPositive Time Detect: %1\n").arg(l_PosTimeDetect);
                }
            }

            ui->packetsText->append(l_PacketStr);
        }

        ui->packetLoadProgressBar->setValue(100);
    }
    else
    {
        ui->packetsText->append("No collected packets to display");
    }
}

void MainWindow::on_SaveAction()
{
    QString l_FileName = QFileDialog::getSaveFileName(this
                                                    , "Save Config"
                                                    , QString()
                                                    , "Binary files (*.dat)");

    if(!l_FileName.isEmpty())
    {
        if(!SpectDMDll::SaveConfiguration(l_FileName.toStdString().c_str()))
        {
            QMessageBox::warning(this, "SpectDM", SpectDMDll::GetLastError().c_str());
        }
    }
}

#ifdef INTERNAL_BUILD

void MainWindow::on_simCycleCheck_clicked(
    bool checked
)
{
    if(checked)
    {
        ui->simNoOfPhotonsCombo->setEnabled(false);
        ui->simPhotonTypeCombo->setEnabled(false);

        EnablePhotonTypeTimeAndEnergy(0, true);
        EnablePhotonTypeTimeAndEnergy(1, true);
        EnablePhotonTypeTimeAndEnergy(2, true);
    }
    else
    {
        ui->simNoOfPhotonsCombo->setEnabled(true);
        ui->simPhotonTypeCombo->setEnabled(true);

        EnablePhotonTypeTimeAndEnergy(0, false);
        EnablePhotonTypeTimeAndEnergy(1, false);
        EnablePhotonTypeTimeAndEnergy(2, false);

        switch(ui->simPhotonTypeCombo->currentText().toInt())
        {
            case 0:
            {
                EnablePhotonTypeTimeAndEnergy(0, true);
                break;
            }
            case 1:
            {
                EnablePhotonTypeTimeAndEnergy(1, true);
                break;
            }
            case 2:
            {
                EnablePhotonTypeTimeAndEnergy(2, true);
                break;
            }
            default:
            {
                Q_ASSERT_X(0, "MainWindow::on_simCycleCheck_clicked()", "Invalid photon type");
            }
        }
    }
}

void MainWindow::on_startSimBtn_clicked()
{
    bool l_Success = true;
    QString l_Err;

    // check options
    SimulatorOptions l_Options = SimulatorOption_None;

    if(ui->simTimeDataCheck->isChecked())
    {
        l_Options |= SimulatorOption_EnableTimeData;
    }

    if(ui->simNegSignalData->isChecked())
    {
        l_Options |= SimulatorOption_EnableNegativeSignalData;
    }

    if(ui->simCycleCheck->isChecked())
    {
        l_Options |= SimulatorOption_Cycle;
    }

    if(ui->simCathodeGroup->isChecked())
    {
        l_Options |= SimulatorOption_EnableCathodeData;
    }

    SpectDMDll::SetSimulatorOptions(l_Options);

    SimulatorPacketFrequency l_PktFreq = SimulatorPacketFrequency_Undefined;

    if(ui->simPktFreq_100Hz_radio->isChecked())
    {
        l_PktFreq = SimulatorPacketFrequency_100Hz;
    }
    else if(ui->simPktFreq_1kHz_radio->isChecked())
    {
        l_PktFreq = SimulatorPacketFrequency_1kHz;
    }
    else if(ui->simPktFreq_10kHz_radio->isChecked())
    {
        l_PktFreq = SimulatorPacketFrequency_10kHz;
    }
    else if(ui->simPktFreq_100kHz_radio->isChecked())
    {
        l_PktFreq = SimulatorPacketFrequency_100kHz;
    }
    else
    {
        Q_ASSERT_X(0, "MainWindow::on_startSimBtn_clicked()", "No pkt frequency set");
    }

    if(!SpectDMDll::SetSimulatorPacketFrequency(l_PktFreq))
    {
        l_Success = false;
    }

    // process number of photons if there hasn't been an error AND cycle isn't enabled
    if(l_Success && ((l_Options & SimulatorOption_Cycle) != SimulatorOption_Cycle))
    {
        SpectDMDll::SetSimulatorPhotonsPerPacket(ui->simNoOfPhotonsCombo->currentText().toInt());
    }

    // process GMNo
    if(l_Success)
    {
        if(!SpectDMDll::SetSimulatorGMNo(ui->simGMNoCombo->currentText().toInt()))
        {
            l_Success = false;
        }
    }

    // process number of packets
    if(l_Success)
    {
        QString l_NoOfPkts = ui->simNoOfPktsEdit->text();

        if(l_NoOfPkts.isEmpty())
        {
            l_Success = false;
            l_Err = "No of packets has not been set";
        }
        else
        {
            SpectDMDll::SetSimulatorNoOfPackets(HexStringToInt(l_NoOfPkts));
        }
    }

    // process time offset
    if(l_Success)
    {
        QString l_TimeOffset = ui->simTimeOffsetEdit->text();

        if(l_TimeOffset.isEmpty())
        {
            l_Success = false;
            l_Err = "Time offset has not been set";
        }
        else
        {
            if(!SpectDMDll::SetSimulatorTimeOffset(HexStringToInt(l_TimeOffset)))
            {
                l_Success = false;
            }
        }
    }

    // process photon type - check if we are cycling to see if we need to set it
    if(l_Success && ((l_Options & SimulatorOption_Cycle) != SimulatorOption_Cycle))
    {
        // SimulatorPhotonType needed here
        int l_PhotonTypeInt = ui->simPhotonTypeCombo->currentText().toInt();

        // Don't cast to SimulatorPhotonType, 0 is undefined!
        SimulatorPhotonType l_PhotonType = SimulatorPhotonType_Undefined;

        switch(l_PhotonTypeInt)
        {
            case 0:
            {
                l_PhotonType = SimulatedPhotonType_Type0;
                break;
            }
            case 1:
            {
                l_PhotonType = SimulatedPhotonType_Type1;
                break;
            }
            case 2:
            {
                l_PhotonType = SimulatedPhotonType_Type2;
                break;
            }
            default:
            {
                Q_ASSERT_X(0, "OnSimStart()", "Invalid photon type");
                break;
            }
        }

        if(!SpectDMDll::SetSimulatorPhotonType(l_PhotonType))
        {
            l_Success = false;
        }
    }

    // now for the energies and times
    // need to check cycling here to see what we have.
    if(l_Success)
    {
        if((l_Options & SimulatorOption_Cycle) || (SpectDMDll::GetSimulatorPhotonType() == SimulatedPhotonType_Type0))
        {
            QString l_E1 = ui->simE1Edit->text();

            if(l_E1.isEmpty())
            {
                l_Success = false;
                l_Err = "E1 has not been set";
            }
            else
            {
                if(!SpectDMDll::SetSimulatorEnergy(SimulatorEnergyType_PhotonType0, HexStringToInt(l_E1)))
                {
                    l_Success = false;
                }
            }

            if(l_Success)
            {
                QString l_T1 = ui->simT1Edit->text();

                if(l_T1.isEmpty())
                {
                    l_Success = false;
                    l_Err = "T1 has not been set";
                }
                else
                {
                    if(!SpectDMDll::SetSimulatorTime(SimulatorTimeType_PhotonType0, HexStringToInt(l_T1)))
                    {
                        l_Success = false;
                    }
                }
            }
        }

        if(l_Success
        && ((l_Options & SimulatorOption_Cycle) || (SpectDMDll::GetSimulatorPhotonType() == SimulatedPhotonType_Type1)))
        {
            QString l_E21 = ui->simE21Edit->text();

            if(l_E21.isEmpty())
            {
                l_Success = false;
                l_Err = "E21 has not been set";
            }
            else
            {
                if(!SpectDMDll::SetSimulatorEnergy(SimulatorEnergyType_PhotonType1_E1, HexStringToInt(l_E21)))
                {
                    l_Success = false;
                }
            }

            if(l_Success)
            {
                QString l_E22 = ui->simE22Edit->text();

                if(l_E22.isEmpty())
                {
                    l_Success = false;
                    l_Err = "E22 has not been set";
                }
                else
                {
                    if(!SpectDMDll::SetSimulatorEnergy(SimulatorEnergyType_PhotonType1_E2, HexStringToInt(l_E22)))
                    {
                        l_Success = false;
                    }
                }

                if(l_Success)
                {
                    QString l_T2 = ui->simT2Edit->text();

                    if(l_T2.isEmpty())
                    {
                        l_Success = false;
                        l_Err = "T2 has not been set";
                    }
                    else
                    {
                        if(!SpectDMDll::SetSimulatorTime(SimulatorTimeType_PhotonType1, HexStringToInt(l_T2)))
                        {
                            l_Success = false;
                        }
                    }
                }
            }
        }

        if(l_Success
        && ((l_Options & SimulatorOption_Cycle) || (SpectDMDll::GetSimulatorPhotonType() == SimulatedPhotonType_Type2)))
        {
            QString l_E31 = ui->simE31Edit->text();

            if(l_E31.isEmpty())
            {
                l_Success = false;
                l_Err = "E31 has not been set";
            }
            else
            {
                if(!SpectDMDll::SetSimulatorEnergy(SimulatorEnergyType_PhotonType2_E1, HexStringToInt(l_E31)))
                {
                    l_Success = false;
                }
            }

            if(l_Success)
            {
                QString l_E32 = ui->simE32Edit->text();

                if(l_E32.isEmpty())
                {
                    l_Success = false;
                    l_Err = "E32 has not been set";
                }
                else
                {
                    if(!SpectDMDll::SetSimulatorEnergy(SimulatorEnergyType_PhotonType2_E2, HexStringToInt(l_E32)))
                    {
                        l_Success = false;
                    }
                }

                if(l_Success)
                {
                    QString l_T3 = ui->simT3Edit->text();

                    if(l_T3.isEmpty())
                    {
                        l_Success = false;
                        l_Err = "T3 has not been set";
                    }
                    else
                    {
                        if(!SpectDMDll::SetSimulatorTime(SimulatorTimeType_PhotonType2, HexStringToInt(l_T3)))
                        {
                            l_Success = false;
                        }
                    }
                }
            }
        }
    }

    if(l_Success)
    {
        // process cathode data, if cathode group is enabled
        if(ui->simCathodeGroup->isChecked())
        {
            QString l_ECAT = ui->simECATEdit->text();

            if(l_ECAT.isEmpty())
            {
                l_Success = false;
                l_Err = "ECAT has not been set";
            }
            else
            {
                l_Success = SpectDMDll::SetSimulatorEnergy(SimulatorEnergyType_Cathode, HexStringToInt(l_ECAT));
            }

            if(l_Success)
            {
                QString l_TCAT = ui->simTCATEdit->text();

                if(l_TCAT.isEmpty())
                {
                    l_Success = false;
                    l_Err = "TCAT has not been set";
                }
                else
                {
                    l_Success = SpectDMDll::SetSimulatorTime(SimulatorTimeType_Cathode, HexStringToInt(l_TCAT));
                }
            }
        }
    }

    // finally
    if(l_Success)
    {
        // returns bool?
        SpectDMDll::StartSimulator();
        ui->startSimBtn->setEnabled(false);
        ui->stopSimBtn->setEnabled(true);
    }
    else
    {
        QString l_ErrMsg = l_Err.isEmpty() ? SpectDMDll::GetLastError().c_str() : l_Err.toStdString().c_str();
        QMessageBox::warning(this, "SpectDM", l_ErrMsg);
    }
}

void MainWindow::on_simPhotonTypeCombo_currentTextChanged(
    const QString &arg1
)
{
    switch(arg1.toInt())
    {
        case 0:
        {
            EnablePhotonTypeTimeAndEnergy(0, true);
            EnablePhotonTypeTimeAndEnergy(1, false);
            EnablePhotonTypeTimeAndEnergy(2, false);
            break;
        }
        case 1:
        {
            EnablePhotonTypeTimeAndEnergy(0, false);
            EnablePhotonTypeTimeAndEnergy(1, true);
            EnablePhotonTypeTimeAndEnergy(2, false);
            break;
        }
        case 2:
        {
            EnablePhotonTypeTimeAndEnergy(0, false);
            EnablePhotonTypeTimeAndEnergy(1, false);
            EnablePhotonTypeTimeAndEnergy(2, true);
            break;
        }
        default:
        {
            Q_ASSERT_X(0, "MainWindow::on_simPhotonTypeCombo_currentTextChanged()", "Invalid photon type");
            break;
        }
    }
}

void MainWindow::on_stopSimBtn_clicked()
{
    SpectDMDll::StopSimulator();
    ui->startSimBtn->setEnabled(true);
    ui->stopSimBtn->setEnabled(false);
}

void MainWindow::on_simRegReadButton_clicked()
{
    unsigned char l_MSB;
    unsigned char l_LSB;

    if(SpectDMDll::SimulatorRegRead(ui->simRegNoCombo->currentText().toInt(), &l_MSB, &l_LSB))
    {
        ui->simRegMSBEdit->setText(IntToHexString(l_MSB));
        ui->simRegLSBEdit->setText(IntToHexString(l_LSB));
    }
    else
    {
        QMessageBox::warning(this, "SpectDM", SpectDMDll::GetLastError().c_str());
    }
}

void MainWindow::on_simRegWriteButton_clicked()
{
    int l_MSB = HexStringToInt(ui->simRegMSBEdit->text());
    int l_LSB = HexStringToInt(ui->simRegLSBEdit->text());

    if(!SpectDMDll::SimulatorRegWrite(ui->simRegNoCombo->currentText().toInt(), l_MSB, l_LSB))
    {
        QMessageBox::warning(this, "SpectDM", SpectDMDll::GetLastError().c_str());
    }
}

#endif

void MainWindow::on_actionAbout_triggered()
{
    QString l_AboutText;

    QString l_FileName;

#ifdef QT_DEBUG
    l_FileName = "../../Package/packageversion.txt";
#else
    l_FileName = "packageversion.txt";
#endif

    QFile l_PkgVersionFile(l_FileName);

    if(l_PkgVersionFile.open(QFile::ReadOnly))
    {
        QString l_Version = l_PkgVersionFile.readLine();

        l_AboutText = QString("Package version\t: %1\n").arg(l_Version);

        l_PkgVersionFile.close();
    }

    l_AboutText += QString("API version\t\t: %1\n").arg(SpectDMDll::GetVersion().c_str());
    l_AboutText += QString("Test App version\t: %1").arg(APP_VERSION);

    QMessageBox::about(this, "About SpectDM", l_AboutText);
}

void MainWindow::on_allCathodesRadio_clicked()
{
    ui->cathodeChannelNoCombo->setEnabled(false);
}

void MainWindow::on_allAnodesRadio_clicked()
{
    ui->anodeChannelNoCombo->setEnabled(false);
}

void MainWindow::on_individualAnodeRadio_clicked()
{
    ui->anodeChannelNoCombo->setEnabled(true);

    if(SpectDMDll::IsActiveGMSet())
    {
        UpdateSelectedAnodeChannelItems();
    }
}

void MainWindow::on_individualCathodeRadio_clicked()
{
    ui->cathodeChannelNoCombo->setEnabled(true);

    if(SpectDMDll::IsActiveGMSet())
    {
        UpdateSelectedCathodeChannelItems();
    }
}

void MainWindow::on_noFuncRadio_clicked()
{
    DisableAnalogOutputCombos();
}

void MainWindow::on_baselineRadio_clicked()
{
    DisableAnalogOutputCombos();
}

void MainWindow::on_temperatureRadio_clicked()
{
    DisableAnalogOutputCombos();
}

void MainWindow::on_DACSradio_clicked()
{
    // disable all combos then enable the one we want
    DisableAnalogOutputCombos();
    ui->DACScombo->setEnabled(true);
}

void MainWindow::on_cathodeEnergyTimingRadio_clicked()
{
    DisableAnalogOutputCombos();
    ui->cathEnergyTimingCombo->setEnabled(true);
}

void MainWindow::on_anodeChannelRadio_clicked()
{
    DisableAnalogOutputCombos();
    ui->anodeChannelMonitorCombo->setEnabled(true);
}

void MainWindow::on_actionSpect_DM_Help_triggered()
{
    StartHelp();
}

void MainWindow::on_actionConfiguration_triggered()
{
    QFileDialog l_SaveConfigDlg(this, "Save Configuration", QString(), "Binary files (*.dat)");
    l_SaveConfigDlg.setAcceptMode(QFileDialog::AcceptSave);

    if(l_SaveConfigDlg.exec())
    {
        QStringList l_SelectedFiles = l_SaveConfigDlg.selectedFiles();

        if(!l_SelectedFiles.isEmpty())
        {
            if(!SpectDMDll::SaveConfiguration(l_SelectedFiles.at(0).toStdString().c_str()))
            {
                QMessageBox::warning(this, "SpectDM", SpectDMDll::GetLastError().c_str());
            }

            l_SaveConfigDlg.saveState();
        }
    }
}

void MainWindow::on_actionPackets_triggered()
{
    QFileDialog l_SavePacketsDlg(this, "Save Packets", QString(), "CSV files (*.csv)");
    l_SavePacketsDlg.setAcceptMode(QFileDialog::AcceptSave);

    if(l_SavePacketsDlg.exec())
    {
        QStringList l_SelectedFiles = l_SavePacketsDlg.selectedFiles();

        if(!l_SelectedFiles.isEmpty())
        {
            if(SpectDMDll::SaveCollection(l_SelectedFiles.at(0).toStdString().c_str()))
            {
                SetupProgressDialog("Saving Packets...", "Cancel");

                // prior to executing, setup any operation specific callbacks and signal/slots here
                // for instance, on cancelling a save packets we want to delete the file.

                connect(m_ProgressDialog, SIGNAL(canceled()), this, SLOT(OnPacketSaveOperationCancelled()));

                // launch the progress dialog
                m_ProgressDialog->exec();
            }
            else
            {
                QMessageBox::warning(this, "SpectDM", SpectDMDll::GetLastError().c_str());
            }

            // save where the file was saved to for so dialog is opened at this dir
            // next time
            l_SavePacketsDlg.saveState();
        }
    }
}

void MainWindow::on_anodeTestPulseSpinner_valueChanged(int arg1)
{
    ui->anodeTestPulseValEdit->setText(GetTestPulseStr(arg1));
}

void MainWindow::on_cathodeTestPulseSpinner_valueChanged(int arg1)
{
    ui->cathodeTestPulseValEdit->setText(GetTestPulseStr(arg1));
}

void MainWindow::on_anodePosEnergyThreshSpinner_valueChanged(int arg1)
{
    ui->anodePosEnergyThreshValEdit->setText(GetThresholdStr(arg1));
}

void MainWindow::on_anodeNegEnergyThreshSpinner_valueChanged(int arg1)
{
    ui->anodeNegEnergyThreshValEdit->setText(GetThresholdStr(arg1));
}

void MainWindow::on_cathodeEnergyThreshSpinner_valueChanged(int arg1)
{
    ui->cathodeEnergyThreshValEdit->setText(GetThresholdStr(arg1));
}

void MainWindow::on_cathodeTimingPrimaryThreshSpinner_valueChanged(int arg1)
{
    ui->cathodeTimingPrimaryThreshValEdit->setText(GetThresholdStr(arg1));
}

void MainWindow::on_cathodeTimingUnipolarThreshSpinner_valueChanged(int arg1)
{
    ui->cathodeTimingUnipolarThreshValEdit->setText(GetThresholdStr(arg1));
}

void MainWindow::on_cathodeChnlUnipolarTrimSpinner_valueChanged(int arg1)
{
    ui->cathodeChnlUnipolarTrimValEdit->setText(GetThresholdTrimStr(arg1));
}

void MainWindow::on_cathodeChnlFirstMultiTrimSpinner_valueChanged(int arg1)
{
    ui->cathodeChnlFirstMultiTrimValEdit->setText(GetThresholdTrimStr(arg1));
}

void MainWindow::on_cathodeChnlSecondMultiTrimSpinner_valueChanged(int arg1)
{
    ui->cathodeChnlSecondMultiTrimValEdit->setText(GetThresholdTrimStr(arg1));
}

void MainWindow::on_cathodeChnlThirdMultiTrimSpinner_valueChanged(int arg1)
{
    ui->cathodeChnlThirdMultiTrimValEdit->setText(GetThresholdTrimStr(arg1));
}

void MainWindow::on_cathodeChnlPosTrimSpinner_valueChanged(int arg1)
{
    ui->cathodeChnlPosTrimValEdit->setText(GetPulseThresholdTrimStr(arg1));
}

void MainWindow::on_anodeChnlPosTrimSpinner_valueChanged(int arg1)
{
    ui->anodeChnlPosTrimValEdit->setText(GetPulseThresholdTrimStr(arg1));
}

void MainWindow::on_anodeChnlsNegTrimSpinner_valueChanged(int arg1)
{
    ui->anodeChnlsNegTrimValEdit->setText(GetPulseThresholdTrimStr(arg1));
}

void MainWindow::on_secondaryMultiThreshDispSpinner_valueChanged(int arg1)
{
    ui->secondaryMultiThreshDispValEdit->setText(GetMultiThreshDisplacementStr(arg1));
}

void MainWindow::on_cameraUpdateAutomaticRadio_clicked()
{
    if(!SpectDMDll::SetCameraUpdateMode(CameraUpdateMode_Auto))
    {
       QMessageBox::warning(this, "SpectDM", SpectDMDll::GetLastError().c_str());
    }
}

void MainWindow::on_cameraUpdateModeManualRadio_clicked()
{
    if(!SpectDMDll::SetCameraUpdateMode(CameraUpdateMode_Manual))
    {
       QMessageBox::warning(this, "SpectDM", SpectDMDll::GetLastError().c_str());
    }
}

void MainWindow::on_allGMsRadio_clicked()
{
    if(SpectDMDll::SetGMUpdateType(GMUpdateType_Broadcast))
    {
        // need to disable all single GM widgets
        EnableSingleGMWidgets(false);
    }
    else
    {
        QMessageBox::warning(this, "SpectDM", SpectDMDll::GetLastError().c_str());
    }
}

void MainWindow::on_singleGMRadio_clicked()
{
    EnableSingleGMWidgets(true);
}

void MainWindow::on_GMCombo_currentTextChanged(const QString &arg1)
{
    // If an actual GM no has been selected
    if(arg1 != "-")
    {
        if(SpectDMDll::SetActiveGM(arg1.toInt()))
        {
            // if we have successfully set the active GM, we want to update
            // FPGA and ASIC with the info associated with it
            UpdateGMWidgets();
        }
        else
        {
            QMessageBox::warning(this, "SpectDM", SpectDMDll::GetLastError().c_str());
        }
    }
}

void MainWindow::on_actionPackets_Bin_triggered()
{
    QFileDialog l_SavePacketsDlg(this, "Save Packets Binary", QString(), "Binary files (*.dat)");
    l_SavePacketsDlg.setAcceptMode(QFileDialog::AcceptSave);

    if(l_SavePacketsDlg.exec())
    {
        QStringList l_SelectedFiles = l_SavePacketsDlg.selectedFiles();

        if(!l_SelectedFiles.isEmpty())
        {
            if(!SpectDMDll::SaveBinaryCollection(l_SelectedFiles.at(0).toStdString().c_str()))
            {
                QMessageBox::warning(this, "SpectDM", SpectDMDll::GetLastError().c_str());
            }

            // save where the file was saved to for so dialog is opened at this dir
            // next time
            l_SavePacketsDlg.saveState();
        }
    }
}

void MainWindow::OnPacketSaveOperationCancelled()
{
    SpectDMDll::CancelPacketSave();
}
