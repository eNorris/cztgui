#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QProcess>
#include <QProgressDialog>

// This isn't included in the API as customer should never see it so
// only include if this is an internal build
#if INTERNAL_BUILD
    #include "../../API/SpectDM/simulator_types.h"
#endif

// If this is defined then the API will know that we have the SpectDM
// project structure in place and that it is safe to use the relative paths
// to the various includes in the API
// #define KROMEK_BUILD TRUE

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    static MainWindow* m_pInstance;
    static MainWindow* GetInstance();
    static void DeleteInstance();

    static void ConnectStatusCallbackFunc(
        const char* a_Data
    );

    static void SysStatusCallbackFunc(
        const char* a_Data
    );

    static void ToolbarStatusCallbackFunc(
        const char* a_Data
    );

    static void OperationErrorCallbackFunc(
        const char* a_Data
    );

    static void OperationCompleteCallbackFunc();

    static void OperationProgressCallbackFunc(
        int a_Progress
    );

    void AddConnectStatusEntry(
        const std::string &a_Status
    );

    void AddSysStatusEntry(
        const std::string &a_Status
    );

protected:
    void keyPressEvent(
        QKeyEvent* a_KeyEvent
    );

signals:
    void progressUpdate(int);

private slots:
    void on_connectButton_clicked();

    void on_disconnectButton_clicked();

    void on_sendConfigButton_clicked();

    void on_UpdateGMButton_clicked();

//    void on_gmRegReadButton_clicked();

    //void on_asicTestBtn_clicked();

    void on_ASICGlobal_UpdateASICButton_clicked();

    void on_ASICAnode_UpdateASICButton_clicked();

    void on_updateAnodeChannelButton_clicked();

    void on_anodeChannelNoCombo_currentIndexChanged(const QString &);

    void on_cathode_UpdateASICButton_clicked();

    void on_updateCathodeChannelButton_clicked();

    void on_cathodeChannelNoCombo_currentTextChanged(const QString &);

    void on_writeRegButton_clicked();

    void on_readRegButton_clicked();

    void on_gmRegReadButton_clicked();

    void on_gmWriteRegButton_clicked();

//    void on_saveConfigBtn_clicked();

//    void on_loadSysConfigBtn_clicked();

    void on_readNoOfPacketsBtn_clicked();

    void on_readPacketBtn_clicked();

    void on_readPhotonBtn_clicked();

    void on_startCollectBtn_clicked();

    void on_stopCollectBtn_clicked();

    void on_GMStatusBtn_clicked();

    void on_ADC1ReadButton_clicked();

    void on_ADC2ReadBtn_clicked();

    void on_loadFromFileBtn_clicked();

    void on_ReadGBEFirmwareBtn_clicked();

    void on_ReadGMFirmwareBtn_clicked();

    void on_startSysBtn_clicked();

    void on_stopSysBtn_clicked();

    void on_copyGMConfigBtn_clicked();

    void on_anodeChannelRadio_toggled(bool checked);

    void on_resetPacketTextBtn_clicked();

    void on_readAllPacketsBtn_clicked();

    void on_SaveAction();

#ifdef INTERNAL_BUILD
    void on_simCycleCheck_clicked(bool checked);

    void on_startSimBtn_clicked();

    void on_simPhotonTypeCombo_currentTextChanged(const QString &arg1);

    void on_stopSimBtn_clicked();

    void on_simRegReadButton_clicked();

    void on_simRegWriteButton_clicked();
#endif

    void on_actionAbout_triggered();

    void on_allCathodesRadio_clicked();

    void on_allAnodesRadio_clicked();

    void on_individualAnodeRadio_clicked();

    void on_individualCathodeRadio_clicked();

    void on_noFuncRadio_clicked();

    void on_baselineRadio_clicked();

    void on_temperatureRadio_clicked();

    void on_DACSradio_clicked();

    void on_cathodeEnergyTimingRadio_clicked();

    void on_anodeChannelRadio_clicked();

    void on_actionSpect_DM_Help_triggered();

    void on_actionConfiguration_triggered();

    void on_actionPackets_triggered();

    void on_anodeTestPulseSpinner_valueChanged(int arg1);

    void on_cathodeTestPulseSpinner_valueChanged(int arg1);

    void on_anodePosEnergyThreshSpinner_valueChanged(int arg1);

    void on_anodeNegEnergyThreshSpinner_valueChanged(int arg1);

    void on_cathodeEnergyThreshSpinner_valueChanged(int arg1);

    void on_cathodeTimingPrimaryThreshSpinner_valueChanged(int arg1);

    void on_cathodeTimingUnipolarThreshSpinner_valueChanged(int arg1);

    void on_cathodeChnlUnipolarTrimSpinner_valueChanged(int arg1);

    void on_cathodeChnlFirstMultiTrimSpinner_valueChanged(int arg1);

    void on_cathodeChnlSecondMultiTrimSpinner_valueChanged(int arg1);

    void on_cathodeChnlThirdMultiTrimSpinner_valueChanged(int arg1);

    void on_cathodeChnlPosTrimSpinner_valueChanged(int arg1);

    void on_anodeChnlPosTrimSpinner_valueChanged(int arg1);

    void on_anodeChnlsNegTrimSpinner_valueChanged(int arg1);

    void on_secondaryMultiThreshDispSpinner_valueChanged(int arg1);

    void on_cameraUpdateAutomaticRadio_clicked();

    void on_cameraUpdateModeManualRadio_clicked();

    void on_allGMsRadio_clicked();

    void on_singleGMRadio_clicked();

    void on_GMCombo_currentTextChanged(const QString &arg1);

    void on_actionPackets_Bin_triggered();

    void OnPacketSaveOperationCancelled();

private:

    MainWindow(QWidget *parent = 0);
    ~MainWindow();

    void SetupSpectUI();

    void UpdateGMItems();

    void UpdateASICGlobalItems();

    void UpdateASICAnodeItems();

    void UpdateASICCathodeItems();

    void UpdateSelectedAnodeChannelItems();

    void UpdateSelectedCathodeChannelItems();

    void UpdateSysConfigItems();

#ifdef INTERNAL_BUILD
    void UpdateSimulatorItems();
#endif

    void EnablePhotonTypeTimeAndEnergy(
        int a_Type
      , bool a_Enable
    );

    void DisableAnalogOutputCombos();

    QString GetCurrTabIndexHelpPage() const;

    void StartHelp();

    void EnableNonConnectTabs(
        bool a_Enable
    );

    QString IntToHexString(
        int a_No
    );

    int HexStringToInt(
        const QString& a_HexString
    );

    void UpdateHGAffectedWidgets(
        bool a_HGSet
    );

    QString GetTestPulseStr(
        int a_PulseStep
    ) const;

    QString GetThresholdStr(
        int a_ThresholdStep
    ) const;

    QString GetThresholdTrimStr(
        int a_ThresholdTrimStep
    ) const;

    QString GetPulseThresholdTrimStr(
        int a_PulseThresholdTrimStep
    ) const;

    QString GetMultiThreshDisplacementStr(
        int a_MultiThreshDispStep
    ) const;

    void EnableSingleGMWidgets(
        bool a_Enable
    );

    void UpdateGMWidgets();

    void UpdateGMPixelMapCheckbox();

    // returns arg passed to function prepended with timestamp for
    // adding strings to status text areas
    QString CreateTimestampedStr(
        const std::string& a_Str
    );

    void CloseProgressDlg();

    void ReportWarning(
        const QString& a_Warning
    );

    void SetupProgressDialog(
        const QString& a_Caption
      , const QString& a_CancelText
    );

    Ui::MainWindow *ui;
    bool m_Connected;
    QProcess m_HelpProcess;
    QProgressDialog* m_ProgressDialog;
    long m_PacketCount;
};

#endif // MAINWINDOW_H
