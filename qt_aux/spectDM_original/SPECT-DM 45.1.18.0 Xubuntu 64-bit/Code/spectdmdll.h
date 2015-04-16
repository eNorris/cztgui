#ifndef SPECTDMDLL_H
#define SPECTDMDLL_H

#include "spectdmsharedlib_global.h"

#ifdef INTERNAL_BUILD
    #include "../SpectDM/simulator_types.h"
#endif

#ifdef KROMEK_BUILD
    #include "../../../evCommonLibs/evDetectorCommon/common_types.h"
    #include "../H3DASIC/H3DASIC_types.h"
    #include "../SpectDM/SpectDM_types.h"
#else
    #include "common_types.h"
    #include "H3DASIC_types.h"
    #include "SpectDM_types.h"
#endif

#include <iostream>

class SpectDMDll
{

public:

    static SPECTDMSHAREDLIBSHARED_EXPORT std::string GetVersion();

    static SPECTDMSHAREDLIBSHARED_EXPORT void SetHostIPAddress(
        const char* hostIPAddress
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT void SetCameraIPAddress(
        const char* cameraIPAddress
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT std::string GetCameraIPAddress();

    static SPECTDMSHAREDLIBSHARED_EXPORT bool Initialize();

    static SPECTDMSHAREDLIBSHARED_EXPORT bool IsSystemInitialized();

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GetGBEFirmwareVersion(
        int* version
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GetGMFirmwareVersion(
        int* version
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT int GetNoOfGMs();

    static SPECTDMSHAREDLIBSHARED_EXPORT int GetNoOfLoadedGMs();

    static SPECTDMSHAREDLIBSHARED_EXPORT bool CopyLoadedGMToGM(
        int loadedGM
      , int GM
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool StartSys();

    static SPECTDMSHAREDLIBSHARED_EXPORT bool StopSys();

    static SPECTDMSHAREDLIBSHARED_EXPORT bool StartPhotonCollection(int collectionTimeInSeconds = -1);

    static SPECTDMSHAREDLIBSHARED_EXPORT bool StopPhotonCollection();

    static SPECTDMSHAREDLIBSHARED_EXPORT bool IsSystemCollectingPhotons();

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetDebugMode(
        bool enable
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool IsDebugModeActive();

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetMBMultiplexerAddressType(
        MBMultiplexerAddrType addressType
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT MBMultiplexerAddrType GetMBMultiplexerAddressType();

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetPacketTransferRate(
        int transferRate
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT int GetPacketTransferRate();

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetPixelMappingMode(
        PixelMappingMode pixelMappingMode
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT PixelMappingMode GetPixelMappingMode();

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetDefaultGM(
        int GM_ID
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GetDefaultGM(
        int* GM_ID
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetActiveGM(
        int GM_ID
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT int GetActiveGM();

    static SPECTDMSHAREDLIBSHARED_EXPORT bool IsActiveGMSet();

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GetGMStatus(
        GMStatus* status
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetGMOptions(
        GMOption options
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GetGMOptions(
        GMOption* options
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetGMADC1Channel(
        int channel
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GetGMADC1Channel(
        int* channel
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetGMADC2Channel(
        int channel
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GetGMADC2Channel(
        int* channel
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool ReadGMADC1(
        int* DAC
      , int* volts
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool ReadGMADC2(
        int* DAC
      , int* volts
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetDelayTime(
        int delayTimeInNanoSeconds
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GetDelayTime(
        int* delayTimeInNanoSeconds
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetTimestampResolution(
        int resolutionInMicroSeconds
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GetTimestampResolution(
        int* resolutionInMicroSeconds
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool RegWrite(
        int regNum
      , unsigned char MSB
      , unsigned char LSB
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool RegRead(
        int regNum
      , unsigned char* MSB
      , unsigned char* LSB
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GMRegWrite(
        int GMRegNum
      , unsigned char data
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GMRegRead(
        int GMRegNum
      , unsigned char* data
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetGMReadoutOptions(
        GMReadoutOption options
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GetGMReadoutOptions(
        GMReadoutOption* options
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetGMReadoutMode(
        GMReadoutMode mode
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GetGMReadoutMode(
        GMReadoutMode* mode
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetGMCathodeMode(
        GMCathodeMode mode
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GetGMCathodeMode(
        GMCathodeMode* mode
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetGMPulserFrequency(
        GMPulserFrequency freq
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GetGMPulserFrequency(
        GMPulserFrequency* freq
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetGMPulserOptions(
        GMPulserOption options
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GetGMPulserOptions(
        GMPulserOption* options
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetNoOfPulses(
        int noOfPulses
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GetNoOfPulses(
        int* noOfPulses
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetASICGlobalOptions(
        ASICGlobalOptions options
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GetASICGlobalOptions(
        ASICGlobalOptions* options
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetTimingChannelUnipolarGain(
        TimingChannelUnipolarGain gain
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GetTimingChannelUnipolarGain(
        TimingChannelUnipolarGain* gain
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetASICReadoutMode(
        GMASICReadoutMode mode
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GetASICReadoutMode(
        GMASICReadoutMode* mode
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetChannelGain(
        ChannelGain gain
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GetChannelGain(
        ChannelGain* gain
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetCathodeTestModeInput(
        TestModeInput input
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GetCathodeTestModeInput(
        TestModeInput* input
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetInternalLeakageCurrentGenerator(
        InternalLeakageCurrentGenerator currGen
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GetInternalLeakageCurrentGenerator(
        InternalLeakageCurrentGenerator* currGen
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetAnodeTestPulseEdge(
        TestPulseEdge edge
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GetAnodeTestPulseEdge(
        TestPulseEdge* edge
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetPeakDetectTimeout(
        ChannelType type
      , int timeoutInMicroSeconds
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GetPeakDetectTimeout(
        ChannelType type
      , int* timeoutInMicroSeconds
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetTimeDetectRampLength(
        ChannelType type
      , int rampLengthInMicroSeconds
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GetTimeDetectRampLength(
        ChannelType type
      , int* rampLengthInMicroSeconds
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetPeakingTime(
        ChannelType type
      , float peakingTimeInMicroSeconds
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GetPeakingTime(
        ChannelType type
      , float* peakingTimeInMicroSeconds
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetCathodeTimingChannelsShaperPeakingTime(
        TimingChannelsShaperPeakingTime peakingTime
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GetCathodeTimingChannelsShaperPeakingTime(
        TimingChannelsShaperPeakingTime* peakingTime
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetCathodeTimingChannelsSecondaryMultiThresholdsDisplacementStep(
        int displacementStep
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GetCathodeTimingChannelsSecondaryMultiThresholdsDisplacementStep(
        int* displacementStep
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetMultipleFiringSuppressTime(
        MultipleFiringSuppressionTime suppressTime
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GetMultipleFiringSuppressTime(
        MultipleFiringSuppressionTime* suppressTime
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetTestPulseStep(
        ChannelType type
      , int testPulseStep
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GetTestPulseStep(
        ChannelType type
      , int* testPulseStep
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetChannelThresholdStep(
        ChannelThresholdType type
      , int thresholdStep
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GetChannelThresholdStep(
        ChannelThresholdType type
      , int* thresholdStep
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetCathodeTestClockType(
        CathodeTestClockType clockType
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GetCathodeTestClockType(
        CathodeTestClockType* clockType
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetTimingChannelBiPolarGain(
        TimingChannelBipolarGain gain
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GetTimingChannelBiPolarGain(
        TimingChannelBipolarGain* gain
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetCathodeChannelInternalLeakageCurrentGenerator(
        int channelNumber
      , CathodeChannel::InternalLeakageCurrentGenerator currGen
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GetCathodeChannelInternalLeakageCurrentGenerator(
        int channelNumber
      , CathodeChannel::InternalLeakageCurrentGenerator* currGen
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetAnalogOutputToMonitor(
        AnalogOutput output
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GetAnalogOutputMonitored(
        AnalogOutput* output
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetAnodeChannelToMonitor(
        int channel
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GetAnodeChannelMonitored(
        int* channel
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetCathodeEnergyTimingToMonitor(
        CathodeEnergyTiming energyTiming
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GetCathodeEnergyTimingMonitored(
        CathodeEnergyTiming* energyTiming
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetDACToMonitor(
        DACS dac
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GetDACMonitored(
        DACS* dac
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool MaskChannel(
        bool mask
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool IsChannelMasked(
        ChannelType type
      , int channelNumber
      , bool* mask
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool EnableChannelTestCapacitor(
        bool enable
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool IsChannelTestCapacitorEnabled(
        ChannelType type
      , int channelNumber
      , bool* enabled
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool MonitorAnodeSignal(
        Signal signal
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GetAnodeSignalMonitored(
        int channelNumber
      , Signal* signal
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetCathodeChannelShapedTimingSignal(
        CathodeChannel::ShapedTimingSignal signal
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GetCathodeChannelShapedTimingSignal(
        int channelNumber
      , CathodeChannel::ShapedTimingSignal* signal
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetCathodeChannelTimingMode(
        CathodeChannel::TimingMode mode
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GetCathodeChannelTimingMode(
        int channelNumber
      , CathodeChannel::TimingMode* mode
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetCathodeTimingChannelTrimStep(
        CathodeTimingChannelType channelType
      , int trimStep
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GetCathodeTimingChannelTrimStep(
        int channelNumber
      , CathodeTimingChannelType channelType
      , int* trimStep
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetChannelPositivePulseThresholdTrimStep(
        int trimStep
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GetChannelPositivePulseThresholdTrimStep(
        ChannelType channelType
      , int channelNumber
      , int* trimStep
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetAnodeChannelNegativePulseThresholdTrimStep(
        int trimStep
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GetAnodeChannelNegativePulseThresholdTrimStep(
        int channelNumber
      , int* trimStep
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetSysTokenType(
        SysTokenType sysTokenType
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT SysTokenType GetSysTokenType();

    static SPECTDMSHAREDLIBSHARED_EXPORT std::string GetLastError();

    static SPECTDMSHAREDLIBSHARED_EXPORT void SetConnectStatusFunction(
        callback_function connectStatusFunc
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT void SetSysStatusFunction(
        callback_function sysStatusFunc
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT void SetToolbarStatusFunction(
        callback_function toolbarStatusFunc
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT void Disconnect();

    static SPECTDMSHAREDLIBSHARED_EXPORT int GetNoOfCollectedPackets();

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GetPacketData(
        int packetNo
      , PacketData data
      , int* returnedData
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GetPhotonData(
        int packetNo
      , int photonNo
      , PhotonData data
      , int* returnedData
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SaveConfiguration(
        const char* filename
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool LoadConfiguration(
        const char* filename
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetGMUpdateType(
        GMUpdateType GMUpdateType
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT GMUpdateType GetGMUpdateType();

    static SPECTDMSHAREDLIBSHARED_EXPORT void SetChannelUpdateType(
        ChannelUpdateType channelUpdateType
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT void SetActiveChannelType(
        ChannelType channelType
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetActiveChannel(
        int channel
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SaveCollection(
        const char* filename
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SaveBinaryCollection(
        const char* filename
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetCameraUpdateMode(
        CameraUpdateMode cameraUpdateMode
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT CameraUpdateMode GetCameraUpdateMode();

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SendGBEControlData();

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SendGMData();

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SendASICGlobalData();

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SendASICChannelData();

#ifdef INTERNAL_BUILD
    static SPECTDMSHAREDLIBSHARED_EXPORT void SetSimulatorOptions(
        SimulatorOptions options
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT SimulatorOptions GetSimulatorOptions();

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetSimulatorPacketFrequency(
        SimulatorPacketFrequency pktFreq
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT SimulatorPacketFrequency GetSimulatorPacketFrequency();

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetSimulatorPhotonType(
        SimulatorPhotonType photonType
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT SimulatorPhotonType GetSimulatorPhotonType();

    static SPECTDMSHAREDLIBSHARED_EXPORT void SetSimulatorPhotonsPerPacket(
        int photonsPerPacket
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT int GetSimulatorPhotonsPerPacket();

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetSimulatorGMNo(
        int GMNo
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT int GetSimulatorGMNo();

    static SPECTDMSHAREDLIBSHARED_EXPORT void SetSimulatorNoOfPackets(
        unsigned short noOfPackets
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT unsigned short GetSimulatorNoOfPackets();

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetSimulatorEnergy(
        SimulatorEnergyType energyType
      , int energy
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GetSimulatorEnergy(
        SimulatorEnergyType energyType
      , int* energy
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetSimulatorTime(
        SimulatorTimeType timeType
      , int time
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool GetSimulatorTime(
        SimulatorTimeType timeType
      , int* time
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SetSimulatorTimeOffset(
        int timeOffset
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT int GetSimulatorTimeOffset();

    static SPECTDMSHAREDLIBSHARED_EXPORT bool StartSimulator();

    static SPECTDMSHAREDLIBSHARED_EXPORT bool StopSimulator();

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SimulatorRegWrite(
        int simulatorRegNum
      , unsigned char MSB
      , unsigned char LSB
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT bool SimulatorRegRead(
        int simulatorRegNum
      , unsigned char* MSB
      , unsigned char* LSB
    );
#endif

    static SPECTDMSHAREDLIBSHARED_EXPORT void Close();

    static SPECTDMSHAREDLIBSHARED_EXPORT void SetOperationErrorFunction(
        callback_function_op_str opErrorFunction
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT void SetOperationProgressFunction(
        callback_function_op_int opProgressFunction
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT void SetOperationCompleteFunction(
        callback_function_op opCompleteFunction
    );

    static SPECTDMSHAREDLIBSHARED_EXPORT void CancelPacketSave();
};

#endif // SPECTDMDLL_H
