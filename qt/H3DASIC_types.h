#ifndef H3DASIC_TYPES_H
#define H3DASIC_TYPES_H

typedef int ASICGlobalOptions;
const ASICGlobalOptions ASICGlobal_None = 0;
const ASICGlobalOptions ASICGlobal_SingleEventMode = 1;
const ASICGlobalOptions ASICGlobal_EnergyMultipleFiringSuppressor = 1 << 1;
const ASICGlobalOptions ASICGlobal_Validation = 1 << 2;
const ASICGlobalOptions ASICGlobal_MonitorOutputs = 1 << 3;
const ASICGlobalOptions ASICGlobal_RouteTempMonitorToAXPK62 = 1 << 4;
const ASICGlobalOptions ASICGlobal_TimingMultipleFiringSuppressor = 1 << 5;
const ASICGlobalOptions ASICGlobal_DisableMultipleResetAcquisitionMode = 1 << 6;
const ASICGlobalOptions ASICGlobal_RouteMonitorToPinTDO = 1 << 7;
const ASICGlobalOptions ASICGlobal_BufferChnl62PreAmplifierMonitorOutput = 1 << 8;
const ASICGlobalOptions ASICGlobal_BufferChnl63PreAmplifierMonitorOutput = 1 << 9;
const ASICGlobalOptions ASICGlobal_BufferPeakAndTimeDetectorOutputs = 1 << 10;
const ASICGlobalOptions ASICGlobal_BufferAuxMonitorOutput = 1 << 11;
const ASICGlobalOptions ASICGlobal_HighGain = 1 << 12;

const int minThresholdStep = 0;
const int maxThresholdStep = 1023;

const int minThresholdTrimStep = 0;
const int maxThresholdTrimStep = 15;

const int minPulseThresholdTrimStep = 0;
const int maxPulseThresholdTrimStep = 31;

const int minCathMultiThresholdsDisplacementStep = 0;
const int maxCathMultiThresholdsDisplacementStep = 15;

enum AnodeTestPulseEdge
{
    AnodeTestPulseEdge_Undefined
  , AnodeTestPulseEdge_InjectNegativeCharge
  , AnodeTestPulseEdge_InjectPosAndNegCharge
};

enum MonitorSignal
{
    MonitorSignal_Positive
  , MonitorSignal_Negative
};

enum ChannelType
{
    ChannelType_Anode
  , ChannelType_Cathode
};

enum TimingChannelUnipolarGain
{
    TimingChannelUnipolarGain_Undefined
  , TimingChannelUnipolarGain_27mV
  , TimingChannelUnipolarGain_81mV
};

enum ChannelGain
{
    ChannelGain_Undefined
  , ChannelGain_20mV
  , ChannelGain_60mV
};

enum TestModeInput
{
    TestModeInput_Undefined
  , TestModeInput_Step
  , TestModeInput_Ramp
};

enum InternalLeakageCurrentGenerator
{
    InternalLeakageCurrentGenerator_Undefined
  , InternalLeakageCurrentGenerator_60pA
  , InternalLeakageCurrentGenerator_0A
};

enum TimingChannelsShaperPeakingTime
{
    TimingChannelsShaperPeakingTime_Undefined
  , TimingChannelsShaperPeakingTime_100nS
  , TimingChannelsShaperPeakingTime_200nS
  , TimingChannelsShaperPeakingTime_400nS
  , TimingChannelsShaperPeakingTime_800nS
};

enum MultipleFiringSuppressionTime
{
    MultipleFiringSuppressionTime_Undefined
  , MultipleFiringSuppressionTime_62_5nS
  , MultipleFiringSuppressionTime_125nS
  , MultipleFiringSuppressionTime_250nS
  , MultipleFiringSuppressionTime_600nS
};

enum ChannelThresholdType
{
    ChannelThresholdType_CathodeTimingPrimaryMultiThresholdBiPolar
  , ChannelThresholdType_CathodeTimingUnipolar
  , ChannelThresholdType_CathodeEnergy
  , ChannelThresholdType_AnodeNegativeEnergy
  , ChannelThresholdType_AnodePositiveEnergy
};

enum CathodeTestClockType
{
    CathodeTestClockType_Undefined
  , CathodeTestClockType_CopyAnodeTestClock
  , CathodeTestClockType_ArrivesOnSDI_NSDI
};

enum TimingChannelBipolarGain
{
    TimingChannelBipolarGain_Undefined
  , TimingChannelBipolarGain_21mV
  , TimingChannelBipolarGain_55mV
  , TimingChannelBipolarGain_63mV
  , TimingChannelBipolarGain_164mV
};

enum AnalogOutput
{
    AnalogOutput_Undefined
  , AnalogOutput_NoFunction
  , AnalogOutput_Baseline
  , AnalogOutput_Temperature
  , AnalogOutput_DACS
  , AnalogOutput_CathodeEnergyTiming
  , AnalogOutput_AnodeEnergy
};

enum CathodeEnergyTiming
{
    CathodeEnergyTiming_Undefined
  , CathodeEnergyTiming_Channel1Energy
  , CathodeEnergyTiming_Channel1Timing
  , CathodeEnergyTiming_Channel2Energy
  , CathodeEnergyTiming_Channel2Timing
};

enum DACS
{
    DACS_Undefined
  , DACS_AnodeEnergyThreshold
  , DACS_AnodeEnergyTransient
  , DACS_CathodeEnergyThreshold
  , DACS_CathodeTimingUnipolarThreshold
  , DACS_CathodeTimingFirstMultiThreshold
  , DACS_CathodeTimingSecondMultiThreshold
  , DACS_CathodeTimingThirdMultiThreshold
  , DACS_AnodeTestSignal
  , DACS_CathodeTestSignal
};

enum Signal
{
    Signal_Undefined
  , Signal_Positive
  , Signal_Negative
};

enum ChannelUpdateType
{
    ChannelUpdateType_SingleChannel
  , ChannelUpdateType_Broadcast
};

enum TestPulseEdge
{
    TestPulseEdge_Undefined
  , TestPulseEdge_InjectNegCharge
  , TestPulseEdge_InjectPosAndNegCharge
};

enum CathodeTimingChannelType
{
    CathodeTimingChannelType_FirstMultiThresholdBiPolar
  , CathodeTimingChannelType_SecondMultiThreshold
  , CathodeTimingChannelType_ThirdMultiThreshold
  , CathodeTimingChannelType_Unipolar
};

enum GMASICReadoutMode
{
    GMASICReadout_Undefined
  , GMASICReadout_NormalSparsified
  , GMASICReadout_EnhancedSparsified
};

namespace CathodeChannel
{
    enum InternalLeakageCurrentGenerator
    {
        InternalLeakageCurrentGenerator_Undefined
      , InternalLeakageCurrentGenerator_350pA
      , InternalLeakageCurrentGenerator_2nA
    };

    enum ShapedTimingSignal
    {
        ShapedTimingSignal_Undefined
      , ShapedTimingSignal_Unipolar
      , ShapedTimingSignal_Bipolar
    };

    enum TimingMode
    {
        TimingMode_Undefined
      , TimingMode_Unipolar
      , TimingMode_MultiThreshold_Unipolar
      , TimingMode_BiPolar_Unipolar
    };
}

#endif // H3DASIC_TYPES_H
