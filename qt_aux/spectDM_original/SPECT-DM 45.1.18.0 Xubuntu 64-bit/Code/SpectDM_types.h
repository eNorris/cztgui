#ifndef SPECTDM_TYPES_H
#define SPECTDM_TYPES_H

#include <QMetaType>

typedef int GMOption;
const GMOption GMOption_None                 = 0;
const GMOption GMOption_DisablePhotonCollect = 1;
const GMOption GMOption_DebugMode            = 1 << 1;
const GMOption GMOption_Channel1TestMode     = 1 << 2;
const GMOption GMOption_EnablePixelMap       = 1 << 3;

typedef int GMReadoutOption;
const GMReadoutOption GMReadoutOpt_None           = 0;
const GMReadoutOption GMReadoutOpt_NegativeEnergy = 1;
const GMReadoutOption GMReadoutOpt_TimeDetect     = 1 << 1;

typedef int GMPulserOption;
const GMPulserOption GMPulserOpt_None    = 0;
const GMPulserOption GMPulserOpt_Anode   = 1;
const GMPulserOption GMPulserOpt_Cathode = 1 << 1;

typedef int GMStatus;
const GMStatus GMStatus_Undefined = -1;
const GMStatus GMStatus_Idle = 1;
const GMStatus GMStatus_ASICLoadError = 2;
const GMStatus GMStatus_FIFOFull = 4;

const int pktTransferRateStep = 1000;
const int minPktTransferRate = 1000;
const int maxPktTransferRate = 63000;

enum GMReadoutMode
{
    GMReadout_Undefined
  , GMReadout_ReadAll
  , GMReadout_SparsifiedMode
};

enum GMCathodeMode
{
    GMCathMode_Undefined
  , GMCathMode_Unipolar
  , GMCathMode_MultiThreshold
  , GMCathMode_Bipolar
};

enum GMPulserFrequency
{
    GMPulserFreq_Undefined
  , GMPulserFreq_100Hz
  , GMPulserFreq_1kHz
  , GMPulserFreq_10kHz
  , GMPulserFreq_100kHz
};

enum GMASICTestModeCathodeChannelInput
{
    GMASICTestModeCathodeChannelInput_Undefined
  , GMASICTestModeCathodeChannelInput_Step
  , GMASICTestModeCathodeChannelInput_Ramp
};

enum GMUpdateType
{
    GMUpdateType_Undefined
  , GMUpdateType_SingleGM
  , GMUpdateType_Broadcast
};

enum SysTokenType
{
    SysTokenType_Undefined
  , SysTokenType_DaisyChain
  , SysTokenType_GM3Only
};

enum MBMultiplexerAddrType
{
    MBMultiplexerAddrType_Undefined = -1
  , MBMultiplexerAddrType_GM1_AUX = 0
  , MBMultiplexerAddrType_GM2_AUX
  , MBMultiplexerAddrType_GM3_AUX
  , MBMultiplexerAddrType_GM4_AUX
  , MBMultiplexerAddrType_GM0AXPK62
  , MBMultiplexerAddrType_GM0AXPK63
  , MBMultiplexerAddrType_EMKO_IN
  , MBMultiplexerAddrType_MB_TEMP
  , MBMultiplexerAddrType_REF2V
  , MBMultiplexerAddrType_S3_3V
  , MBMultiplexerAddrType_P2_5V
  , MBMultiplexerAddrType_A2_5V
  , MBMultiplexerAddrType_D2_5V
  , MBMultiplexerAddrType_F2_5V
  , MBMultiplexerAddrType_F1_5V
  , MBMultiplexerAddrType_S5V
};

enum PacketData
{
    PacketData_Undefined
  , PacketData_GMNo
  , PacketData_Timestamp
  , PacketData_PhotonCount
  , PacketData_ActualPhotonCount
};

enum PhotonData
{
    PhotonData_Undefined
  , PhotonData_Coordinate
  , PhotonData_Energy
  , PhotonData_EnergyPosEvent
  , PhotonData_TimeDetect
  , PhotonData_TimeDetectPosEvent
};

enum CameraUpdateMode
{
    CameraUpdateMode_Undefined
  , CameraUpdateMode_Manual
  , CameraUpdateMode_Auto
};

enum PixelMappingMode
{
    PixelMappingMode_Undefined
  , PixelMappingMode_Global
  , PixelMappingMode_GMBased
};

Q_DECLARE_METATYPE(MBMultiplexerAddrType)
Q_DECLARE_METATYPE(PixelMappingMode)

#endif // SPECTDM_TYPES_H
