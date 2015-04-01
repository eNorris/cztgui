#ifndef GLOBALS
#define GLOBALS

#include <QString>

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

// Utility functions
QString createTimestampedStr(const std::string &a_Str);


#endif // GLOBALS

