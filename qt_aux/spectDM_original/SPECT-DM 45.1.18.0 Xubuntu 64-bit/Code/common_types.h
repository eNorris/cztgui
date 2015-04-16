#ifndef COMMON_TYPES_H
#define COMMON_TYPES_H

enum OperationType
{
    OperationType_SavePackets,
    OperationType_Collection
};

typedef void(*void_callback_function)();
typedef void(*callback_function)(const char*);
typedef void(*callback_function_str_and_size)(const char*, int);
typedef void(*callback_function_int)(int);
typedef void(*callback_function_op)(OperationType);
typedef void(*callback_function_op_int)(OperationType, int);
typedef void(*callback_function_op_str)(OperationType, const char*) ;

enum UpdateType
{
    UpdateType_Single
  , UpdateType_Broadcast
};

enum DeviceUpdateMode
{
    DeviceUpdateMode_Manual
  , DeviceUpdateMode_Auto
};

#endif // COMMON_TYPES_H
