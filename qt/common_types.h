#ifndef COMMON_TYPES_H
#define COMMON_TYPES_H

typedef void(*void_callback_function)();
typedef void(*callback_function)(const char*);
typedef void(*callback_function_int)(int);

enum RegisterOp
{
  RegOp_Write
, RegOp_Read
};

#define OneMBInBits 1048576

#endif // COMMON_TYPES_H
