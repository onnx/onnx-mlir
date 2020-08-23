//===---------------------- DataType.h - ONNX DataTypes -------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains declaration of ONNX data types and type size mapping.
// It is provided as a convenience and not used by RtMemRef implementation.
//
//===----------------------------------------------------------------------===//

#ifndef __DATA_TYPE_H__
#define __DATA_TYPE_H__

#ifdef __cplusplus
#include <cstdint>
#include <map>
#include <string>
#else
#include <stdbool.h>
#include <stdint.h>
#endif

enum RTMEMREF_DATA_TYPE {
#define OM_TYPE_METADATA_DEF(ENUM_NAME, ENUM_VAL, DTYPE_SIZE)                  \
  ENUM_NAME = ENUM_VAL,
#include "DataTypeMetaData.h"
#undef OM_TYPE_METADATA_DEF
};

const int RTMEMREF_DATA_TYPE_SIZE[] = {
#define OM_TYPE_METADATA_DEF(ENUM_NAME, ENUM_VAL, DTYPE_SIZE) DTYPE_SIZE,
#include "DataTypeMetaData.h"
#undef OM_TYPE_METADATA_DEF
};

#ifdef __cplusplus
// Note by design const map has no [] operator since [] creates a default
// key value mapping when the key is not found which changes the map
const std::map<std::string, int> RTMEMREF_DATA_TYPE_CPP_TO_ONNX = {
    {"b", ONNX_TYPE_BOOL},   // bool  -> BOOL
    {"c", ONNX_TYPE_INT8},   // char  -> INT8 (platform dependent, can be UINT8)
    {"a", ONNX_TYPE_INT8},   // int8_t   -> INT8
    {"h", ONNX_TYPE_UINT8},  // uint8_t  -> UINT8,  unsigned char  -> UNIT 8
    {"s", ONNX_TYPE_INT16},  // int16_t  -> INT16,  short          -> INT16
    {"t", ONNX_TYPE_UINT16}, // uint16_t -> UINT16, unsigned short -> UINT16
    {"i", ONNX_TYPE_INT32},  // int32_t  -> INT32,  int            -> INT32
    {"j", ONNX_TYPE_UINT32}, // uint32_t -> UINT32, unsigned int   -> UINT32
    {"l", ONNX_TYPE_INT64},  // int64_t  -> INT64,  long           -> INT64
    {"m", ONNX_TYPE_UINT64}, // uint64_t -> UINT64, unsigned long  -> UINT64
    {"f", ONNX_TYPE_FLOAT},  // float    -> FLOAT
    {"d", ONNX_TYPE_DOUBLE}, // double   -> DOUBLE
};
#endif

#endif