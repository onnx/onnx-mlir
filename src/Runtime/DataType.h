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
#pragma once
#include <cstdint>
#include <map>
#include <string>

enum RTMEMREF_DATA_TYPE {
  ONNX_TYPE_UNDEFINED = 0,
  // Basic types.
  ONNX_TYPE_FLOAT = 1,  // float
  ONNX_TYPE_UINT8 = 2,  // uint8_t
  ONNX_TYPE_INT8 = 3,   // int8_t
  ONNX_TYPE_UINT16 = 4, // uint16_t
  ONNX_TYPE_INT16 = 5,  // int16_t
  ONNX_TYPE_INT32 = 6,  // int32_t
  ONNX_TYPE_INT64 = 7,  // int64_t
  ONNX_TYPE_STRING = 8, // string
  ONNX_TYPE_BOOL = 9,   // bool

  // IEEE754 half-precision floating-point format (16 bits wide).
  // This format has 1 sign bit, 5 exponent bits, and 10 mantissa bits.
  ONNX_TYPE_FLOAT16 = 10,

  ONNX_TYPE_DOUBLE = 11, // double
  ONNX_TYPE_UINT32 = 12, // uint32_t
  ONNX_TYPE_UINT64 = 13, // uint64_t
  ONNX_TYPE_COMPLEX64 =
      14, // complex with float32 real and imaginary components
  ONNX_TYPE_COMPLEX128 =
      15, // complex with float64 real and imaginary components

  // Non-IEEE floating-point format based on IEEE754 single-precision
  // floating-point number truncated to 16 bits.
  // This format has 1 sign bit, 8 exponent bits, and 7 mantissa bits.
  ONNX_TYPE_BFLOAT16 = 16,

  // Future extensions go here.
};

const int RTMEMREF_DATA_TYPE_SIZE[] = {
    0,                // UNDEFINED
    sizeof(float),    // FLOAT
    sizeof(uint8_t),  // UINT8
    sizeof(int8_t),   // INT8
    sizeof(uint16_t), // UINT16
    sizeof(int16_t),  // INT16
    sizeof(int32_t),  // INT32
    sizeof(int64_t),  // INT64
    0,                // STRING
    sizeof(bool),     // BOOL
    2,                // FLOAT16
    sizeof(double),   // DOUBLE
    sizeof(uint32_t), // UINT32
    sizeof(uint64_t), // UINT64
    8,                // COMPLEX64
    16,               // COMPLEX128
    2,                // BFLOAT16
};

// Note by design const map has no [] operator since [] creates a default
// key value mapping when the key is not found which changes the map
const std::map<std::string, int> RTMEMREF_DATA_TYPE_CPP_TO_ONNX = {
    {"b", ONNX_TYPE_BOOL}, // bool     -> BOOL
    {"c",
        ONNX_TYPE_INT8}, // char     -> INT8 (platform dependent, can be UINT8)
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
