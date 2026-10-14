/*
 * SPDX-License-Identifier: Apache-2.0
 */
// ====-- OnnxDataTypeMetaData.inc ----------------------------------------===//
//
// Copyright 2025-2026 The IBM Research Authors.
//
// =============================================================================
//
// This file contains macro definition for Onnx Mlir Meta Data Type info.
//
//===----------------------------------------------------------------------===//

#if defined(OM_TYPE_METADATA_DEF)
// clang-format off
// Data type metadata declared in the following format:
// OM_TYPE_METADATA_DEF(dtype enum name, dtype enum value, dtype size, dtype name, mlir name, numpy name)
// dtype enum values are standard ONNX data types defined in
// https://github.com/onnx/onnx/blob/main/onnx/onnx.proto#L484
//                   enum name,              val, dtype size,        C++ dtype name,    mlir name,     numpy name
OM_TYPE_METADATA_DEF(ONNX_TYPE_UNDEFINED,      0,  0,                "undefined",       "undefined",   "")
OM_TYPE_METADATA_DEF(ONNX_TYPE_FLOAT,          1,  sizeof(float),    "float",           "f32",         "float32")
OM_TYPE_METADATA_DEF(ONNX_TYPE_UINT8,          2,  sizeof(uint8_t),  "uint8_t",         "ui8",         "uint8")
OM_TYPE_METADATA_DEF(ONNX_TYPE_INT8,           3,  sizeof(int8_t),   "int8_t",          "i8",          "int8")
OM_TYPE_METADATA_DEF(ONNX_TYPE_UINT16,         4,  sizeof(uint16_t), "uint16_t",        "ui16",        "uint16")
OM_TYPE_METADATA_DEF(ONNX_TYPE_INT16,          5,  sizeof(int16_t),  "int16_t",         "i16",         "int16")
OM_TYPE_METADATA_DEF(ONNX_TYPE_INT32,          6,  sizeof(int32_t),  "int32_t",         "i32",         "int32")
OM_TYPE_METADATA_DEF(ONNX_TYPE_INT64,          7,  sizeof(int64_t),  "int64_t",         "i64",         "int64")
OM_TYPE_METADATA_DEF(ONNX_TYPE_STRING,         8,  sizeof(const char *), "const char *","string",      "str")
OM_TYPE_METADATA_DEF(ONNX_TYPE_BOOL,           9,  sizeof(bool),     "_Bool",           "i1",          "bool_")
OM_TYPE_METADATA_DEF(ONNX_TYPE_FLOAT16,        10, 2,                "_Float16",        "f16",         "float16")
OM_TYPE_METADATA_DEF(ONNX_TYPE_DOUBLE,         11, sizeof(double),   "double",          "f64",         "float64")
OM_TYPE_METADATA_DEF(ONNX_TYPE_UINT32,         12, sizeof(uint32_t), "uint32_t",        "ui32",        "uint32")
OM_TYPE_METADATA_DEF(ONNX_TYPE_UINT64,         13, sizeof(uint64_t), "uint64_t",        "ui64",        "uint64")
OM_TYPE_METADATA_DEF(ONNX_TYPE_COMPLEX64,      14, 8,                "_Complex float",  "complex64",   "complex64")
OM_TYPE_METADATA_DEF(ONNX_TYPE_COMPLEX128,     15, 16,               "_Complex double", "complex128",  "complex128")
OM_TYPE_METADATA_DEF(ONNX_TYPE_BFLOAT16,       16, 2,                "bfloat16",        "bf16",        "bfloat16")  /* C++23, before name only. */
OM_TYPE_METADATA_DEF(ONNX_TYPE_FLOAT8E4M3FN,   17, 1,                "FLOAT8E4M3FN",    "f8e4m3fn",   "")           /* No C/C++ name. */
OM_TYPE_METADATA_DEF(ONNX_TYPE_FLOAT8E4M3FNUZ, 18, 1,                "FLOAT8E4M3FNUZ",  "f8e4m3fnuz", "")           /* No C/C++ name. */
OM_TYPE_METADATA_DEF(ONNX_TYPE_FLOAT8E5M2,     19, 1,                "FLOAT8E5M2",      "f8e5m2",     "")           /* No C/C++ name. */
OM_TYPE_METADATA_DEF(ONNX_TYPE_UINT4,          20, 1 /*sub byte*/,   "UINT4",           "ui4",        "")           /* No C/C++ byte size & name. */
OM_TYPE_METADATA_DEF(ONNX_TYPE_INT4,           21, 1 /*sub byte*/,   "INT4",            "i4",         "")           /* No C/C++ byte size & name. */

OM_TYPE_METADATA_DEF(ONNX_TYPE_LAST,           21, 0,                "OUT OF BOUND",    "OUT OF BOUND", "") /* MUST be last enum. */
// clang-format on
#else
#error "Must define OM_TYPE_METADATA_DEF macro."
#endif
