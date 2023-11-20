/*
 * SPDX-License-Identifier: Apache-2.0
 */

#if defined(OM_TYPE_METADATA_DEF)
// clang-format off
// Data type metadata declared in the following format:
// OM_TYPE_METADATA_DEF(dtype enum name, dtype enum value, dtype size, dtype name) 
// dtype enum values are standard ONNX data types defined in
// https://github.com/onnx/onnx/blob/main/onnx/onnx.proto#L484
OM_TYPE_METADATA_DEF(ONNX_TYPE_UNDEFINED,   0,  0,                "undefined")
OM_TYPE_METADATA_DEF(ONNX_TYPE_FLOAT,       1,  sizeof(float),    "float")
OM_TYPE_METADATA_DEF(ONNX_TYPE_UINT8,       2,  sizeof(uint8_t),  "uint8_t")
OM_TYPE_METADATA_DEF(ONNX_TYPE_INT8,        3,  sizeof(int8_t),   "int8_t")
OM_TYPE_METADATA_DEF(ONNX_TYPE_UINT16,      4,  sizeof(uint16_t), "uint16_t")
OM_TYPE_METADATA_DEF(ONNX_TYPE_INT16,       5,  sizeof(int16_t),  "int16_t")
OM_TYPE_METADATA_DEF(ONNX_TYPE_INT32,       6,  sizeof(int32_t),  "int32_t")
OM_TYPE_METADATA_DEF(ONNX_TYPE_INT64,       7,  sizeof(int64_t),  "int64_t")
OM_TYPE_METADATA_DEF(ONNX_TYPE_STRING,      8,  sizeof(const char *), "const char *")
OM_TYPE_METADATA_DEF(ONNX_TYPE_BOOL,        9,  sizeof(bool),     "_Bool")
OM_TYPE_METADATA_DEF(ONNX_TYPE_FLOAT16,     10, 2,                "_Float16")
OM_TYPE_METADATA_DEF(ONNX_TYPE_DOUBLE,      11, sizeof(double),   "double")
OM_TYPE_METADATA_DEF(ONNX_TYPE_UINT32,      12, sizeof(uint32_t), "uint32_t")
OM_TYPE_METADATA_DEF(ONNX_TYPE_UINT64,      13, sizeof(uint64_t), "uint64_t")
OM_TYPE_METADATA_DEF(ONNX_TYPE_COMPLEX64,   14, 8,                "_Complex float")
OM_TYPE_METADATA_DEF(ONNX_TYPE_COMPLEX128,  15, 16,               "_Complex double")
OM_TYPE_METADATA_DEF(ONNX_TYPE_BFLOAT16,    16, 2,                "")
// clang-format on
#else
#error "Must define OM_TYPE_METADATA_DEF macro."
#endif
