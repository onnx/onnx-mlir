/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx-mlir/Runtime/OnnxDataType.h"

const int OM_DATA_TYPE_SIZE[] = {
#define OM_TYPE_METADATA_DEF(ENUM_NAME, ENUM_VAL, DTYPE_SIZE) DTYPE_SIZE,
#include "onnx-mlir/Runtime/OnnxDataTypeMetaData.inc"

#undef OM_TYPE_METADATA_DEF
};