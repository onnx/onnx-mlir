/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ SmallFPConversion.h -------------------------===//
//
// Conversion to and from 16 bits floating point types.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_SMALLFPCONVERSION_H
#define ONNX_MLIR_SMALLFPCONVERSION_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef ONNX_MLIR_HAS_Float16

inline float om_f16_to_f32(uint16_t u16) { return *(_Float16 *)&u16; }

inline uint16_t om_f32_to_f16(float f32) {
  _Float16 f16 = f32;
  return *(uint16_t *)&f16;
}

#else // ONNX_MLIR_HAS_Float16

float om_f16_to_f32(uint16_t u16);

uint16_t om_f32_to_f16(float f32);

#endif // ONNX_MLIR_HAS_Float16

float om_bf16_to_f32(uint16_t u16);

uint16_t om_f32_to_bf16(float f32);

#ifdef __cplusplus
}
#endif

#endif // ONNX_MLIR_SMALLFPCONVERSION_H
