/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ SmallFPConversion.h -------------------------===//
//
// Conversion to and from 16 bits floating point types.
//
//===----------------------------------------------------------------------===//
// LLVM-FREE FILE -- DO NOT ADD LLVM / MLIR / ONNX-MLIR COMPILER DEPENDENCES.
//
// Part of the lightweight onnx-mlir build used for the pip-installable
// packages (om_pyrt). Any LLVM/MLIR reference here will break those packages.
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_SMALLFPCONVERSION_H
#define ONNX_MLIR_SMALLFPCONVERSION_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

float om_f16_to_f32(uint16_t u16);

uint16_t om_f32_to_f16(float f32);

float om_bf16_to_f32(uint16_t u16);

uint16_t om_f32_to_bf16(float f32);

#ifdef __cplusplus
}
#endif

#endif // ONNX_MLIR_SMALLFPCONVERSION_H
