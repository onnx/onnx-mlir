
/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------- OMUnique.h - Unique implementation --------===//
//
// Copyright 2014-2026 The IBM Research Authors.
//
//===----------------------------------------------------------------------===//
// LLVM-FREE FILE -- DO NOT ADD LLVM / MLIR / ONNX-MLIR COMPILER DEPENDENCES.
//
// Part of the lightweight onnx-mlir build used for the pip-installable
// packages (om_pyrt). Any LLVM/MLIR reference here will break those packages.
//===----------------------------------------------------------------------===//

#include "onnx-mlir/Runtime/OMTensor.h"
// #include "onnx-mlir/Runtime/OnnxDataType.h"

void omTensorUnique(OMTensor *totalTensor, const OMTensor *inputTensor,
    int64_t inputAxis, uint64_t sorted, OMTensor *Y, OMTensor *indices,
    OMTensor *inverse_indices, OMTensor *counts);
