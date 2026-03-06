
/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------- OMUnique.h - Unique implementation --------===//
//
// Copyright 2014-2026 The IBM Research Authors.
//

#include "onnx-mlir/Runtime/OMTensor.h"
// #include "onnx-mlir/Runtime/OnnxDataType.h"

void omTensorUnique(OMTensor *totalTensor, const OMTensor *inputTensor,
    int64_t inputAxis, uint64_t sorted, OMTensor *Y, OMTensor *indices,
    OMTensor *inverse_indices, OMTensor *counts);
