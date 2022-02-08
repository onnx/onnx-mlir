/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ ONNXDialect.cpp - ONNX Operations -----------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Traits.h"
#include "mlir/IR/DialectImplementation.h"

#include "src/Dialect/ONNX/ONNXDialect.hpp"

using namespace mlir;

// Code for ONNX_Dialect class
#include "src/Dialect/ONNX/ONNXDialect.cpp.inc"
