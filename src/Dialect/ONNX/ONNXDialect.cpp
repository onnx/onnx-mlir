/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- ONNXDialect.hpp ---------------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

//===----------------------------------------------------------------------===//
// ONNX Dialect: TableGen generated implementation
//===----------------------------------------------------------------------===//

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
void ONNXDialect::initialize() {
  // Types and attributes are added in these private methods which are
  // implemented in ONNXTypes.cpp and ONNXAttributes.cpp where they have
  // the necessary access to the underlying storage classes from
  // TableGen generated code in ONNXTypes.cpp.inc and ONNXAttributes.cpp.inc.
  // (This emulates the approach in the mlir builtin dialect.)
  registerTypes();
  registerAttributes();

  addOperations<
#define GET_OP_LIST
#include "src/Dialect/ONNX/ONNXOps.cpp.inc"
      >();
}

// Code for ONNX_Dialect class
#include "src/Dialect/ONNX/ONNXDialect.cpp.inc"
