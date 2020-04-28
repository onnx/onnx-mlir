//====------ ONNXOpsHelper.hpp - Helper functions for ONNX dialects -------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file contains helper functions for lowering ONNX ops to Krnl Dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"

using namespace mlir;

// Identity affine map
AffineMap getIdentityDimMap(Builder &builder);

// Pool/conv affine map
AffineMap getConvDimMap(Builder &builder, bool ceilMode);
