/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------- ONNXOps.hpp - ONNX Operations -------------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file defines ONNX operations in the MLIR operation set.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/OpDefinition.h"
#include "src/Dialect/ONNX/ONNXAttributes.hpp"
#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Dialect/ONNX/ONNXTypes.hpp"
#include "src/Interface/HasOnnxSubgraphOpInterface.hpp"
#include "src/Interface/ONNXOperationTrait.hpp"
#include "src/Interface/ResultTypeInferenceOpInterface.hpp"
#include "src/Interface/ShapeInferenceOpInterface.hpp"

#include <variant>

#define GET_OP_CLASSES
#include "src/Dialect/ONNX/ONNXOps.hpp.inc"

namespace mlir {
// OpSet level supported by onnx-mlir
static constexpr int CURRENT_ONNX_OPSET = 17;

namespace detail {
using ONNXOpsT = std::variant<
#define GET_OP_LIST
#include "src/Dialect/ONNX/ONNXOps.cpp.inc"
    >;

template <typename Action, typename... Ts>
inline void foreachONNXOpImpl(Action &&act, std::variant<Ts...>) {
  (act(Ts()), ...);
}
} // namespace detail

template <typename OP>
struct OpTypeToken {
  using OpType = OP;
};

template <typename Action>
inline void foreachONNXOp(Action &&act) {
  return detail::foreachONNXOpImpl(act, detail::ONNXOpsT());
}

} // end namespace mlir
