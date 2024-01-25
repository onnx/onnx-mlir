/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- LSTM.hpp - Lowering LSTM Op --------------------------===//
//
// Copyright 2024
//
// =============================================================================
//
// This file includes utilities for lowering the ONNX LSTM Operators.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXConversionCommon/RNN/RNNBase.hpp"

namespace onnx_mlir {

struct LstmActivationPack {
  RNNActivation f;
  RNNActivation g;
  RNNActivation h;
};

struct LstmWeightPack {
  mlir::Value WT;
  mlir::Value RT;
};

struct LstmBiasPack {
  bool hasBias = false;
  mlir::Value Wbi;
  mlir::Value Wbo;
  mlir::Value Wbf;
  mlir::Value Wbc;
  mlir::Value Rbi;
  mlir::Value Rbo;
  mlir::Value Rbf;
  mlir::Value Rbc;
  // Put peephole here.
  bool hasPeephole = false;
  mlir::Value Pi;
  mlir::Value Po;
  mlir::Value Pf;
};

template <>
bool hasAllNoneOutput<mlir::ONNXLSTMOp>(mlir::ONNXLSTMOp *op);

template <>
std::tuple<LstmActivationPack, LstmActivationPack>
getActivationPack<mlir::ONNXLSTMOp, LstmActivationPack>(mlir::ONNXLSTMOp *op);

} // namespace onnx_mlir
