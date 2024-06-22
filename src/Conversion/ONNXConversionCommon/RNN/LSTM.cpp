/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- LSTM.cpp - Lowering LSTM Op --------------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
// Modifications Copyright 2023-2024
//
// =============================================================================
//
// This file includes utilities for lowering the ONNX LSTM Operators.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXConversionCommon/RNN/LSTM.hpp"

using namespace mlir;

namespace onnx_mlir {

template <>
bool hasAllNoneOutput<ONNXLSTMOp>(ONNXLSTMOp *op) {
  return (isNoneValue(op->getY()) && isNoneValue(op->getYH()) &&
          isNoneValue(op->getYC()));
}

template <>
std::tuple<LstmActivationPack, LstmActivationPack>
getActivationPack<ONNXLSTMOp, LstmActivationPack>(ONNXLSTMOp *op) {
  auto direction = op->getDirection();
  auto activations = op->getActivations();
  auto activationAlpha = op->getActivationAlpha();
  auto activationBeta = op->getActivationBeta();

  LstmActivationPack activationForward, activationReverse;

  // Get activation function name.
  // Default forward functions
  activationForward.f.name = "sigmoid";
  activationForward.g.name = "tanh";
  activationForward.h.name = "tanh";
  // Default backward functions
  activationReverse.f.name = "sigmoid";
  activationReverse.g.name = "tanh";
  activationReverse.h.name = "tanh";
  if (activations) {
    ArrayAttr activationArrAttr = activations.value();
    if (direction == FORWARD || direction == BIDIRECTIONAL) {
      // Forward activations.
      if (activationArrAttr.size() > 0) {
        activationForward.f.name =
            mlir::cast<StringAttr>(activationArrAttr[0]).getValue();
      }
      if (activationArrAttr.size() > 1) {
        activationForward.g.name =
            mlir::cast<StringAttr>(activationArrAttr[1]).getValue();
      }
      if (activationArrAttr.size() > 2) {
        activationForward.h.name =
            mlir::cast<StringAttr>(activationArrAttr[2]).getValue();
      }
    }

    // Reverse activations.
    if (direction == REVERSE || direction == BIDIRECTIONAL) {
      unsigned int startIndex = (direction == REVERSE) ? 0 : 3;
      if (activationArrAttr.size() > startIndex) {
        activationReverse.f.name =
            mlir::cast<StringAttr>(activationArrAttr[startIndex]).getValue();
      }
      if (activationArrAttr.size() > startIndex + 1) {
        activationReverse.g.name =
            mlir::cast<StringAttr>(activationArrAttr[startIndex + 1])
                .getValue();
      }
      if (activationArrAttr.size() > startIndex + 2) {
        activationReverse.h.name =
            mlir::cast<StringAttr>(activationArrAttr[startIndex + 2])
                .getValue();
      }
    }
  }

  // Get alpha attributes.
  if (activationAlpha) {
    ArrayAttr activationArrAttr = activationAlpha.value();
    if (direction == FORWARD || direction == BIDIRECTIONAL) {
      // Forward activations.
      if (activationArrAttr.size() > 0) {
        activationForward.f.alpha = mlir::cast<FloatAttr>(activationArrAttr[0]);
      }
      if (activationArrAttr.size() > 1) {
        activationForward.g.alpha = mlir::cast<FloatAttr>(activationArrAttr[1]);
      }
      if (activationArrAttr.size() > 2) {
        activationForward.h.alpha = mlir::cast<FloatAttr>(activationArrAttr[2]);
      }
    }

    // Reverse activations.
    if (direction == REVERSE || direction == BIDIRECTIONAL) {
      unsigned int startIndex = (direction == REVERSE) ? 0 : 3;
      if (activationArrAttr.size() > startIndex) {
        activationReverse.f.alpha =
            mlir::cast<FloatAttr>(activationArrAttr[startIndex]);
      }
      if (activationArrAttr.size() > startIndex + 1) {
        activationReverse.g.alpha =
            mlir::cast<FloatAttr>(activationArrAttr[startIndex + 1]);
      }
      if (activationArrAttr.size() > startIndex + 2) {
        activationReverse.h.alpha =
            mlir::cast<FloatAttr>(activationArrAttr[startIndex + 2]);
      }
    }
  }

  // Get beta attributes.
  if (activationBeta) {
    ArrayAttr activationArrAttr = activationBeta.value();
    if (direction == FORWARD || direction == BIDIRECTIONAL) {
      // Forward activations.
      if (activationArrAttr.size() > 0) {
        activationForward.f.beta = mlir::cast<FloatAttr>(activationArrAttr[0]);
      }
      if (activationArrAttr.size() > 1) {
        activationForward.g.beta = mlir::cast<FloatAttr>(activationArrAttr[1]);
      }
      if (activationArrAttr.size() > 2) {
        activationForward.h.beta = mlir::cast<FloatAttr>(activationArrAttr[2]);
      }
    }

    // Reverse activations.
    if (direction == REVERSE || direction == BIDIRECTIONAL) {
      unsigned int startIndex = (direction == REVERSE) ? 0 : 3;
      if (activationArrAttr.size() > startIndex) {
        activationReverse.f.beta =
            mlir::cast<FloatAttr>(activationArrAttr[startIndex]);
      }
      if (activationArrAttr.size() > startIndex + 1) {
        activationReverse.g.beta =
            mlir::cast<FloatAttr>(activationArrAttr[startIndex + 1]);
      }
      if (activationArrAttr.size() > startIndex + 2) {
        activationReverse.h.beta =
            mlir::cast<FloatAttr>(activationArrAttr[startIndex + 2]);
      }
    }
  }

  return std::make_tuple(activationForward, activationReverse);
}

} // namespace onnx_mlir
