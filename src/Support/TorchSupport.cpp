/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====---------- KrnlSupport.cpp - Krnl-level support functions -----------===//
//
// Copyright 2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains support code used at the level of the KRNL dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Support/TorchSupport.hpp"

using namespace mlir;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Return various operations.
//===----------------------------------------------------------------------===//

typedef struct dim_pads {
  int dim_start;
  int dim_end;
} dim_pads;

std::vector<IntegerAttr> setUpSymmetricPadding(
    ::mlir::ArrayAttr &pads, Type ty) {
  dim_pads dimArray[pads.size()];
  std::vector<IntegerAttr> translatepadsList;

  bool is_symmetric = true;
  for (unsigned int i = 0; i < pads.size(); i += 2) {
    if (pads[i] != pads[i + 1]) {
      is_symmetric = false;
      break;
    }
  }
  assert(
      is_symmetric && "Frontend transformations only handle symmetric padding");

  if (is_symmetric) {
    for (unsigned int i = 0; i < pads.size(); i += 2) {
      auto pad_value =
          (pads[i].dyn_cast<IntegerAttr>()).getValue().getZExtValue();
      auto f0 = IntegerAttr::get(ty, pad_value);
      translatepadsList.push_back(f0);
    }
  } else {
    /* ------------------ TODO:: Handle Asymmetric Padding in the future
    ---------- int j = 0; for (unsigned int i = 0; i < pads.size(); i++) {
      dimArray[j].dim_start =
        (pads[i].dyn_cast<IntegerAttr>()).getValue().getZExtValue();
      i++;
      dimArray[j].dim_end =
        (pads[i].dyn_cast<IntegerAttr>()).getValue().getZExtValue();
      j++;
    }

    // read the onnx pad values from array(dim_start values)
    int k = 0;
    for (unsigned int i = 0; i < pads.size(); i = i + 2) {
      auto f0 = IntegerAttr::get(ty, (dimArray[k].dim_start));
      Value p0v = rewriter.create<arith::ConstantIntOp>(loc, f0);
      translatepadsList.push_back(p0v);
      k++;
    }

    // read the onnx pad values from array(dim_end values)
    k = 0;
    for (unsigned int i = 0; i < pads.size(); i = i + 2) {
      auto f1 = IntegerAttr::get(ty, (dimArray[k].dim_end));
      Value p1v = rewriter.create<arith::ConstantIntOp>(loc, f1);
      translatepadsList.push_back(p1v);
      k++;
    }
    */
  }

  return translatepadsList;
}
