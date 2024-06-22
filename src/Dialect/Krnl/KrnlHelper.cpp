/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====---------------- KrnlHelper.cpp - Krnl Dialect Helper----------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file declares helper methods to build Krnl Dialect Ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"

#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace krnl {

void printDimAndSymbolList(Operation::operand_iterator &begin, unsigned numDims,
    unsigned numSymbols, OpAsmPrinter &p) {
  p << '(';
  p.printOperands(begin, begin + numDims);
  p << ')';

  if (numSymbols) {
    p << '[';
    p.printOperands(begin + numDims, begin + numDims + numSymbols);
    p << ']';
  }

  begin = std::next(begin, numDims + numSymbols);
}

void printBound(AffineMapAttr boundMap,
    Operation::operand_iterator &boundOperandsBeg, const char *prefix,
    OpAsmPrinter &p) {
  AffineMap map = boundMap.getValue();

  // Check if this bound should be printed using custom assembly form.
  // The decision to restrict printing custom assembly form to trivial cases
  // comes from the will to roundtrip MLIR binary -> text -> binary in a
  // lossless way.
  // Therefore, custom assembly form parsing and printing is only supported for
  // zero-operand constant maps and single symbol operand identity maps.
  if (map.getNumResults() == 1) {
    AffineExpr expr = map.getResult(0);

    // Print constant bound.
    if (map.getNumDims() == 0 && map.getNumSymbols() == 0) {
      if (auto constExpr = llvm::dyn_cast<AffineConstantExpr>(expr)) {
        p << constExpr.getValue();
        return;
      }
    }

    // Print bound that consists of a single SSA symbol if the map is over a
    // single symbol.
    if (map.getNumDims() == 0 && map.getNumSymbols() == 1) {
      if (auto symExpr = llvm::dyn_cast<AffineSymbolExpr>(expr)) {
        p.printOperand(*(boundOperandsBeg++));
        return;
      }
    }
  } else {
    // Map has multiple results. Print 'min' or 'max' prefix.
    p << prefix << ' ';
  }

  // Print the map and its operands.
  p << boundMap;
  printDimAndSymbolList(
      boundOperandsBeg, map.getNumDims(), map.getNumSymbols(), p);
}

//====---------------- KrnlIterateOperandPack -----------------------------===//

void KrnlIterateOperandPack::pushConstantBound(int64_t bound) {
  if (boundMaps.size() % 2 == 0)
    operands.emplace_back(inputLoops[boundMaps.size() / 2]);
  AffineMap map = builder.getConstantAffineMap(bound);
  boundMaps.emplace_back(AffineMapAttr::get(map));
}

void KrnlIterateOperandPack::pushOperandBound(Value operand) {
  if (boundMaps.size() % 2 == 0)
    operands.emplace_back(inputLoops[boundMaps.size() / 2]);
  AffineMap map = builder.getSymbolIdentityMap();
  boundMaps.emplace_back(AffineMapAttr::get(map));
  operands.emplace_back(operand);
}

void KrnlIterateOperandPack::pushAffineMapBound(
    AffineMap map, ArrayRef<Value> operands) {
  if (boundMaps.size() % 2 == 0)
    this->operands.emplace_back(inputLoops[boundMaps.size() / 2]);
  boundMaps.emplace_back(AffineMapAttr::get(map));
  for (auto operand : operands)
    this->operands.emplace_back(operand);
}

// Bound could be a constant, Value or AffineMap
void KrnlIterateOperandPack::pushIndexExprBound(IndexExpr expr, bool isLb) {
  if (expr.isLiteral())
    pushConstantBound(expr.getLiteral());
  else if (expr.isAffine() && !expr.isPredType()) {
    AffineMap map;
    SmallVector<Value, 4> list;
    expr.getAffineMapAndOperands(map, list);
    pushAffineMapBound(map, list);
  } else {
    Value val = expr.getValue();
    if ((val.getDefiningOp<affine::AffineMinOp>() && !isLb) ||
        (val.getDefiningOp<affine::AffineMaxOp>() && isLb)) {
      // Have a Affine Min in an upper bound computation, or have an Affine Max
      // in a lower bound computation,  will extract the list of affine min/max
      // for the loop bounds.
      AffineMap map;
      SmallVector<Value, 4> list;
      expr.getAffineMapAndOperands(map, list);
      pushAffineMapBound(map, list);
    } else {
      // Assume the expr is loop invariant if there is any outer loop
      pushOperandBound(val);
    }
  }
}

void KrnlIterateOperandPack::pushIndexExprsBound(
    SmallVectorImpl<IndexExpr> &exprVector) {
  SmallVector<AffineExpr, 4> AEVector;
  // Important to get the affine expressions before getting the num
  // Dim/Symbols as it may add some dims and symbol itself.
  for (IndexExpr expr : exprVector) {
    assert(!expr.isPredType() && "no affine support for predicate type");
    AEVector.push_back(expr.getAffineExpr());
  }
  IndexExpr expr = exprVector.front();
  int dimNum = expr.getScope().getNumDims();
  int symNum = expr.getScope().getNumSymbols();
  AffineMap map =
      AffineMap::get(dimNum, symNum, AEVector, builder.getContext());
  SmallVector<Value, 4> list;
  expr.getScope().getDimAndSymbolList(list);
  pushAffineMapBound(map, list);
}

DenseElementsAttr getDenseElementAttributeFromKrnlValue(Value value) {
  KrnlGlobalOp globalOp =
      dyn_cast_or_null<mlir::KrnlGlobalOp>(value.getDefiningOp());
  if (globalOp)
    if (globalOp.getValue().has_value())
      return mlir::dyn_cast<DenseElementsAttr>(globalOp.getValueAttr());

  return nullptr;
}

//====---------------- Support for simple transpose -------------------===//

// create an identity
void generateIndexMap(
    SmallVectorImpl<int64_t> &map, int64_t size, bool transposeInner2) {
  for (int i = 0; i < size; ++i)
    map.emplace_back(i); // Indentity map.
  if (size < 2)
    return;
  if (transposeInner2) {
    map[size - 2] = size - 1;
    map[size - 1] = size - 2;
  }
}

//====---------------- Common helper functions --------------------------===//

bool isKrnlGlobalConstant(Value result) {
  Operation *op = result.getDefiningOp();
  KrnlGlobalOp constOp = llvm::dyn_cast_or_null<KrnlGlobalOp>(op);
  // Not a constant.
  if (!constOp)
    return false;

  if (!op->getAttrOfType<::mlir::Attribute>("value"))
    return false;

  return true;
}

} // namespace krnl
} // namespace onnx_mlir
