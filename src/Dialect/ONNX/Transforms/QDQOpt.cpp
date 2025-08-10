//===- QDQOpt.cpp - Remove QDQ operations --------*- C++ -*-===//
//
// (c) Copyright 2022 - 2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Pass/Passes.hpp"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include <cmath>

using namespace mlir;
using namespace onnx_mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

static ElementsAttr getElementAttributeFromConstant(Value val) {
  if (!val)
    return nullptr;
  if (auto constOp = val.getDefiningOp<ONNXConstantOp>())
    return mlir::dyn_cast<ElementsAttr>(constOp.getValueAttr());
  return nullptr;
}

static bool scaleParamsMatch(
    Value scale1, Value scale2, double tolerance = 1e-20) {
  auto scaleAttr1 = getElementAttributeFromConstant(scale1);
  auto scaleAttr2 = getElementAttributeFromConstant(scale2);

  if (!scaleAttr1 || !scaleAttr2)
    return false;

  // Shapes and element types must match.
  if (scaleAttr1.getType() != scaleAttr2.getType())
    return false;

  // Get the tensors of floating-point values from the attributes.
  auto scaleVals1 = scaleAttr1.getValues<APFloat>();
  auto scaleVals2 = scaleAttr2.getValues<APFloat>();

  // Check if the tensors have the same size. If not, they don't match.
  if (scaleVals1.size() != scaleVals2.size()) {
    return false;
  }

  // Iterate through the tensors and find the difference between corresponding
  // elements. The code will return false as soon as it finds a difference that
  // is greater than or equal to the tolerance.
  for (auto [v1, v2] : llvm::zip(scaleVals1, scaleVals2)) {
    double diff = std::fabs(v1.convertToDouble() - v2.convertToDouble());
    if (diff >= tolerance) {
      return false;
    }
  }
  // If the loop completes without finding any significant differences, the
  // tensors match.
  return true;
}

static mlir::LogicalResult equalsDefaultIntegerAttr(
    mlir::IntegerAttr ia, int64_t defaultValue) {
  auto it = mlir::cast<mlir::IntegerType>(ia.getType());
  int64_t got = it.isUnsignedInteger()
                    ? static_cast<int64_t>(ia.getValue().getZExtValue())
                    : ia.getValue().getSExtValue();
  return (got == defaultValue) ? mlir::success() : mlir::failure();
}

static mlir::LogicalResult equalsDefaultIntElements(
    mlir::ElementsAttr ea, int64_t defaultValue) {
  auto st = mlir::dyn_cast<mlir::ShapedType>(ea.getType());
  if (!st)
    return mlir::failure();

  mlir::Type et = st.getElementType();
  if (!et.isIntOrIndex())
    return mlir::failure();

  const bool isUnsigned = et.isa<mlir::IntegerType>() &&
                          et.cast<mlir::IntegerType>().isUnsignedInteger();

  if (ea.isSplat()) {
    llvm::APInt api = ea.getSplatValue<llvm::APInt>();
    int64_t got = isUnsigned ? static_cast<int64_t>(api.getZExtValue())
                             : api.getSExtValue();
    return (got == defaultValue) ? mlir::success() : mlir::failure();
  }

  for (const llvm::APInt &api : ea.getValues<llvm::APInt>()) {
    int64_t got = isUnsigned ? static_cast<int64_t>(api.getZExtValue())
                             : api.getSExtValue();
    if (got != defaultValue)
      return mlir::failure();
  }
  return mlir::success();
}

static mlir::LogicalResult checkAttrAgainstDefault(
    mlir::Attribute attr, int64_t defaultValue) {
  if (!attr)
    return mlir::failure();
  if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(attr))
    return equalsDefaultIntegerAttr(ia, defaultValue);
  if (auto ea = mlir::dyn_cast<mlir::ElementsAttr>(attr))
    return equalsDefaultIntElements(ea, defaultValue);
  return mlir::failure();
}

// --- main checker ---

static mlir::LogicalResult checkIntegerAttributeEquals(mlir::Operation *op1,
    mlir::Operation *op2, mlir::StringRef attrName, int64_t defaultValue) {

  mlir::Attribute attr1 = op1->getAttr(attrName);
  mlir::Attribute attr2 = op2->getAttr(attrName);

  // Case 0: both missing => both implicitly default
  if (!attr1 && !attr2)
    return mlir::success();

  // Case 1: both present and identical
  if (attr1 && attr2 && attr1 == attr2)
    return mlir::success();

  // Case 2: one side missing => present side must equal default
  if (!attr1)
    return checkAttrAgainstDefault(attr2, defaultValue);
  if (!attr2)
    return checkAttrAgainstDefault(attr1, defaultValue);

  // Case 3: both present but not identical
  return mlir::failure();
}

//===----------------------------------------------------------------------===//
// Pattern to remove QDQ pairs
//===----------------------------------------------------------------------===//

struct RemoveQDQPattern : public OpRewritePattern<ONNXQuantizeLinearOp> {
  using OpRewritePattern<ONNXQuantizeLinearOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXQuantizeLinearOp qOp, PatternRewriter &rewriter) const override {
    auto dqOp = qOp.getX().getDefiningOp<ONNXDequantizeLinearOp>();
    if (!dqOp)
      return failure();

    // 1. Check attributes with defaults (axis=1, block_size=0,
    // saturate=1)
    Operation *dqOperation = dqOp.getOperation();
    Operation *qOperation = qOp.getOperation();

    if (failed(
            checkIntegerAttributeEquals(dqOperation, qOperation, "axis", 1)) ||
        failed(checkIntegerAttributeEquals(
            dqOperation, qOperation, "block_size", 0)) ||
        failed(checkIntegerAttributeEquals(
            dqOperation, qOperation, "saturate", 1))) {
      return failure();
    }

    // 2. Check zero-points
    auto zpAttr1 = getElementAttributeFromConstant(dqOp.getXZeroPoint());
    auto zpAttr2 = getElementAttributeFromConstant(qOp.getYZeroPoint());
    if (zpAttr1 != zpAttr2)
      return failure();

    // 3. Check Scales with a tolerance
    if (!scaleParamsMatch(dqOp.getXScale(), qOp.getYScale()))
      return failure();

    // 3. Check data types for consistency.
    // The output of DQ must be a float tensor, and the input of Q must be the
    // same float type.
    auto dqOutTypeOp = dqOp.getResult().getType();
    auto qInTypeOp = qOp.getX().getType();

    if (auto dqOutTensorType = dqOutTypeOp.dyn_cast<TensorType>()) {
      if (auto qInTensorType = qInTypeOp.dyn_cast<TensorType>()) {
        if (dqOutTensorType.getElementType() !=
            qInTensorType.getElementType()) {
          return failure();
        }
      } else {
        return failure();
      }
    } else {
      return failure();
    }

    // 4. Check data type consistency of the entire DQ->Q chain.
    // The original quantized type before DQ must match the final quantized type
    // after Q.
    auto dqInTypeOp = dqOp.getX().getType();
    auto qOutTypeOp = qOp.getResult().getType();

    if (auto dqInTensorType = dqInTypeOp.dyn_cast<TensorType>()) {
      if (auto qOutTensorType = qOutTypeOp.dyn_cast<TensorType>()) {
        if (qOutTensorType.getElementType() !=
            dqInTensorType.getElementType()) {
          return failure();
        }
      } else {
        return failure();
      }
    } else {
      return failure();
    }
    rewriter.replaceOp(qOp, dqOp.getX());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass to run QDQ removal
//===----------------------------------------------------------------------===//

struct QDQOptONNXToONNXPass
    : public PassWrapper<QDQOptONNXToONNXPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QDQOptONNXToONNXPass)
  StringRef getArgument() const override { return "qdq-opt-onnx-to-onnx"; }
  StringRef getDescription() const override {
    return "Remove QDQ ops and surrounding QDQ if safe.";
  }

  void runOnOperation() override {
    auto function = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<RemoveQDQPattern>(&getContext());
    if (failed(applyPatternsGreedily(function, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace onnx_mlir {
std::unique_ptr<mlir::Pass> createQDQOptONNXToONNXPass() {
  return std::make_unique<QDQOptONNXToONNXPass>();
}
} // namespace onnx_mlir