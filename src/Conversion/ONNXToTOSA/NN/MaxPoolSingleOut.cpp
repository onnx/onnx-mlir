/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- MaxPoolSingleOut.cpp - MaxPoolSingleOut Op-----------===//
//
// Copyright (c) 2022-2023 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNX MaxpoolSingleOut operator to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "llvm/ADT/ArrayRef.h"
#include <cstdint>
#include <src/Dialect/Mlir/IndexExpr.hpp>

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXMaxPoolSingleOutOpLoweringToTOSA : public ConversionPattern {
public:
  ONNXMaxPoolSingleOutOpLoweringToTOSA(MLIRContext *ctx)
      : ConversionPattern(ONNXMaxPoolSingleOutOp::getOperationName(), 1, ctx) {}
  using OpAdaptor = typename ONNXMaxPoolSingleOutOp::Adaptor;
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto maxpoolOp = llvm::cast<ONNXMaxPoolSingleOutOp>(op);
    OpAdaptor adaptor(operands, op->getAttrDictionary());

    Value input = adaptor.getX();
    // The attributes storage_order and dilations are unsupported
    IntegerAttr storageOrder = adaptor.getStorageOrderAttr();
    ArrayAttr dilations = adaptor.getDilationsAttr();

    if (mlir::isa<MemRefType>(input.getType())) {
      return rewriter.notifyMatchFailure(
          op, "memrefs as inputs are unsupported by TOSA");
    }
    if (dilations) {
      return rewriter.notifyMatchFailure(
          maxpoolOp, "dilations attribute is unsupported by TOSA");
    }
    if (storageOrder && storageOrder.getSInt() != 0) {
      return rewriter.notifyMatchFailure(
          maxpoolOp, "storage_order attribute is unsupported by TOSA");
    }

    FailureOr<Value> newMaxpoolOp =
        tosa::convertPoolOp<ONNXMaxPoolSingleOutOp, mlir::tosa::MaxPool2dOp>(
            rewriter, op);

    if (failed(newMaxpoolOp)) {
      return rewriter.notifyMatchFailure(
          maxpoolOp, "Could not create maxpool op.");
    }

    rewriter.replaceOp(op, *newMaxpoolOp);
    return success();
  }
};

} // namespace

void populateLoweringONNXMaxPoolSingleOutOpToTOSAPattern(
    ConversionTarget &target, RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXMaxPoolSingleOutOpLoweringToTOSA>(ctx);
}

} // namespace onnx_mlir
