/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- PaddingOp.cpp ------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// ========================================================================
//
// This file implements a combined pass that dynamically invoke several
// transformation on ONNX ops.
//
//===-----------------------------------------------------------------===//
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "llvm/Support/Casting.h"

#ifdef _WIN32
#include <io.h>
#endif

using namespace mlir;

//  ONNX Pad operation
//    “Given a tensor containing the data to be padded (data),
//    a tensor containing the number of start and end pad values for
//    axis (pads), (optionally) a mode, and (optionally) constant_value,
//    ” “a padded tensor (output) is generated.
//
//  Attributes:
//    mode	::mlir::StringAttr	string attribute
//
//  Operands:
//  data	  tensor of 8-bit/16-bit/32-bit/64-bit unsigned integer values
//  or 	  tensor of 8-bit/16-bit/32-bit/64-bit signless integer values or
//  tensor of bfloat16 type values or tensor of 16-bit/32-bit/64-bit 	  float
//  values or tensor of string type values or tensor of 1-bit 	  signless
//  integer values or tensor of complex type with 	  32-bit/64-bit float
//  elements values or memref of any type values.
//
//  pads   tensor of 64-bit signless integer values or memref of
//  	  any type values.
//
//  constant_value
//  	  tensor of 8-bit/16-bit/32-bit/64-bit unsigned integer values or
//         tensor of 8-bit/16-bit/32-bit/64-bit signless integer values or
//         tensor of bfloat16 type values or tensor of 16-bit/32-bit/64-bit
//         float values or tensor of string type values or tensor of 1-bit
//         signless integer values or tensor of complex type with
//         32-bit/64-bit float elements values or memref of any type values
//         or none type.
//
// Results:
//  output
//         tensor of 8-bit/16-bit/32-bit/64-bit unsigned integer values or
//         tensor of 8-bit/16-bit/32-bit/64-bit signless integer values or
//         tensor of bfloat16 type values or tensor of 16-bit/32-bit/64-bit
//         float values or tensor of string type values or tensor of 1-bit
//         signless integer values or tensor of complex type with
//         32-bit/64-bit float elements values or memref of any type values
//         or none type.
//
//  Validation
//  ----------
//  ./scripts/docker/build_with_docker.py --external-build --build-dir build
// --command
// "build/Ubuntu1804-Release/third-party/onnx-mlir/Release/bin/onnx-mlir
// --EmitONNXIR --debug --run-torch-pass
//  /home/sachin/try10/FlexML/third-party/onnx-mlir/third_party/onnx/onnx/
//  backend/test/data/pytorch-operator/test_operator_pad/model.onnx"

namespace onnx_mlir {

typedef struct dim_pads {
  int dim_start;
  int dim_end;
} dim_pads;

class ONNXPadOpLoweringToTOSA : public OpConversionPattern<ONNXPadOp> {
public:
  using OpConversionPattern<ONNXPadOp>::OpConversionPattern;
  using OpAdaptor = typename ONNXPadOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXPadOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {

    MLIRContext *context = op.getContext();
    Location loc = op.getLoc();

    Value data = adaptor.data();
    Value pads = adaptor.pads();
    Value constValue = adaptor.constant_value();

    if (!(adaptor.mode() == "constant")) {
      return rewriter.notifyMatchFailure(
          op, "Only 'constant' mode is supported");
    }

    // creating the DenseElementsAttr using pads values.
    DenseElementsAttr denseAttr =
        llvm::dyn_cast_or_null<tosa::ConstOp>(pads.getDefiningOp())
            .getValue()
            .dyn_cast<DenseElementsAttr>();

    // Reading the ONNX side pads values and store in the array.
    std::vector<APInt> intValues;
    bool paddingNeeded = false;
    for (auto n : denseAttr.getValues<APInt>()) {
      intValues.push_back(n);
      if (!n.isZero())
        paddingNeeded = true;
    }
    if (!paddingNeeded) {
      // We do not need to represent the no-op pad in the resulting MLIR
      rewriter.replaceOp(op, {data});
      return success();
    }

    // Rearrange the pad values.
    // ONNX : [b1, b2, b3, b4, e1, e2, e3, e4]
    // TOSA :[[b1, e1], [b2, e2], [b3, e3], [b4, e4]]
    dim_pads dimArray[intValues.size() / 2];

    llvm::SmallVector<int64_t, 8> translatePadsList;
    auto intType = IntegerType::get(op.getContext(), 64);
    if (intValues.size() != 0) {
      unsigned int dimSize = intValues.size() / 2;
      unsigned int lastNonZero = 0;
      for (unsigned int i = 0; i < dimSize; i++) {
        dimArray[i].dim_start = intValues[i].getZExtValue();
        dimArray[i].dim_end = intValues[i + dimSize].getZExtValue();
        if (dimArray[i].dim_start != 0 || dimArray[i].dim_end != 0)
          lastNonZero = i;
      }

      // read the onnx pad values from array(dim_start values)
      RankedTensorType type0 = RankedTensorType::get({}, rewriter.getI64Type());
      for (unsigned int i = 0; i < lastNonZero + 1; i++) {
        // auto f0 = DenseElementsAttr::get(type0, (dimArray[i].dim_start));
        // Value p0v = rewriter.create<tosa::ConstOp>(loc, type0, f0);
        translatePadsList.push_back(dimArray[i].dim_start);

        // auto f1 = DenseElementsAttr::get(intType, (dimArray[i].dim_end));
        // Value p1v = rewriter.create<tosa::ConstOp>(loc, type0, f1);
        translatePadsList.push_back(dimArray[i].dim_end);
      }
    }

    DenseElementsAttr valueAttr =
        llvm::dyn_cast_or_null<tosa::ConstOp>(constValue.getDefiningOp())
            .getValue()
            .dyn_cast<DenseElementsAttr>();
    auto valueIt = valueAttr.getValues<FloatAttr>().begin();
    double valueFloat = (*valueIt).cast<FloatAttr>().getValueAsDouble();
    FloatAttr floatVal =
        FloatAttr::get(mlir::FloatType::getF32(context), valueFloat);
    auto constType = RankedTensorType::get({}, rewriter.getF32Type());
    auto constAttr = DenseElementsAttr::get(constType, static_cast<float>(valueFloat));
    Value constTosaTensor =
        rewriter.replaceOpWithNewOp<tosa::ConstOp>(constValue.getDefiningOp(), constType, constAttr);

    const unsigned int numberOfDims = intValues.size() / 2;
    DenseElementsAttr PaddingAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({numberOfDims, 2}, rewriter.getI64Type()),
        translatePadsList);

    Value padsList1 =
        rewriter.create<tosa::ConstOp>(loc, PaddingAttr.getType(), PaddingAttr);

    mlir::Type resultType =
        getTypeConverter()->convertType(op.getResult().getType());

    rewriter.eraseOp(pads.getDefiningOp());

    rewriter.replaceOpWithNewOp<tosa::PadOp>(
        op, resultType, data, padsList1, constTosaTensor);

    return success();
  }
};

void populateLoweringONNXPadOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXPadOpLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir