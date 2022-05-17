/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- ZHighConstPropagation.cpp - ZHigh High Level Optimizer ------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a set of simple combiners for optimizing operations in
// the ZHigh dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Accelerators/NNPA/Pass/NNPAPasses.hpp"
#include "src/Accelerators/NNPA/Support/LayoutHelper.hpp"
#include "src/Accelerators/NNPA/Transform/ZHigh/Stickify/Stickify.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;
using namespace onnx_mlir;
using namespace onnx_mlir::zhigh;

namespace onnx_mlir {
namespace zhigh {

/// A helper function to check whether a value is produced by a dense
/// ONNXConstantOp.
bool isFromDenseONNXConstantOp(Value result) {
  Operation *op = result.getDefiningOp();

  ONNXConstantOp constOp = llvm::dyn_cast_or_null<ONNXConstantOp>(op);
  // Not a constant.
  if (!constOp)
    return false;

  // The dense attribute must be available.
  if (!(op->getAttrOfType<::mlir::Attribute>("value")))
    return false;
  else {
    DenseElementsAttr denseAttr =
        op->getAttrOfType<::mlir::Attribute>("value").cast<DenseElementsAttr>();
    if (!denseAttr)
      return false;
  }
  // The other attributes must be null.
  if (op->getAttrOfType<::mlir::Attribute>("sparse_value"))
    return false;
  if (op->getAttrOfType<::mlir::Attribute>("value_float"))
    return false;
  if (op->getAttrOfType<::mlir::Attribute>("value_floats"))
    return false;
  if (op->getAttrOfType<::mlir::Attribute>("value_int"))
    return false;
  if (op->getAttrOfType<::mlir::Attribute>("value_ints"))
    return false;
  if (op->getAttrOfType<::mlir::Attribute>("value_string"))
    return false;
  if (op->getAttrOfType<::mlir::Attribute>("value_strings"))
    return false;

  return true;
}

/// Get raw data from a dense attribute.
static void getRawData(DenseElementsAttr denseAttr, std::vector<char> &data) {
  if (!denseAttr.isSplat()) {
    data = denseAttr.getRawData();
  } else {
    ShapedType denseShapeType = denseAttr.getType().cast<ShapedType>();
    std::vector<char> rawData = denseAttr.getRawData();
    int64_t numElements = denseShapeType.getNumElements();
    for (int i = 0; i < numElements; i++)
      data.insert(data.end(), rawData.begin(), rawData.end());
  }
}

/// MLIR type to zDNN type.
zdnn_data_types mlirTypeToZDNNType(Type elementType) {
  if (elementType.isa<FloatType>()) {
    FloatType floatTy = elementType.cast<FloatType>();
    if (floatTy.getWidth() == 16) {
      return FP16;
    } else if (floatTy.getWidth() == 32) {
      return FP32;
    } else
      llvm_unreachable("Unsupported data type.");
  } else
    llvm_unreachable("Unsupported data type.");
}

/// Emit a ZHighStikifiedConstant using information from a stickified ztensor.
ZHighStickifiedConstantOp emitZHighStickifiedConstant(PatternRewriter &rewriter,
    Location loc, zdnn_ztensor *ztensor, Type outputType) {

  // Create a ZHighStickifiedConstantOp.
  ZHighStickifiedConstantOp stickifiedConstant =
      rewriter.create<ZHighStickifiedConstantOp>(loc, outputType,
          /*value=*/nullptr,
          /*alignment=*/rewriter.getI64IntegerAttr(4096));

  // Use an opaque attribute to store stickified data.
  // Attribute type: tensor<sizeInBytes x i8>
  int64_t sizeInBytes = ztensor->buffer_size;
  OpaqueElementsAttr valueAttr =
      OpaqueElementsAttr::get(stickifiedConstant.getOperation()->getDialect(),
          RankedTensorType::get({sizeInBytes}, rewriter.getI8Type()),
          StringRef((char *)ztensor->buffer, sizeInBytes));

  stickifiedConstant.valueAttr(valueAttr);

  return stickifiedConstant;
}

ZHighStickifiedConstantOp createConstantForStick(PatternRewriter &rewriter,
    Value replacingValue, Value input, StringAttr layout) {
  Location loc = replacingValue.getLoc();
  Operation *op = input.getDefiningOp();
  ArrayRef<int64_t> shape = input.getType().cast<ShapedType>().getShape();
  Type elementType = input.getType().cast<ShapedType>().getElementType();
  int rank = shape.size();

  // Read dense attributes.
  DenseElementsAttr dataAttr = op->getAttrOfType<::mlir::Attribute>("value")
                                   .dyn_cast_or_null<mlir::DenseElementsAttr>();
  assert(dataAttr && "Attribute is null");
  // Read attributes's raw data.
  std::vector<char> rawData;
  getRawData(dataAttr, rawData);
  // assert((rawData.size() == (uint64_t)getMemRefSizeInBytes(input)) &&
  //        "Data size mismatched");

  // Call stickify.
  zdnn_tensor_desc pre_tfrmd_desc, tfrmd_desc;
  // pre-transformed desc.
  zdnn_data_layouts zDNNLayout =
      convertLayoutAttrToZDNNDataLayout(rank, layout);
  zdnn_data_types zDNNType = mlirTypeToZDNNType(elementType);
  set_info_pre_transformed_desc(&pre_tfrmd_desc, zDNNLayout, zDNNType, shape);
  // transformed desc.
  zdnn_status status = generate_transformed_desc(&pre_tfrmd_desc, &tfrmd_desc);
  assert(status == ZDNN_OK);
  // Stick data using the software stickify.
  zdnn_ztensor ztensor;
  init_ztensor(&pre_tfrmd_desc, &tfrmd_desc, &ztensor);
  status = allochelper_ztensor_alloc(&ztensor);
  assert(status == ZDNN_OK);
  status = stickify(&ztensor, rawData.data());
  assert(status == ZDNN_OK);
  // Emit a constant global in ZHigh dialect.
  ZHighStickifiedConstantOp constantOp = emitZHighStickifiedConstant(
      rewriter, loc, &ztensor, replacingValue.getType());

  return constantOp;
}

ZHighStickifiedConstantOp createConstantForStickForLSTM(
    PatternRewriter &rewriter, Value replacingValue, Value inputF, Value inputI,
    Value inputC, Value inputO) {
  Location loc = replacingValue.getLoc();
  Operation *fOp = inputF.getDefiningOp();
  Operation *iOp = inputI.getDefiningOp();
  Operation *cOp = inputC.getDefiningOp();
  Operation *oOp = inputO.getDefiningOp();

  ArrayRef<int64_t> fShape = inputF.getType().cast<ShapedType>().getShape();
  assert((fShape.size() == 2 || fShape.size() == 3) && "Wrong tensor shape");
  Type elementType = inputF.getType().cast<ShapedType>().getElementType();

  // Read dense attributes.
  DenseElementsAttr fDataAttr =
      fOp->getAttrOfType<::mlir::Attribute>("value")
          .dyn_cast_or_null<mlir::DenseElementsAttr>();
  DenseElementsAttr iDataAttr =
      iOp->getAttrOfType<::mlir::Attribute>("value")
          .dyn_cast_or_null<mlir::DenseElementsAttr>();
  DenseElementsAttr cDataAttr =
      cOp->getAttrOfType<::mlir::Attribute>("value")
          .dyn_cast_or_null<mlir::DenseElementsAttr>();
  DenseElementsAttr oDataAttr =
      oOp->getAttrOfType<::mlir::Attribute>("value")
          .dyn_cast_or_null<mlir::DenseElementsAttr>();
  assert((fDataAttr && iDataAttr && cDataAttr && oDataAttr) &&
         "Attribute is null");
  // Read attributes's raw data.
  std::vector<char> rawFData, rawIData, rawCData, rawOData;
  getRawData(fDataAttr, rawFData);
  getRawData(iDataAttr, rawIData);
  getRawData(cDataAttr, rawCData);
  getRawData(oDataAttr, rawOData);

  // Call stickify.
  zdnn_tensor_desc pre_tfrmd_desc, tfrmd_desc;
  // pre-transformed desc.
  int rank = fShape.size();
  zdnn_data_layouts zDNNLayout = (rank == 2) ? ZDNN_2DS : ZDNN_3DS;
  zdnn_data_types zDNNType = mlirTypeToZDNNType(elementType);
  set_info_pre_transformed_desc(&pre_tfrmd_desc, zDNNLayout, zDNNType, fShape);
  // transformed desc.
  zdnn_concat_info concatInfo = RNN_TYPE_LSTM |
                                ((rank == 2) ? USAGE_BIASES : USAGE_WEIGHTS) |
                                PREV_LAYER_NONE;
  zdnn_status status = generate_transformed_desc_concatenated(
      &pre_tfrmd_desc, concatInfo, &tfrmd_desc);
  assert(status == ZDNN_OK);
  // Stick data using the software stickify.
  zdnn_ztensor ztensor;
  init_ztensor(&pre_tfrmd_desc, &tfrmd_desc, &ztensor);
  status = allochelper_ztensor_alloc(&ztensor);
  assert(status == ZDNN_OK);
  status = stickify(&ztensor, rawFData.data(), rawIData.data(), rawCData.data(),
      rawOData.data());
  assert(status == ZDNN_OK);

  // Emit a constant global in ZHigh dialect.
  ZHighStickifiedConstantOp constantOp = emitZHighStickifiedConstant(
      rewriter, loc, &ztensor, replacingValue.getType());

  return constantOp;
}

ZHighStickifiedConstantOp createConstantForStickForGRU(
    PatternRewriter &rewriter, Value replacingValue, Value inputZ, Value inputR,
    Value inputH) {
  Location loc = replacingValue.getLoc();
  Operation *zOp = inputZ.getDefiningOp();
  Operation *rOp = inputR.getDefiningOp();
  Operation *hOp = inputH.getDefiningOp();

  ArrayRef<int64_t> zShape = inputZ.getType().cast<ShapedType>().getShape();
  assert((zShape.size() == 2 || zShape.size() == 3) && "Wrong tensor shape");
  Type elementType = inputZ.getType().cast<ShapedType>().getElementType();

  // Read dense attributes.
  DenseElementsAttr zDataAttr =
      zOp->getAttrOfType<::mlir::Attribute>("value")
          .dyn_cast_or_null<mlir::DenseElementsAttr>();
  DenseElementsAttr rDataAttr =
      rOp->getAttrOfType<::mlir::Attribute>("value")
          .dyn_cast_or_null<mlir::DenseElementsAttr>();
  DenseElementsAttr hDataAttr =
      hOp->getAttrOfType<::mlir::Attribute>("value")
          .dyn_cast_or_null<mlir::DenseElementsAttr>();
  assert((zDataAttr && rDataAttr && hDataAttr) && "Attribute is null");
  // Read attributes's raw data.
  std::vector<char> rawZData, rawHData, rawRData, rawOData;
  getRawData(zDataAttr, rawZData);
  getRawData(rDataAttr, rawRData);
  getRawData(hDataAttr, rawHData);

  // Call stickify.
  zdnn_tensor_desc pre_tfrmd_desc, tfrmd_desc;
  // pre-transformed desc.
  int rank = zShape.size();
  zdnn_data_layouts zDNNLayout = (rank == 2) ? ZDNN_2DS : ZDNN_3DS;
  zdnn_data_types zDNNType = mlirTypeToZDNNType(elementType);
  set_info_pre_transformed_desc(&pre_tfrmd_desc, zDNNLayout, zDNNType, zShape);
  // transformed desc.
  zdnn_concat_info concatInfo = RNN_TYPE_GRU |
                                ((rank == 2) ? USAGE_BIASES : USAGE_WEIGHTS) |
                                PREV_LAYER_NONE;
  zdnn_status status = generate_transformed_desc_concatenated(
      &pre_tfrmd_desc, concatInfo, &tfrmd_desc);
  assert(status == ZDNN_OK);
  // Stick data using the software stickify.
  zdnn_ztensor ztensor;
  init_ztensor(&pre_tfrmd_desc, &tfrmd_desc, &ztensor);
  status = allochelper_ztensor_alloc(&ztensor);
  assert(status == ZDNN_OK);
  status =
      stickify(&ztensor, rawZData.data(), rawRData.data(), rawHData.data());
  assert(status == ZDNN_OK);
  // Emit a constant global in ZHigh dialect.
  ZHighStickifiedConstantOp constantOp = emitZHighStickifiedConstant(
      rewriter, loc, &ztensor, replacingValue.getType());

  return constantOp;
}

//===----------------------------------------------------------------------===//
// ZHigh Stick to Krnl Lowering Pass
//===----------------------------------------------------------------------===//

namespace {
/// Use anonymous namespace to avoid duplication symbol `populateWithGenerated`
/// among multiple tablegen-based definitions.

/// Include the patterns defined in the Declarative Rewrite framework.
#include "src/Accelerators/NNPA/Transform/ZHigh/ONNXZHighConstPropagation.inc"

struct ZHighConstPropagationPass
    //: public PassWrapper<ZHighConstPropagationPass, OperationPass<ModuleOp>> {
    : public PassWrapper<ZHighConstPropagationPass,
          OperationPass<func::FuncOp>> {

  StringRef getArgument() const override { return "constprop-zhigh"; }

  StringRef getDescription() const override {
    return "Constant propagation for ZHigh operations.";
  }

  void runOnOperation() override {
    auto function = getOperation();
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    populateWithGenerated(patterns);
    (void)applyPatternsAndFoldGreedily(function, std::move(patterns));
  }
};
} // anonymous namespace

std::unique_ptr<Pass> createZHighConstPropagationPass() {
  return std::make_unique<ZHighConstPropagationPass>();
}

} // namespace zhigh
} // namespace onnx_mlir
