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

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/OpHelper.hpp"
#include "src/Accelerators/NNPA/Pass/NNPAPasses.hpp"
#include "src/Accelerators/NNPA/Support/LayoutHelper.hpp"
#include "src/Accelerators/NNPA/Support/Stickify/Stickify.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/OnnxElementsAttrBuilder.hpp"

using namespace mlir;
using namespace onnx_mlir;
using namespace onnx_mlir::zhigh;

namespace onnx_mlir {
namespace zhigh {

/// Get raw data from a dense attribute.
static void getRawData(ElementsAttr attr_, std::vector<char> &data) {
  ShapedType tensorType = mlir::cast<ShapedType>(attr_.getType());
  Type elemTy = tensorType.getElementType();
  int64_t numElements = tensorType.getNumElements();

  // Use DenseElementsAttr for boolean values. DisposableElementsAttr handles
  // bool differently.
  ElementsAttr attr = attr_;
  if (elemTy.isInteger(1))
    attr = ElementsAttrBuilder::toDenseElementsAttr(attr_);

  auto denseAttr = mlir::dyn_cast_or_null<DenseElementsAttr>(attr);
  auto disposalAttr = mlir::dyn_cast_or_null<DisposableElementsAttr>(attr);
  assert((denseAttr || disposalAttr) &&
         "Must be DenseElementsAttr or DisposableElementsAttr");

  if (disposalAttr) {
    ArrayBuffer<char> dstBytes = disposalAttr.getRawBytes();
    data = dstBytes.get();
    return;
  }

  ArrayRef<char> rawData = denseAttr.getRawData();
  if (denseAttr.isSplat()) {
    // Broadcast the splat value.
    for (int i = 0; i < numElements; i++)
      data.insert(data.end(), rawData.begin(), rawData.end());
  } else {
    data = rawData;
  }
}

/// MLIR type to zDNN type.
zdnn_data_types mlirTypeToZDNNType(Type elementType) {
  if (mlir::isa<FloatType>(elementType)) {
    FloatType floatTy = mlir::cast<FloatType>(elementType);
    if (floatTy.getWidth() == 16) {
      return FP16;
    } else if (floatTy.getWidth() == 32) {
      return FP32;
    } else
      llvm_unreachable("Unsupported data type.");
  } else if (elementType.isInteger(8)) {
    return INT8; // INT8 is accepted by verify_pre_transformed_descriptor
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

  // Attribute type: tensor<sizeInBytes x i8>
  int64_t sizeInBytes = ztensor->buffer_size;

  // Currently, using DenseResourceElementsAttr leads to less memory consumption
  // at compile time.
  // In the future, if there is a need to do constant prop for ZHigh Ops whose
  // inputs are stickified data, then using ElementsAttr is potentially better.
  // In this case, to print or parse ElementsAttr in lit tests,
  // ZHighStickifiedConstantOp would be updated to support custom printer and
  // parser.
  bool useDenseResourceElementsAttr = true;
  if (useDenseResourceElementsAttr) {
    DenseResourceElementsAttr valueAttr = DenseUI8ResourceElementsAttr::get(
        RankedTensorType::get({sizeInBytes}, rewriter.getI8Type()),
        stickifiedConstant.getOperation()
            ->getDialect()
            ->getNamespace(), // use the dialect as the blob "hint"
        HeapAsmResourceBlob::allocateAndCopyWithAlign(
            llvm::ArrayRef((char *)ztensor->buffer, sizeInBytes),
            alignof(char)));
    allochelper_ztensor_free(ztensor);
    stickifiedConstant.setValueAttr(valueAttr);
  } else {
    RankedTensorType dataType =
        RankedTensorType::get({sizeInBytes}, rewriter.getI8Type());
    std::unique_ptr<llvm::MemoryBuffer> memBuf =
        llvm::MemoryBuffer::getMemBuffer(
            StringRef((char *)ztensor->buffer, sizeInBytes), "",
            /*RequiresNullTerminator*/ false);
    ElementsAttr valueAttr = OnnxElementsAttrBuilder(rewriter.getContext())
                                 .fromMemoryBuffer(dataType, std::move(memBuf));
    stickifiedConstant.setValueAttr(valueAttr);
  }

  return stickifiedConstant;
}

ZHighStickifiedConstantOp createConstantForStick(PatternRewriter &rewriter,
    Value replacingValue, Value input, StringAttr layout) {
  Location loc = replacingValue.getLoc();
  ArrayRef<int64_t> shape = mlir::cast<ShapedType>(input.getType()).getShape();
  Type elementType = mlir::cast<ShapedType>(input.getType()).getElementType();
  int rank = shape.size();

  // Read dense attributes.
  ElementsAttr dataAttr = getElementAttributeFromONNXValue(input);
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
  // If zDNNLayout is NHWC, we stickify directly from NCHW.
  if (zDNNLayout == ZDNN_NHWC)
    zDNNLayout = ZDNN_NCHW;
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

bool isFoldableQuantizedStickOp(Value res) {
  ZTensorEncodingAttr::QuantizedType qtype =
      getZTensorQuantizedType(res.getType());
  return (qtype == ZTensorEncodingAttr::QuantizedType::WEIGHTS ||
          qtype == ZTensorEncodingAttr::QuantizedType::INT8);
}

ZHighStickifiedConstantOp createQuantizedConstantForStick(
    PatternRewriter &rewriter, Value replacingValue, Value input,
    Value recScale, Value offset, StringAttr layout, StringAttr quantizeType) {
  Location loc = replacingValue.getLoc();
  ArrayRef<int64_t> shape = mlir::cast<ShapedType>(input.getType()).getShape();
  Type elementType = mlir::cast<ShapedType>(input.getType()).getElementType();
  int rank = shape.size();

  // Read dense attributes.
  ElementsAttr dataAttr = getElementAttributeFromONNXValue(input);
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
  // If zDNNLayout is NHWC, we stickify directly from NCHW.
  if (zDNNLayout == ZDNN_NHWC)
    zDNNLayout = ZDNN_NCHW;
  zdnn_data_types zDNNType = mlirTypeToZDNNType(elementType);
  set_info_pre_transformed_desc(&pre_tfrmd_desc, zDNNLayout, zDNNType, shape);
  // Check the condition for transformed desc.
  // Currently, only QUANTIZED_WEIGHTS_INT8 is supported.
  // The condition of being the weight for QuantizedMatMul has been checked
  // by the matching pattern.
  assert(zDNNType == INT8);
  zdnn_quantized_transform_types transform_type = QUANTIZED_WEIGHTS_INT8;
  zdnn_status status = generate_quantized_transformed_desc(
      &pre_tfrmd_desc, transform_type, &tfrmd_desc);
  assert(status == ZDNN_OK);
  // Stick data using the software stickify.
  zdnn_ztensor ztensor;
  // init_quantized_ztensor can be used if the constant value for recScale and
  // offset is extracted at compile time. However, in the following
  // transformation for the quantized weight tensor, the recScale and offset
  // is not used. The parameters are kept for possible future use.
  init_ztensor(&pre_tfrmd_desc, &tfrmd_desc, &ztensor);
  status = allochelper_ztensor_alloc(&ztensor);
  assert(status == ZDNN_OK);
  status = quantized_stickify(&ztensor, rawData.data());
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

  ArrayRef<int64_t> fShape =
      mlir::cast<ShapedType>(inputF.getType()).getShape();
  assert((fShape.size() == 2 || fShape.size() == 3) && "Wrong tensor shape");
  Type elementType = mlir::cast<ShapedType>(inputF.getType()).getElementType();

  // Read dense attributes.
  ElementsAttr fDataAttr = getElementAttributeFromONNXValue(inputF);
  ElementsAttr iDataAttr = getElementAttributeFromONNXValue(inputI);
  ElementsAttr cDataAttr = getElementAttributeFromONNXValue(inputC);
  ElementsAttr oDataAttr = getElementAttributeFromONNXValue(inputO);
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

  ArrayRef<int64_t> zShape =
      mlir::cast<ShapedType>(inputZ.getType()).getShape();
  assert((zShape.size() == 2 || zShape.size() == 3) && "Wrong tensor shape");
  Type elementType = mlir::cast<ShapedType>(inputZ.getType()).getElementType();

  // Read dense attributes.
  ElementsAttr zDataAttr = getElementAttributeFromONNXValue(inputZ);
  ElementsAttr rDataAttr = getElementAttributeFromONNXValue(inputR);
  ElementsAttr hDataAttr = getElementAttributeFromONNXValue(inputH);
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

static void replaceOpAndGC(
    PatternRewriter &rewriter, Operation *op, ValueRange newValues) {
  for (Value v : op->getOperands()) {
    // v is consumed by only the current stick op.
    if (!v.hasOneUse())
      continue;
    if (auto cop = v.getDefiningOp<ONNXConstantOp>()) {
      if (auto disposableAttr =
              mlir::dyn_cast<DisposableElementsAttr>(cop.getValueAttr())) {
        // Since the current op is the only consummer of the constant,
        // this constant op will be dead soon after the current op is replaced
        // (but the attribute's buffer is not disposed automatically until the
        // next call of garbage collector). So, it's safe to dispose the
        // attribute's buffer now in order to eagerly save memory.
        //
        // Once the buffer is dispose, any touch to the attribute would be
        // invalid. So we just remove it from the constant operation.
        disposableAttr.dispose();
        cop.removeValueAttr();
      }
    }
  }
  rewriter.replaceOp(op, newValues);
}

// zhigh.Stick (c) = krnl.global(c1), where c1 is stickified data.
// Always saturate constants.
struct ConstantStickPattern : public OpRewritePattern<ZHighStickOp> {
  ConstantStickPattern(MLIRContext *context) : OpRewritePattern(context) {}
  LogicalResult matchAndRewrite(
      ZHighStickOp stickOp, PatternRewriter &rewriter) const override {
    Value input = stickOp.getIn();
    Value output = stickOp.getOut();
    StringAttr layout = stickOp.getLayoutAttr();

    // Match
    if (!isDenseONNXConstant(input)) {
      return failure();
    }

    // Rewrite
    Value stickifiedVal =
        createConstantForStick(rewriter, output, input, layout);
    replaceOpAndGC(rewriter, stickOp, stickifiedVal);
    return success();
  }
};

// zhigh.StickForGRU (c1, c2, c3) = krnl.global(c)
// where c is stickified data.
struct ConstantStickForGRUPattern
    : public OpRewritePattern<ZHighStickForGRUOp> {
  ConstantStickForGRUPattern(MLIRContext *context)
      : OpRewritePattern(context) {}
  LogicalResult matchAndRewrite(
      ZHighStickForGRUOp stickOp, PatternRewriter &rewriter) const override {
    Value zGate = stickOp.getZGate();
    Value rGate = stickOp.getRGate();
    Value hGate = stickOp.getHGate();
    Value output = stickOp.getOut();

    // Match
    if (!isDenseONNXConstant(zGate) || !isDenseONNXConstant(rGate) ||
        !isDenseONNXConstant(hGate)) {
      return failure();
    }

    // Rewrite
    Value stickifiedVal =
        createConstantForStickForGRU(rewriter, output, zGate, rGate, hGate);
    replaceOpAndGC(rewriter, stickOp, stickifiedVal);
    return success();
  }
};

// zhigh.StickForLSTM (c1, c2, c3, c4) = krnl.global(c)
// where c is stickified data.
struct ConstantStickForLSTMPattern
    : public OpRewritePattern<ZHighStickForLSTMOp> {
  ConstantStickForLSTMPattern(MLIRContext *context)
      : OpRewritePattern(context) {}
  LogicalResult matchAndRewrite(
      ZHighStickForLSTMOp stickOp, PatternRewriter &rewriter) const override {
    Value fGate = stickOp.getFGate();
    Value iGate = stickOp.getIGate();
    Value cGate = stickOp.getCGate();
    Value oGate = stickOp.getOGate();
    Value output = stickOp.getOut();

    // Match
    if (!isDenseONNXConstant(fGate) || !isDenseONNXConstant(iGate) ||
        !isDenseONNXConstant(cGate) || !isDenseONNXConstant(oGate)) {
      return failure();
    }

    // Rewrite
    Value stickifiedVal = createConstantForStickForLSTM(
        rewriter, output, fGate, iGate, cGate, oGate);
    replaceOpAndGC(rewriter, stickOp, stickifiedVal);
    return success();
  }
};

// zhigh.QuantizedStick (c) = krnl.global(c1), where c1 is stickified data.
// Always saturate constants.
struct ConstantQuantizedStickPattern
    : public OpRewritePattern<ZHighQuantizedStickOp> {
  ConstantQuantizedStickPattern(MLIRContext *context)
      : OpRewritePattern(context) {}
  LogicalResult matchAndRewrite(
      ZHighQuantizedStickOp stickOp, PatternRewriter &rewriter) const override {
    Value input = stickOp.getIn();
    Value recscale = stickOp.getRecScale();
    Value offset = stickOp.getOffset();
    Value output = stickOp.getOut();
    StringAttr layout = stickOp.getLayoutAttr();
    StringAttr quantizedType = stickOp.getQuantizedTypeAttr();

    // Match
    if (!isDenseONNXConstant(input) || !isFoldableQuantizedStickOp(output)) {
      return failure();
    }

    // Rewrite
    Value stickifiedVal = createQuantizedConstantForStick(
        rewriter, output, input, recscale, offset, layout, quantizedType);
    replaceOpAndGC(rewriter, stickOp, {stickifiedVal, recscale, offset});
    return success();
  }
};

struct ZHighConstPropagationPass
    : public PassWrapper<ZHighConstPropagationPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ZHighConstPropagationPass)

  StringRef getArgument() const override { return "constprop-zhigh"; }

  StringRef getDescription() const override {
    return "Constant propagation for ZHigh operations.";
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    patterns.insert<ConstantQuantizedStickPattern>(patterns.getContext());
    patterns.insert<ConstantStickPattern>(patterns.getContext());
    patterns.insert<ConstantStickForGRUPattern>(patterns.getContext());
    patterns.insert<ConstantStickForLSTMPattern>(patterns.getContext());
    (void)applyPatternsAndFoldGreedily(moduleOp, std::move(patterns));
  }
};
} // anonymous namespace

std::unique_ptr<Pass> createZHighConstPropagationPass() {
  return std::make_unique<ZHighConstPropagationPass>();
}

} // namespace zhigh
} // namespace onnx_mlir
