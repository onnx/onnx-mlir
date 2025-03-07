/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ZHighToZLow.cpp - ZHigh dialect to ZLow lowering -------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file implements the lowering of ZHigh operations to ZLow operations.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Debug.h"

#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/DialectResourceBlobManager.h"

#include "src/Accelerators/NNPA/Conversion/ZHighToZLow/ProcessStickData.hpp"
#include "src/Accelerators/NNPA/Conversion/ZHighToZLow/ZHighToZLow.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/OpHelper.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/ShapeHelper.hpp"
#include "src/Accelerators/NNPA/Dialect/ZLow/DialectBuilder.hpp"
#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowOps.hpp"
#include "src/Accelerators/NNPA/Pass/NNPAPasses.hpp"
#include "src/Accelerators/NNPA/Support/LayoutHelper.hpp"
#include "src/Accelerators/NNPA/Support/Stickify/Convert.hpp"
#include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Conversion/ONNXToKrnl/Quantization/QuantizeHelper.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

#define DEBUG_TYPE "zhigh-to-zlow"

using namespace mlir;
using namespace onnx_mlir::zlow;

namespace onnx_mlir {
namespace zhigh {

using MDBuilder = MultiDialectBuilder<IndexExprBuilderForKrnl, KrnlBuilder,
    MathBuilder, MemRefBuilder, VectorBuilder, AffineBuilder, SCFBuilder>;

//===----------------------------------------------------------------------===//
// Helper function of Zhigh to Zlow lowering
// Insert an allocation for the given dimensions and layout.
// By default, set alignment to 4K.
//===----------------------------------------------------------------------===//

Value insertAllocForZMemRefByDim(ArrayRef<IndexExpr> dims,
    ZTensorEncodingAttr::DataLayout layout,
    ZTensorEncodingAttr::QuantizedType qtype, Operation *op,
    PatternRewriter &rewriter, int64_t alignment = gAlignment) {
  // Construct a MemRefType for the given dimensions and element type.
  SmallVector<int64_t, 4> shape;
  for (IndexExpr d : dims)
    shape.emplace_back((d.isLiteral() ? d.getLiteral() : ShapedType::kDynamic));
  RankedTensorType tensorType =
      RankedTensorType::get(shape, rewriter.getF32Type(),
          ZTensorEncodingAttr::get(op->getContext(), layout, qtype));
  ZMemRefType zMemRefType = convertZTensorToMemRefType(tensorType);

  // Insert alloc.
  Value alloc =
      insertAllocForZMemRef(zMemRefType, dims, op, rewriter, alignment);

  return alloc;
}

//===----------------------------------------------------------------------===//
// Helper function of Zhigh to Zlow lowering
// Insert an allocation for the given ZMemRefType.
// By default, set alignment to 4K.
//===----------------------------------------------------------------------===//

Value insertAllocForZMemRef(ZMemRefType zType, ArrayRef<IndexExpr> dims,
    Operation *op, PatternRewriter &rewriter, int64_t alignment = gAlignment) {

  Location loc = op->getLoc();
  MemRefType resType = zType.value;

  // Insert alloc.
  SmallVector<IndexExpr> dimList(dims.begin(), dims.end());
  MultiDialectBuilder<MemRefBuilder> create(rewriter, loc);
  return create.mem.alignedAlloc(resType, dimList, alignment);
}

/// Insert allocation for a 4K-aligned buffer of type
/// <sizexi8> to be used as work_area in LSTM/GRU, where size is computed as
/// follows.
//
/// Enough contiguous storage for the following stickified ztensors
/// dimensions:
///     Fused ztensor
///         dim1 = hidden_state_size
///         dim2 = batch
///         dim3 = 1
///         dim4 = numOfGates * timestep
///
///     Bias Add ztensor
///         dim1 = hidden_state_size
///         dim2 = batch
///         dim3 = 1
///         dim4 = numOfGates
///
///     C output ztensor
///         dim1 = hidden_state_size
///         dim2 = batch
///         dim3 = 1
///         dim4 = 2
///
/// For bidirectional, twice the amount of contiguous storage is required.
///
/// The start of the buffer must be 4k aligned. The work area size is
/// computed as follows.
///   zdnn_tensor_desc desc;
///   desc.dim4 = (numOfGates * timestep) + numOfGates + 2;
///   desc.dim3 = 1;
///   desc.dim2 = batch;
///   desc.dim1 = hidden_state_size;
///   uint64_t work_area_size = zdnn_getsize_ztensor(&desc);
///
/// It is unfolded to:
/// work_area_size in bytes = dim4 * dim3 *
///                           CEIL(dim2, AIU_STICKS_PER_PAGE) *
///                           CEIL(dim1, AIU_2BYTE_CELLS_PER_STICK) *
///                           AIU_PAGESIZE_IN_BYTES;
/// where CEIL(a, b) = (a + b - 1) / b,
///       AIU_STICKS_PER_PAGE = 32,
///       AIU_2BYTE_CELLS_PER_STICK = 64,
///       AIU_PAGESIZE_IN_BYTES = 4K.
///
/// timestep and batchsize are obtained from the LSTM/GRU input tensor.
/// hidden_size is obtained from the LSTM/GRU initial hidden tensor.
static Value insertAllocForWorkAreaForRNNOps(IndexExprBuilderForKrnl &createIE,
    PatternRewriter &rewriter, Location loc, Value rnnInput,
    Value rnnHiddenWeight, unsigned numOfGates, bool isDouble) {
  SmallVector<IndexExpr, 4> inputDims, hiddenWeightDims;
  createIE.getShapeAsDims(rnnInput, inputDims);
  createIE.getShapeAsDims(rnnHiddenWeight, hiddenWeightDims);

  IndexExpr timestepExp = inputDims[0];
  IndexExpr Lit2 = LitIE(2);
  IndexExpr NumOfGatesLit = LitIE(numOfGates);
  IndexExpr dim1 = hiddenWeightDims[1];
  IndexExpr dim2 = inputDims[1];
  IndexExpr dim3 = LitIE(1);
  IndexExpr dim4 = NumOfGatesLit * timestepExp + NumOfGatesLit + Lit2;

  IndexExpr Lit1 = LitIE(1);
  IndexExpr Lit32 = LitIE(32);
  IndexExpr Lit64 = LitIE(64);
  IndexExpr Lit4K = LitIE(4096);
  IndexExpr ceilDim2 = (dim2 + Lit32 - Lit1).floorDiv(Lit32);
  IndexExpr ceilDim1 = (dim1 + Lit64 - Lit1).floorDiv(Lit64);
  IndexExpr sizeExpr = dim4 * dim3 * ceilDim2 * ceilDim1 * Lit4K;

  // Double the work area if required.
  if (isDouble)
    sizeExpr = sizeExpr * Lit2;

  // Emit alloc ops.
  int64_t size =
      sizeExpr.isLiteral() ? sizeExpr.getLiteral() : ShapedType::kDynamic;
  MemRefType resultType = MemRefType::get({size}, rewriter.getIntegerType(8));
  SmallVector<IndexExpr> dims(1, sizeExpr);
  MultiDialectBuilder<MemRefBuilder> create(rewriter, loc);
  return create.mem.alignedAlloc(resultType, dims, gAlignment);
}

/// Get a dense resource attribute to store stickified data of a given i8 value.
/// Attribute type: tensor<sizeInBytes x i8>
DenseResourceElementsAttr getDenseResourceElementsAttrOfValue(
    PatternRewriter &rewriter, ZHighStickifiedConstantOp stickifiedConstant,
    int8_t val, int64_t sizeInBytes) {
  char *rawData = static_cast<char *>(malloc(sizeInBytes));
  assert(rawData && "failed to allocate memory for stickified data");
  memset(rawData, val, sizeInBytes);
  DenseResourceElementsAttr valueAttr = DenseUI8ResourceElementsAttr::get(
      RankedTensorType::get({sizeInBytes}, rewriter.getI8Type()),
      stickifiedConstant.getOperation()
          ->getDialect()
          ->getNamespace(), // use the dialect as the blob "hint"
      HeapAsmResourceBlob::allocateAndCopyWithAlign(
          llvm::ArrayRef(rawData, sizeInBytes), alignof(char)));
  free(rawData);
  return valueAttr;
}

/// This function emits a buffer of zero elements for the given dimensions and
/// layout. If the given dimensions are static, then a stickified constant is
/// returned.
Value insertAllocOrEmitZeroConstant(ArrayRef<IndexExpr> dims,
    ZTensorEncodingAttr::DataLayout layout, Operation *op,
    PatternRewriter &rewriter, Location loc,
    ZTensorEncodingAttr::QuantizedType qtype =
        ZTensorEncodingAttr::QuantizedType::UNDEFINED) {
  Value res;
  bool allStaticDims =
      llvm::all_of(dims, [](IndexExpr ie) { return ie.isLiteral(); });
  if (allStaticDims) {
    // Construct a MemRefType for the given dimensions and element type.
    SmallVector<int64_t, 4> shape;
    for (IndexExpr d : dims)
      shape.emplace_back(d.getLiteral());
    Type elemType = rewriter.getF32Type();
    if (qtype == ZTensorEncodingAttr::QuantizedType::WEIGHTS ||
        qtype == ZTensorEncodingAttr::QuantizedType::INT8)
      elemType = rewriter.getI8Type();
    RankedTensorType tensorType = RankedTensorType::get(shape, elemType,
        ZTensorEncodingAttr::get(op->getContext(), layout, qtype));
    ZMemRefType zMemRefType = convertZTensorToMemRefType(tensorType);
    MemRefType resType =
        affine::normalizeMemRefType(mlir::cast<MemRefType>(zMemRefType.value));

    // Create a ZHighStickifiedConstantOp.
    ZHighStickifiedConstantOp stickifiedConstant =
        rewriter.create<ZHighStickifiedConstantOp>(loc, resType,
            /*value=*/nullptr,
            /*alignment=*/rewriter.getI64IntegerAttr(4096));

    // Use an dense resource attribute to store stickified data.
    // Attribute type: tensor<sizeInBytes x i8>
    int64_t sizeInBytes =
        affine::getIntOrFloatMemRefSizeInBytes(resType).value();
    DenseResourceElementsAttr valueAttr = getDenseResourceElementsAttrOfValue(
        rewriter, stickifiedConstant, 0, sizeInBytes);
    stickifiedConstant.setValueAttr(valueAttr);
    res = stickifiedConstant.getResult();
  } else {
    MultiDialectBuilder<KrnlBuilder, MathBuilder> create(rewriter, loc);
    res = insertAllocForZMemRefByDim(dims, layout, qtype, op, rewriter);
    Value initValue = create.math.constant(getElementType(res.getType()), 0);
    create.krnl.memset(res, initValue, /*delayed=*/true);
  }
  return res;
}

/// Emit instructions to allocate a buffer to store original dimensions.
Value insertShapeMemRefI64(
    PatternRewriter &rewriter, Location loc, ArrayRef<IndexExpr> originalDims) {
  MultiDialectBuilder<KrnlBuilder, MathBuilder, MemRefBuilder> create(
      rewriter, loc);
  MemRefType shapeMemRefType = MemRefType::get(
      {static_cast<int64_t>(originalDims.size())}, rewriter.getIntegerType(64));
  Value shapeMemRef = create.mem.alignedAlloc(shapeMemRefType);
  for (uint64_t i = 0; i < originalDims.size(); ++i) {
    Value dim =
        create.math.cast(rewriter.getI64Type(), originalDims[i].getValue());
    create.krnl.storeIE(dim, shapeMemRef, {LitIE(i)});
  }
  return shapeMemRef;
}

/// Get the corresponding MemRefType and layout of a given ZTensorType.
ZMemRefType convertZTensorToMemRefType(Type type) {
  ZMemRefType resZMemRefType;
  if (mlir::isa<TensorType>(type)) {
    OpBuilder b(type.getContext());
    RankedTensorType tensorType = mlir::dyn_cast<RankedTensorType>(type);
    assert(tensorType && "expected only ranked shapes");
    ArrayRef<int64_t> shape = tensorType.getShape();
    Type elementType = tensorType.getElementType();
    int64_t rank = shape.size();
    if (tensorType.getEncoding()) {
      // Obtain element type.
      ZTensorEncodingAttr::QuantizedType qtype = getZTensorQuantizedType(type);
      if (qtype == ZTensorEncodingAttr::QuantizedType::DLFLOAT16)
        elementType = b.getF16Type();
      else if (qtype == ZTensorEncodingAttr::QuantizedType::INT8)
        elementType = b.getI8Type();
      else if (qtype == ZTensorEncodingAttr::QuantizedType::WEIGHTS)
        elementType = b.getI8Type();
      else
        elementType = b.getF16Type();
      // Obtain affine map.
      AffineExpr constExpr0 = getAffineConstantExpr(0, b.getContext());
      AffineExpr constExpr31 = getAffineConstantExpr(31, b.getContext());
      AffineExpr constE2Block = getAffineConstantExpr(32, b.getContext());
      AffineExpr constE1Block = getAffineConstantExpr(64, b.getContext());
      if (qtype == ZTensorEncodingAttr::QuantizedType::INT8) {
        // For quantized i8, 128 cells per stick.
        constE1Block = getAffineConstantExpr(128, b.getContext());
      } else if (qtype == ZTensorEncodingAttr::QuantizedType::WEIGHTS) {
        // WEIGHTS has two vectors interleaved, therefore only 64 cells vs 128
        // Due to this interleaving, number_of_sticks is halved, but must be
        // rounded up to stay even for proper interleaving.
        constE2Block = getAffineConstantExpr(64, b.getContext());
      }

      unsigned e4, e3, e2, e1;
      AffineExpr n, c, h, w, res32, res64;
      SmallVector<AffineExpr, 6> dimExpr;

      ZTensorEncodingAttr::DataLayout layout = getZTensorLayout(tensorType);
      if (layout == ZTensorEncodingAttr::DataLayout::_1D) {
        // (e1) -> (1, 1, 1, e1) -> (1, ceil(e1/64), 1, 1, 32, 64)
        e1 = 0;
        n = constExpr0;
        h = b.getAffineDimExpr(e1).floorDiv(constE1Block);
        w = constExpr0;
        c = constExpr0;
        res32 = constExpr31;
        res64 = b.getAffineDimExpr(e1) % constE1Block;
      } else if (layout == ZTensorEncodingAttr::DataLayout::_2D) {
        // (e2, e1) -> (1, 1, e2, e1) -> (1, ceil(e1/64), 1, ceil(e2/32), 32
        // 64)
        e2 = 0;
        e1 = 1;
        n = constExpr0;
        h = b.getAffineDimExpr(e1).floorDiv(constE1Block);
        w = constExpr0;
        c = b.getAffineDimExpr(e2).floorDiv(constE2Block);
        res32 = b.getAffineDimExpr(e2) % constE2Block;
        res64 = b.getAffineDimExpr(e1) % constE1Block;
      } else if (layout == ZTensorEncodingAttr::DataLayout::_3D) {
        // (e3, e2, e1) -> (1, e3, e2, e1)
        // -> (1, ceil(e1/64), e3, ceil(e2/32), 32, 64)
        e3 = 0;
        e2 = 1;
        e1 = 2;
        n = constExpr0;
        h = b.getAffineDimExpr(e1).floorDiv(constE1Block);
        w = b.getAffineDimExpr(e3);
        c = b.getAffineDimExpr(e2).floorDiv(constE2Block);
        res32 = b.getAffineDimExpr(e2) % constE2Block;
        res64 = b.getAffineDimExpr(e1) % constE1Block;
      } else if (layout == ZTensorEncodingAttr::DataLayout::_4D) {
        // (e4, e3, e2, e1) -> (e4, ceil(e1/64), e3, ceil(e2/32), 32, 64)
        e4 = 0;
        e3 = 1;
        e2 = 2;
        e1 = 3;
        n = b.getAffineDimExpr(e4);
        h = b.getAffineDimExpr(e1).floorDiv(constE1Block);
        w = b.getAffineDimExpr(e3);
        c = b.getAffineDimExpr(e2).floorDiv(constE2Block);
        res32 = b.getAffineDimExpr(e2) % constE2Block;
        res64 = b.getAffineDimExpr(e1) % constE1Block;
      } else if (layout == ZTensorEncodingAttr::DataLayout::_2DS) {
        // (e4, e1) -> (e4, 1, 1, e1) -> (e4, ceil(e1/64), 1, 1, 32, 64)
        e4 = 0;
        e1 = 1;
        n = b.getAffineDimExpr(e4);
        h = b.getAffineDimExpr(e1).floorDiv(constE1Block);
        w = constExpr0;
        c = constExpr0;
        res32 = constExpr31;
        res64 = b.getAffineDimExpr(e1) % constE1Block;
      } else if (layout == ZTensorEncodingAttr::DataLayout::_3DS) {
        // (e4, e2, e1) -> (e4, 1, e2, e1)
        // -> (e4, ceil(e1/64), 1, ceil(e2/32), 32, 64)
        e4 = 0;
        e2 = 1;
        e1 = 2;
        n = b.getAffineDimExpr(e4);
        h = b.getAffineDimExpr(e1).floorDiv(constE1Block);
        w = constExpr0;
        c = b.getAffineDimExpr(e2).floorDiv(constE2Block);
        res32 = b.getAffineDimExpr(e2) % constE2Block;
        res64 = b.getAffineDimExpr(e1) % constE1Block;
      } else if (layout == ZTensorEncodingAttr::DataLayout::_4DS) {
        // for normal
        // (e4, e3, e2, e1)
        // -> (e4, ceil(e1/64), e3, ceil(e2/32), 32, 64)
        // for bidirectional rnn
        // (e4, e3, e2, e1)
        // -> (e4, ceil((2 * PADDED(e1))/64), e3, ceil(e2/32), 32, 64)
        assert((shape[1] == 1 || shape[1] == 2) &&
               "wrong direction dimension size");
        e4 = 0;
        e3 = 1;
        e2 = 2;
        e1 = 3;
        n = b.getAffineDimExpr(e4);
        if (shape[1] == 1) {
          h = b.getAffineDimExpr(e1).floorDiv(constE1Block);
        } else {
          AffineExpr padded_e1 =
              b.getAffineDimExpr(e1).ceilDiv(constE1Block) * constE1Block;
          h = (2 * padded_e1).floorDiv(constE1Block);
        }
        w = b.getAffineDimExpr(e3);
        c = b.getAffineDimExpr(e2).floorDiv(constE2Block);
        res32 = b.getAffineDimExpr(e2) % constE2Block;
        res64 = b.getAffineDimExpr(e1) % constE1Block;
      } else if (layout == ZTensorEncodingAttr::DataLayout::NHWC) {
        // (e4, e3, e2, e1) -> (e4, ceil(e1/64), e3, ceil(e2/32), 32, 64)
        e4 = 0;
        e3 = 1;
        e2 = 2;
        e1 = 3;
        n = b.getAffineDimExpr(e4);
        h = b.getAffineDimExpr(e1).floorDiv(constE1Block);
        w = b.getAffineDimExpr(e3);
        c = b.getAffineDimExpr(e2).floorDiv(constE2Block);
        res32 = b.getAffineDimExpr(e2) % constE2Block;
        res64 = b.getAffineDimExpr(e1) % constE1Block;
      } else if (layout == ZTensorEncodingAttr::DataLayout::NCHW) {
        // (e4, e3, e2, e1) -> (e4, ceil(e2/64), e1, ceil(e3/32), 32, 64)
        llvm_unreachable("Not tested yet");
        e4 = 0;
        e3 = 1;
        e2 = 2;
        e1 = 3;
        n = b.getAffineDimExpr(e4);
        h = b.getAffineDimExpr(e2).floorDiv(constE1Block);
        w = b.getAffineDimExpr(e1);
        c = b.getAffineDimExpr(e3).floorDiv(constE2Block);
        res32 = b.getAffineDimExpr(e3) % constE2Block;
        res64 = b.getAffineDimExpr(e2) % constE1Block;
      } else if (layout == ZTensorEncodingAttr::DataLayout::HWCK) {
        // HWCK (e4, e3, e2, e1) -> KHWC (ceil(e1/64), e4,, e3, ceil(e2/32),
        // 32, 64)
        e4 = 0;
        e3 = 1;
        e2 = 2;
        e1 = 3;
        n = b.getAffineDimExpr(e1).floorDiv(constE1Block);
        h = b.getAffineDimExpr(e4);
        w = b.getAffineDimExpr(e3);
        c = b.getAffineDimExpr(e2).floorDiv(constE2Block);
        res32 = b.getAffineDimExpr(e2) % constE2Block;
        res64 = b.getAffineDimExpr(e1) % constE1Block;
      } else if (layout == ZTensorEncodingAttr::DataLayout::FICO) {
        // (e4, e3, e2, e1) -> (e4, 4*ceil(e1/4/64), e3, ceil(e2/32), 32, 64)
        assert(!ShapedType::isDynamic(shape[rank - 1]) &&
               (shape[rank - 1] % 4) == 0 &&
               "wrong concatenated dimension size");
        int64_t s = shape[rank - 1] / 4;
        // ((s + 64 - 1) / 64) * 64;
        int64_t s_pad = ceil(static_cast<double>(s) / 64) * 64;
        int64_t pad_size = s_pad - s;
        AffineExpr constExprS = getAffineConstantExpr(s, b.getContext());
        if (rank == 2) {
          e2 = 0;
          e1 = 1;
          w = constExpr0;
        } else if (rank == 3) {
          e3 = 0;
          e2 = 1;
          e1 = 2;
          w = b.getAffineDimExpr(e3);
        } else {
          llvm_unreachable("Unsupported rank in ZDNN_FICO layout");
        }
        n = constExpr0;
        // shape[0] is the direction dimension for LSTM, and should be 1 or 2
        assert((shape[0] == 1 || shape[0] == 2) &&
               "wrong direction dimension size");
        h = (((rank == 2) ? shape[0] : 1) *
             (b.getAffineDimExpr(e1) +
                 pad_size * (b.getAffineDimExpr(e1).floorDiv(constExprS))))
                .floorDiv(constE1Block);
        c = b.getAffineDimExpr(e2).floorDiv(constE2Block);
        res32 = b.getAffineDimExpr(e2) % constE2Block;
        res64 = (b.getAffineDimExpr(e1) +
                    pad_size * (b.getAffineDimExpr(e1).floorDiv(constExprS))) %
                constE1Block;
      } else if (layout == ZTensorEncodingAttr::DataLayout::ZRH) {
        // (e4, e3, e2, e1) -> (e4, 3*ceil(e1/4/64), e3, ceil(e2/32), 32, 64)
        int64_t hidden_size = shape[rank - 1];
        assert(hidden_size > 0 &&
               "Dynamic dimension in hidden_size not supported "
               "in affine_map generation.");
        assert((hidden_size % 3) == 0 && "wrong concatenated dimension size.");
        int64_t s = hidden_size / 3;
        int64_t s_pad =
            ceil(static_cast<float>(s) / 64) * 64; // ((s + 64 - 1) / 64) * 64;
        int64_t pad_size = s_pad - s;
        AffineExpr constExprS = getAffineConstantExpr(s, b.getContext());
        if (rank == 2) {
          e2 = 0;
          e1 = 1;
          w = constExpr0;
        } else if (rank == 3) {
          e3 = 0;
          e2 = 1;
          e1 = 2;
          w = b.getAffineDimExpr(e3);
        } else {
          llvm_unreachable("Unsupported rank in ZDNN_ZRH layout");
        }
        n = constExpr0;
        // shape[0] is the direction dimension for GRU, and should be 1 or 2
        assert((shape[0] == 1 || shape[0] == 2) &&
               "wrong direction dimension size");
        h = (((rank == 2) ? shape[0] : 1) *
             (b.getAffineDimExpr(e1) +
                 pad_size * (b.getAffineDimExpr(e1).floorDiv(constExprS))))
                .floorDiv(constE1Block);
        c = b.getAffineDimExpr(e2).floorDiv(constE2Block);
        res32 = b.getAffineDimExpr(e2) % constE2Block;
        res64 = (b.getAffineDimExpr(e1) +
                    pad_size * (b.getAffineDimExpr(e1).floorDiv(constExprS))) %
                constE1Block;
      } else if (layout == ZTensorEncodingAttr::DataLayout::BFICO) {
        llvm_unreachable("Unsupported layout yet");
      } else if (layout == ZTensorEncodingAttr::DataLayout::BZRH) {
        llvm_unreachable("Unsupported layout yet");
      } else
        llvm_unreachable("Unsupported layout");

      dimExpr.emplace_back(n);
      dimExpr.emplace_back(h);
      dimExpr.emplace_back(w);
      dimExpr.emplace_back(c);
      dimExpr.emplace_back(res32);
      dimExpr.emplace_back(res64);
      AffineMap smap = AffineMap::get(rank, 0, dimExpr, b.getContext());
      // Output type is F16 for zAIU.
      MemRefType outType = MemRefType::get(shape, elementType);
      resZMemRefType.value =
          MemRefType::Builder(outType).setLayout(AffineMapAttr::get(smap));
      resZMemRefType.layout = convertZTensorDataLayoutToStringAttr(b, layout);
    } else { // Does not have tensorType.getEncoding().
      resZMemRefType.value = MemRefType::get(shape, elementType);
    }
  } else { // Not mlir::isa<TensorType>(type).
    resZMemRefType.value = mlir::dyn_cast<MemRefType>(type);
  }
  return resZMemRefType;
}

//===----------------------------------------------------------------------===//
// Lower ZHigh Stick to ZLow Stick
//===----------------------------------------------------------------------===//

// Support for flatten ztensor

struct ZHighToZLowStickOpLowering : public ConversionPattern {
  ZHighToZLowStickOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, ZHighStickOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    ZHighStickOp stickOp = mlir::cast<ZHighStickOp>(op);

    ZHighStickOpAdaptor operandAdaptor(operands);
    Value input = operandAdaptor.getIn();
    StringAttr layout = stickOp.getLayoutAttr();
    IntegerAttr saturation = stickOp.getSaturationAttr();

    MultiDialectBuilder<OnnxBuilder, IndexExprBuilderForKrnl> create(
        rewriter, loc);
    ZHighStickOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Convert ZTensor type to MemRefType.
    ZMemRefType zMemRefType =
        convertZTensorToMemRefType(*op->result_type_begin());

    // Allocate a buffer for the result MemRef.
    Value alloc = insertAllocForZMemRef(
        zMemRefType, shapeHelper.getOutputDims(), op, rewriter);
    if (isNHWCLayout(layout)) {
      if (!nnpaDisableCompilerStickUnstick) {
        // Compiler-generated stick hasn't supported NCHW yet.
        // Explicitly transpose NCHW to NHWC.
        input = create.onnx.toMemref(
            create.onnx.transposeInt64(input, ArrayRef<int64_t>({0, 2, 3, 1})));
      } else
        // Otherwise, we can directly stickify from NCHW.
        // Set pre-transformed layout to NCHW.
        layout = getNCHWLayoutAttr(rewriter);
    }

    // Else, emit a ZLow operation.
    rewriter.create<ZLowStickOp>(loc, input, alloc, layout, saturation);
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Lower ZHigh StickForLSTM to ZLow StickForLSTM
//===----------------------------------------------------------------------===//

struct ZHighToZLowStickForLSTMOpLowering : public ConversionPattern {
  ZHighToZLowStickForLSTMOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, ZHighStickForLSTMOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    ZHighStickForLSTMOpAdaptor operandAdaptor(operands);

    IndexExprBuilderForKrnl createKrnlIE(rewriter, loc);
    ZHighStickForLSTMOpShapeHelper shapeHelper(op, operands, &createKrnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Convert ZTensor type to MemRefType.
    ZMemRefType zMemRefType =
        convertZTensorToMemRefType(*op->result_type_begin());

    // Allocate a buffer for the result MemRef.
    Value alloc = insertAllocForZMemRef(
        zMemRefType, shapeHelper.getOutputDims(), op, rewriter);

    // Emit a ZLow operation.
    rewriter.create<ZLowStickForLSTMOp>(loc, operandAdaptor.getFGate(),
        operandAdaptor.getIGate(), operandAdaptor.getCGate(),
        operandAdaptor.getOGate(), alloc);

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Lower ZHigh StickForGRU to ZLow StickForGRU
//===----------------------------------------------------------------------===//

struct ZHighToZLowStickForGRUOpLowering : public ConversionPattern {
  ZHighToZLowStickForGRUOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, ZHighStickForGRUOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    ZHighStickForGRUOpAdaptor operandAdaptor(operands);

    IndexExprBuilderForKrnl createKrnlIE(rewriter, loc);
    ZHighStickForGRUOpShapeHelper shapeHelper(op, operands, &createKrnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Convert ZTensor type to MemRefType.
    ZMemRefType zMemRefType =
        convertZTensorToMemRefType(*op->result_type_begin());

    // Allocate a buffer for the result MemRef.
    Value alloc = insertAllocForZMemRef(
        zMemRefType, shapeHelper.getOutputDims(), op, rewriter);

    // Emit a ZLow operation.
    rewriter.create<ZLowStickForGRUOp>(loc, operandAdaptor.getZGate(),
        operandAdaptor.getRGate(), operandAdaptor.getHGate(), alloc);

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Lower ZHigh QuantizedStick to ZLow QuantizedStick
//===----------------------------------------------------------------------===//

struct ZHighToZLowQuantizedStickOpLowering : public ConversionPattern {
  ZHighToZLowQuantizedStickOpLowering(TypeConverter &typeConverter,
      MLIRContext *ctx, bool enableSIMD, bool enableParallel)
      : ConversionPattern(
            typeConverter, ZHighQuantizedStickOp::getOperationName(), 1, ctx) {
    this->enableSIMD = enableSIMD;
    this->enableParallel = enableParallel;
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    auto qstickOp = cast<ZHighQuantizedStickOp>(op);

    ZHighQuantizedStickOpAdaptor operandAdaptor(operands);
    Value X = operandAdaptor.getIn();
    Value XRecScale = operandAdaptor.getInRecScale();
    Value XOffset = operandAdaptor.getInOffset();
    StringAttr layout = qstickOp.getLayoutAttr();
    StringAttr quantizedType = qstickOp.getQuantizedTypeAttr();
    bool symmetricMode = qstickOp.getSymMode() != 0;

    MultiDialectBuilder<IndexExprBuilderForKrnl, KrnlBuilder, MathBuilder,
        MemRefBuilder, OnnxBuilder, ZLowBuilder>
        create(rewriter, loc);
    ZHighQuantizedStickOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Convert ZTensor type to MemRefType.
    ZMemRefType zMemRefType =
        convertZTensorToMemRefType(*op->result_type_begin());

    Type si64Ty = rewriter.getIntegerType(64, true);
    Type i8Ty = rewriter.getIntegerType(8);
    Type f32Ty = rewriter.getF32Type();
    MemRefType scalarF32MemRefTy = MemRefType::get({}, f32Ty);
    MemRefType scalarI8MemRefTy = MemRefType::get({}, i8Ty);

    // Attributes.
    IntegerAttr trueAttr = rewriter.getIntegerAttr(si64Ty, -1);

    // Compute rec_scale and offset.
    Value recScale = nullptr;
    Value offset = nullptr;
    if (!isNoneValue(XRecScale))
      recScale = create.krnl.load(XRecScale);
    if (!isNoneValue(XOffset))
      offset = create.krnl.load(XOffset);

    // Find out more about the original input tensor.
    Type inputOriginalType = op->getOperand(0).getType();
    StringAttr xLayout = getZTensorLayoutAttr(rewriter, inputOriginalType);
    bool xIsZTensorOfDLF16 = (xLayout != nullptr);

    if (!recScale && !offset) {
      if (symmetricMode) {
        if (xIsZTensorOfDLF16) {
          llvm_unreachable("Does not support symmetric quantization for a "
                           "ztensor at this moment");
        }
        offset = create.math.constant(f32Ty, 0.0);
        emitSymmetricQuantRecscaleToScalar(
            rewriter, loc, op, X, 8, recScale, enableSIMD, enableParallel);
      } else {
        // Get layout of the defining operation of X. Do not checking that we
        // have a supported z tensor, as this checking will be performed in
        // emitDynamicQuantizationLinearMinMaxFromStickifiedInput, when called.
        // Compute min/max.
        Value inputMin, inputMax;
        if (xIsZTensorOfDLF16) {
          // Call will test that we can handle the specific xLayout.
          emitDynamicQuantizationLinearMinMaxFromStickifiedInput(rewriter, loc,
              op, X, xLayout, inputMin, inputMax, enableSIMD, enableParallel);
        } else {
          // Proceed with computing min/max using normal tensor of normal types.
          assert(xLayout == nullptr && "expected no layout");
          emitDynamicQuantizationLinearMinMax(rewriter, loc, op, X, inputMin,
              inputMax, enableSIMD, enableParallel);
        }
        // Compute scale & zero point. NNPA uses signed i8 so QMax is 127 and
        // QMin is -128.
        Value scale, quantizedOffset;
        Value qMax = create.math.constant(f32Ty, 127.0);
        Value qMin = create.math.constant(f32Ty, -128.0);
        emitDynamicQuantizationLinearScalarParametersFromMinMax(rewriter, loc,
            op, scalarF32MemRefTy, scalarI8MemRefTy, inputMin, inputMax, qMin,
            qMax, scale, offset, quantizedOffset, /*want zero point*/ true,
            enableParallel);
        // Compute recScale.
        Value one = create.math.constant(f32Ty, 1.0);
        recScale = create.math.div(one, scale);
      }
    }

    // MemRefs for recScale and offset.
    Value memrefRecScale = create.mem.alignedAlloc(scalarF32MemRefTy);
    create.krnl.store(recScale, memrefRecScale);
    Value memrefOffset = create.mem.alignedAlloc(scalarF32MemRefTy);
    create.krnl.store(offset, memrefOffset);

    if (xIsZTensorOfDLF16) {
      // Already stickified.
      rewriter.replaceOp(op, {X, memrefRecScale, memrefOffset});
      return success();
    }

    // Allocate a buffer for the result MemRef.
    Value alloc = insertAllocForZMemRef(
        zMemRefType, shapeHelper.getOutputDims(), op, rewriter);
    // Emit a ZLow operation.
    if (quantizedType.getValue().equals_insensitive(QTYPE_DLFLOAT16)) {
      // Use normal stickification for dlfloat16 type so that we can flexibly
      // switch between compiler-generated and zdnn stick.
      create.zlow.stick(X, alloc, layout, trueAttr);
    } else {
      create.zlow.quantizedStick(
          X, memrefRecScale, memrefOffset, alloc, layout, quantizedType);
    }
    rewriter.replaceOp(op, {alloc, memrefRecScale, memrefOffset});
    return success();
  }

private:
  bool enableSIMD = false;
  bool enableParallel = false;
};

//===----------------------------------------------------------------------===//
// Lower ZHigh Unstick to ZLow Unstick
//===----------------------------------------------------------------------===//

struct ZHighToZLowUnstickOpLowering : public ConversionPattern {
  ZHighToZLowUnstickOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, ZHighUnstickOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();

    ZHighUnstickOpAdaptor operandAdaptor(operands);
    Value input = operandAdaptor.getIn();

    // Get layout attribute. Do not get it from the input in OpAdaptor since
    // that input is the converted type, i.e. MemRefType. Get directly from
    // Operation instead where the type is TensorType that has the layout
    // encoding attribute.
    StringAttr layout =
        getZTensorLayoutAttr(rewriter, op->getOperand(0).getType());

    MultiDialectBuilder<OnnxBuilder, IndexExprBuilderForKrnl> create(
        rewriter, loc);
    ZHighUnstickOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Convert ZTensor type to MemRefType.
    ZMemRefType zMemRefType =
        convertZTensorToMemRefType(*op->result_type_begin());

    // Allocate a buffer for the result MemRef.
    Value alloc = nullptr;
    if (isNHWCLayout(layout)) {
      if (!nnpaDisableCompilerStickUnstick) {
        // Compiler-generated unstick hasn't supported NCHW yet.
        // This code allocates a NHWC buffer. It gets dims from the NCHW input.
        SmallVector<IndexExpr> dimList;
        dimList.emplace_back(shapeHelper.getOutputDims()[0]);
        dimList.emplace_back(shapeHelper.getOutputDims()[2]);
        dimList.emplace_back(shapeHelper.getOutputDims()[3]);
        dimList.emplace_back(shapeHelper.getOutputDims()[1]);
        MultiDialectBuilder<MemRefBuilder> create(rewriter, loc);
        MemRefType resType = zMemRefType.value;
        ArrayRef<int64_t> shape = resType.getShape();
        alloc = create.mem.alignedAlloc(
            MemRefType::get({shape[0], shape[2], shape[3], shape[1]},
                resType.getElementType()),
            dimList);
      } else {
        // Otherwise, we can directly stickify from NCHW.
        // Set pre-transformed layout to NCHW.
        layout = getNCHWLayoutAttr(rewriter);
      }
    }
    if (alloc == nullptr)
      alloc = insertAllocForZMemRef(
          zMemRefType, shapeHelper.getOutputDims(), op, rewriter);

    // Emit a ZLow operation.
    rewriter.create<ZLowUnstickOp>(loc, input, alloc, layout);
    if (isNHWCLayout(layout) && !nnpaDisableCompilerStickUnstick)
      // Compiler-generated unstick hasn't supported NCHW yet.
      // Explicitly transpose NHWC to NCHW.
      alloc =
          create.onnx.transposeInt64(alloc, ArrayRef<int64_t>({0, 3, 1, 2}));
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Lower ZHigh Stickified Constant to KrnlGlobal
//===----------------------------------------------------------------------===//

struct ZHighToZLowStickifiedConstantOpLowering : public ConversionPattern {
  static int constantID;
  ZHighToZLowStickifiedConstantOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(typeConverter,
            ZHighStickifiedConstantOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    ZHighStickifiedConstantOp stickifiedConstOp =
        llvm::dyn_cast<ZHighStickifiedConstantOp>(op);

    // Convert ZTensor type to MemRefType.
    ZMemRefType zMemRefType =
        convertZTensorToMemRefType(*op->result_type_begin());

    // Normalize MemRefType to get a static shape.
    assert(mlir::cast<MemRefType>(zMemRefType.value).getNumDynamicDims() == 0 &&
           "MemRefType has dynamic dimensions");
    MemRefType normalizedType =
        affine::normalizeMemRefType(mlir::cast<MemRefType>(zMemRefType.value));
    ArrayRef<int64_t> normalizedShape = normalizedType.getShape();

    // Validate the stickified tensor.
    Attribute valueAttr = stickifiedConstOp.getValueAttr();
    int64_t sizeInBytes = getMemRefEltSizeInBytes(normalizedType);
    sizeInBytes *= normalizedType.getNumElements();
    if (auto denseAttr = mlir::dyn_cast_or_null<DenseElementsAttr>(valueAttr)) {
      ArrayRef<char> data = denseAttr.getRawData();
      if (denseAttr.isSplat()) {
        // Constant ztensor's buffer is tensor<sizeInBytes x i8>.
        int8_t v = denseAttr.getSplatValue<int8_t>();
        // NNPA does not work with a splat buffer.
        // Expand the memory buffer for NNPA by using DenseResourceElementsAttr.
        valueAttr = getDenseResourceElementsAttrOfValue(
            rewriter, stickifiedConstOp, v, sizeInBytes);
      } else {
        assert(
            (data.size() == static_cast<uint64_t>(sizeInBytes)) &&
            "The stickified tensor's buffer size and MemRef's size mismatched");
      }
    } else if (auto resourceAttr =
                   mlir::dyn_cast_or_null<DenseResourceElementsAttr>(
                       valueAttr)) {
      auto blob = resourceAttr.getRawHandle().getBlob();
      assert(blob && "Expecting dense resource with a valid blob");
      ArrayRef<char> data = blob->getData();
      assert(
          (data.size() == static_cast<uint64_t>(sizeInBytes)) &&
          "The stickified tensor's buffer size and MemRef's size mismatched");
    } else {
      llvm_unreachable("Unsupported ElementsAttr");
    }

    // Create a KrnlGlobalOp.
    KrnlGlobalOp constantGlobal =
        rewriter.create<KrnlGlobalOp>(loc, zMemRefType.value,
            /*shape=*/
            rewriter.getI64ArrayAttr(normalizedShape),
            /*name=*/
            rewriter.getStringAttr(
                "constant_stickify_" + std::to_string(constantID)),
            /*value=*/valueAttr,
            /*offset=*/nullptr,
            /*alignment=*/stickifiedConstOp.getAlignmentAttr());

    // Increment constant ID:
    constantID++;

    rewriter.replaceOp(op, constantGlobal.getResult());
    return success();
  }
};

int ZHighToZLowStickifiedConstantOpLowering::constantID = 0;

template <typename OP_TYPE>
struct ZLowOpFor {
  using Op = void;
};

//===----------------------------------------------------------------------===//
// Lower ZHigh binary ops to ZLow.
//===----------------------------------------------------------------------===//

template <>
struct ZLowOpFor<ZHighAddOp> {
  using Op = ZLowAddOp;
};

template <>
struct ZLowOpFor<ZHighSubOp> {
  using Op = ZLowSubOp;
};

template <>
struct ZLowOpFor<ZHighMulOp> {
  using Op = ZLowMulOp;
};

template <>
struct ZLowOpFor<ZHighDivOp> {
  using Op = ZLowDivOp;
};

template <>
struct ZLowOpFor<ZHighMinOp> {
  using Op = ZLowMinOp;
};

template <>
struct ZLowOpFor<ZHighMaxOp> {
  using Op = ZLowMaxOp;
};

template <typename OP_TYPE>
struct ZHighToZLowBinaryOpLowering : public ConversionPattern {
  ZHighToZLowBinaryOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(typeConverter, OP_TYPE::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    Value inputA = operands[0];
    Value inputB = operands[1];

    // Helper builders.
    MultiDialectBuilder<IndexExprBuilderForKrnl> create(rewriter, loc);

    // Convert ZTensor type to MemRefType.
    ZMemRefType zMemRefType =
        convertZTensorToMemRefType(*op->result_type_begin());

    // Shape helper.
    ZHighBinaryOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Allocate a buffer for the result MemRef.
    Value alloc = insertAllocForZMemRef(
        zMemRefType, shapeHelper.getOutputDims(), op, rewriter);

    // Get the original shape before it is vanished by lower passes.
    SmallVector<IndexExpr, 4> dims;
    create.krnlIE.getShapeAsDims(inputA, dims);
    Value shape = insertShapeMemRefI64(rewriter, loc, dims);

    rewriter.create<typename ZLowOpFor<OP_TYPE>::Op>(
        loc, inputA, inputB, shape, alloc, zMemRefType.layout);
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Lower ZHigh unary ops to ZLow.
//===----------------------------------------------------------------------===//

template <>
struct ZLowOpFor<ZHighLogOp> {
  using Op = ZLowLogOp;
};

template <>
struct ZLowOpFor<ZHighExpOp> {
  using Op = ZLowExpOp;
};

template <>
struct ZLowOpFor<ZHighSqrtOp> {
  using Op = ZLowSqrtOp;
};

template <>
struct ZLowOpFor<ZHighInvSqrtOp> {
  using Op = ZLowInvSqrtOp;
};

template <>
struct ZLowOpFor<ZHighReluOp> {
  using Op = ZLowReluOp;
};

template <>
struct ZLowOpFor<ZHighGeluOp> {
  using Op = ZLowGeluOp;
};

template <>
struct ZLowOpFor<ZHighTanhOp> {
  using Op = ZLowTanhOp;
};

template <>
struct ZLowOpFor<ZHighSigmoidOp> {
  using Op = ZLowSigmoidOp;
};

template <typename OP_TYPE>
struct ZHighToZLowUnaryOpLowering : public ConversionPattern {
  ZHighToZLowUnaryOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(OP_TYPE::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    Value input = operands[0];

    // Helper builders.
    MultiDialectBuilder<IndexExprBuilderForKrnl> create(rewriter, loc);

    // Convert ZTensor type to MemRefType.
    ZMemRefType zMemRefType =
        convertZTensorToMemRefType(*op->result_type_begin());

    // Shape helper.
    ZHighUnaryOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();
    SmallVector<IndexExpr, 4> &dims = shapeHelper.getOutputDims();

    // Allocate a buffer for the result MemRef.
    Value alloc = insertAllocForZMemRef(zMemRefType, dims, op, rewriter);

    // Get the original shape before it is vanished by lower passes.
    Value shape = insertShapeMemRefI64(rewriter, loc, dims);

    // Emit a ZLow operation.
    rewriter.create<typename ZLowOpFor<OP_TYPE>::Op>(
        loc, input, shape, alloc, zMemRefType.layout);
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

// Reshape operation. Code similar to unary lowering, except that we use the
// operation's specialized shape here.
struct ZHighToZLowReshapeOpLowering : public ConversionPattern {
  ZHighToZLowReshapeOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(ZHighReshapeOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    Value input = operands[0];

    // Helper builders.
    MultiDialectBuilder<IndexExprBuilderForKrnl> create(rewriter, loc);

    // Convert ZTensor type to MemRefType.
    ZMemRefType zMemRefType =
        convertZTensorToMemRefType(*op->result_type_begin());

    // Shape helper.
    ZHighReshapeOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();
    SmallVector<IndexExpr, 4> &dims = shapeHelper.getOutputDims();

    // Allocate a buffer for the result MemRef. Follow this pattern to be
    // similar to all the other zlow patterns. Will remove the alloc when
    // lowering zlow.reshape to memref.reinterpret_cast once memrefs are
    // normalized. See code in ReshapeToReinterpretCastPattern.
    Value alloc = insertAllocForZMemRef(zMemRefType, dims, op, rewriter);

    // Note, we do not need to save the shape of the original operation, as this
    // reshape is "no-op" that logically reorganize the shape of the operation
    // into 2 equivalent shapes under their given layout.

    // Emit a ZLow operation.
    rewriter.create<ZLowReshapeOp>(
        loc, input, /* shape,*/ alloc, zMemRefType.layout);
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Lower ZHigh ReduceMax/ReduceMin to ZLow ReduceMax/ReduceMin
//===----------------------------------------------------------------------===//
template <typename OP_TYPE>
struct ZLowReduceOpFor {
  using Op = void;
};

template <>
struct ZLowReduceOpFor<ZHighReduceMaxOp> {
  using Op = ZLowReduceMaxOp;
};

template <>
struct ZLowReduceOpFor<ZHighReduceMinOp> {
  using Op = ZLowReduceMinOp;
};

template <typename OP_TYPE>
struct ZHighToZLowReduceOpLowering : public ConversionPattern {
  ZHighToZLowReduceOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(OP_TYPE::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    Value data = operands[0];

    // Helper builders.
    MultiDialectBuilder<IndexExprBuilderForKrnl, KrnlBuilder, MemRefBuilder>
        create(rewriter, loc);

    // Convert ZTensor type to MemRefType.
    ZMemRefType zMemRefType =
        convertZTensorToMemRefType(*op->result_type_begin());

    // Shape helper.
    ZHighReductionOpShapeHelper<OP_TYPE> shapeHelper(
        op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Allocate a buffer for the result MemRef.
    Value alloc = insertAllocForZMemRef(
        zMemRefType, shapeHelper.getOutputDims(), op, rewriter);

    // Get the original shape before it is vanished by lower passes.
    DimsExpr dataDims;
    create.krnlIE.getShapeAsDims(data, dataDims);
    Value shape = insertShapeMemRefI64(rewriter, loc, dataDims);

    // Emit 'alloc' for work_area that is of 4K-aligned 8K bytes.
    Value workArea = create.mem.alignedAlloc(
        MemRefType::get({8 * 1024}, rewriter.getIntegerType(8)), gAlignment);

    // Emit a ZLow operation.
    rewriter.create<typename ZLowReduceOpFor<OP_TYPE>::Op>(
        loc, data, workArea, shape, alloc, zMemRefType.layout);
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Lower ZHigh Softmax to ZLow Softmax
//===----------------------------------------------------------------------===//
struct ZHighToZLowSoftmaxOpLowering : public ConversionPattern {
  ZHighToZLowSoftmaxOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, ZHighSoftmaxOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    ZHighSoftmaxOp softmaxOp = llvm::dyn_cast<ZHighSoftmaxOp>(op);
    ZHighSoftmaxOpAdaptor operandAdaptor(operands);
    Value input = operands[0];

    // Helper builders.
    MultiDialectBuilder<IndexExprBuilderForKrnl, MemRefBuilder> create(
        rewriter, loc);

    // Convert ZTensor type to MemRefType.
    ZMemRefType zMemRefType =
        convertZTensorToMemRefType(*op->result_type_begin());

    // Shape helper.
    ONNXUnaryOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();
    SmallVector<IndexExpr, 4> &dims = shapeHelper.getOutputDims();

    // Allocate a buffer for the result MemRef.
    Value alloc = insertAllocForZMemRef(zMemRefType, dims, op, rewriter);

    // Get the original shape before it is vanished by lower passes.
    Value shape = insertShapeMemRefI64(rewriter, loc, dims);

    // Emit 'alloc' for work_area that is of 4K-aligned 8K bytes.
    Value workArea = create.mem.alignedAlloc(
        MemRefType::get({8 * 1024}, rewriter.getIntegerType(8)), gAlignment);

    // Emit ZLow.softmax.
    rewriter.create<ZLowSoftmaxOp>(
        loc, input, workArea, shape, alloc, softmaxOp.getActFuncAttr());
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Lower ZHigh MeanReduce2D to ZLow MeanReduce2D
//===----------------------------------------------------------------------===//
struct ZHighToZLowMeanReduce2DOpLowering : public ConversionPattern {
  ZHighToZLowMeanReduce2DOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, ZHighMeanReduce2DOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    ZHighMeanReduce2DOpAdaptor operandAdaptor(operands);
    Value input = operandAdaptor.getInput();

    // Helper builders.
    MultiDialectBuilder<IndexExprBuilderForKrnl> create(rewriter, loc);

    // Convert ZTensor type to MemRefType.
    ZMemRefType zMemRefType =
        convertZTensorToMemRefType(*op->result_type_begin());

    // Compute shape.
    ZHighMeanReduce2DOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Get the original shape before it is vanished by lower passes.
    SmallVector<IndexExpr, 4> dims;
    create.krnlIE.getShapeAsDims(input, dims);
    Value shape = insertShapeMemRefI64(rewriter, loc, dims);

    // Allocate a buffer for the result MemRef.
    Value alloc = insertAllocForZMemRef(
        zMemRefType, shapeHelper.getOutputDims(), op, rewriter);

    rewriter.create<ZLowMeanReduce2DOp>(loc, input, shape, alloc);
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Lower ZHigh Pool2D to ZLow Pool2D
//===----------------------------------------------------------------------===//
template <typename ZHIGH_POOLOP, typename ZLOW_POOLOP>
struct ZHighToZLowPool2DOpLowering : public ConversionPattern {
  ZHighToZLowPool2DOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, ZHIGH_POOLOP::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    auto pool2dOp = llvm::dyn_cast<ZHIGH_POOLOP>(op);
    typename ZHIGH_POOLOP::Adaptor operandAdaptor(operands);

    // Helper builders.
    MultiDialectBuilder<IndexExprBuilderForKrnl> create(rewriter, loc);

    // Compute shape.
    ZHighPoolingOpShapeHelper<ZHIGH_POOLOP> shapeHelper(
        op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Convert type.
    ZMemRefType zMemRefType =
        convertZTensorToMemRefType(pool2dOp.getResult().getType());

    // Allocate result buffers.
    Value alloc = insertAllocForZMemRef(
        zMemRefType, shapeHelper.getOutputDims(), op, rewriter);

    // Create a buffer to store the original shape information.
    Value shapeMemRef =
        insertShapeMemRefI64(rewriter, loc, shapeHelper.allOriginalDims);

    // Create a zLow op.
    rewriter.create<ZLOW_POOLOP>(loc, operandAdaptor.getInput(), shapeMemRef,
        alloc, pool2dOp.getKernelShapeAttr(), pool2dOp.getStridesAttr(),
        pool2dOp.getPaddingTypeAttr());
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Lower ZHigh LeakyRelu to ZLow LeakyRelu
//===----------------------------------------------------------------------===//

struct ZHighToZLowLeakyReluOpLowering : public ConversionPattern {
  ZHighToZLowLeakyReluOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, ZHighLeakyReluOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    ZHighLeakyReluOp leakyreluOp = llvm::dyn_cast<ZHighLeakyReluOp>(op);
    ZHighLeakyReluOpAdaptor operandAdaptor(operands);

    // Helper builders.
    MultiDialectBuilder<IndexExprBuilderForKrnl> create(rewriter, loc);

    // Convert ZTensor type to MemRefType.
    ZMemRefType zMemRefType =
        convertZTensorToMemRefType(*op->result_type_begin());

    // Shape helper.
    ZHighUnaryOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();
    SmallVector<IndexExpr, 4> &dims = shapeHelper.getOutputDims();

    // Allocate a buffer for the result MemRef.
    Value alloc = insertAllocForZMemRef(zMemRefType, dims, op, rewriter);

    // Get the original shape before it is vanished by lower passes.
    Value shape = insertShapeMemRefI64(rewriter, loc, dims);

    // Attributes.
    FloatAttr alphaVal = leakyreluOp.getAlphaAttr();

    // Emit zlow.leakyrelu.
    rewriter.create<ZLowLeakyReluOp>(
        loc, operandAdaptor.getX(), shape, alloc, alphaVal, zMemRefType.layout);
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Lower ZHigh MatMul to ZLow MatMul
//===----------------------------------------------------------------------===//

struct ZHighToZLowMatMulOpLowering : public ConversionPattern {
  ZHighToZLowMatMulOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, ZHighMatMulOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    ZHighMatMulOp matmulOp = llvm::dyn_cast<ZHighMatMulOp>(op);
    ZHighMatMulOpAdaptor operandAdaptor(operands);

    // Helper builders.
    MultiDialectBuilder<IndexExprBuilderForKrnl> create(rewriter, loc);

    // Compute shape.
    ZHighMatMulOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Convert ZTensor type to MemRefType.
    ZMemRefType zMemRefType =
        convertZTensorToMemRefType(*op->result_type_begin());

    Value alloc = insertAllocForZMemRef(
        zMemRefType, shapeHelper.getOutputDims(), op, rewriter);

    // Get the original shape before it is vanished by lower passes.
    // Create a 1D MemRef containing necessary dimensions for constructing
    // original shapes.
    // - In case of unstacked: X(m, n) * Y(n, p) + Bias(p)
    // shape is a 1D MemRef (memref<3xindex>) whose items are:
    //   - 1st item: m
    //   - 2nd item: n
    //   - 3rd item: p
    // - In case of stacked: X(s, m, n) * Y(s, n, p) + Bias(s, p)
    //      or broadcasting1: X(m, n) * Y(s, n, p) + Bias(s, p)
    //      or broadcasting23: X(s, m, n) * Y(n, p) + Bias(p)
    // shape is a 1D MemRef (memref<4xindex>) whose items are:
    //   - 1st item: s
    //   - 2nd item: m
    //   - 3rd item: n
    //   - 4th item: p

    Value shapeMemRef =
        insertShapeMemRefI64(rewriter, loc, shapeHelper.allOriginalDims);

    // Prepare optional bias.
    Value bias = operandAdaptor.getB();
    if (mlir::isa<NoneType>(bias.getType())) {
      SmallVector<IndexExpr, 4> resDims, biasDims;
      create.krnlIE.getShapeAsDims(alloc, resDims);
      ZTensorEncodingAttr::DataLayout biasLayout;
      if (shapeHelper.isStacked || shapeHelper.isBroadcasted1) {
        // Bias type is 2DS.
        biasDims.emplace_back(resDims[0]);
        biasDims.emplace_back(resDims[2]);
        biasLayout = ZTensorEncodingAttr::DataLayout::_2DS;
      } else {
        // Bias type is 1D. Get the last dim size.
        biasDims.emplace_back(resDims[resDims.size() - 1]);
        biasLayout = ZTensorEncodingAttr::DataLayout::_1D;
      }
      // Allocate bias.
      bias = insertAllocOrEmitZeroConstant(
          biasDims, biasLayout, op, rewriter, loc);
    }

    // Attributes.
    int64_t bcast1 = (shapeHelper.isBroadcasted1) ? -1 : 0;
    int64_t bcast23 = (shapeHelper.isBroadcasted23) ? -1 : 0;
    int64_t stacked = (shapeHelper.isStacked) ? -1 : 0;
    int64_t transposeA = (matmulOp.getTransposeA() != 0) ? 1 : 0;
    int64_t transposeB = (matmulOp.getTransposeB() != 0) ? 1 : 0;
    IntegerAttr is_bcast1Attr =
        rewriter.getIntegerAttr(rewriter.getIntegerType(64, true), bcast1);
    IntegerAttr is_bcast23Attr =
        rewriter.getIntegerAttr(rewriter.getIntegerType(64, true), bcast23);
    IntegerAttr is_stackedAttr =
        rewriter.getIntegerAttr(rewriter.getIntegerType(64, true), stacked);
    IntegerAttr transposeAAttr =
        rewriter.getIntegerAttr(rewriter.getIntegerType(64, true), transposeA);
    IntegerAttr transposeBAttr =
        rewriter.getIntegerAttr(rewriter.getIntegerType(64, true), transposeB);

    // Emit zlow.matmul.
    rewriter.create<ZLowMatMulOp>(loc, operandAdaptor.getX(),
        operandAdaptor.getY(), bias, shapeMemRef, alloc, is_bcast1Attr,
        is_bcast23Attr, is_stackedAttr, transposeAAttr, transposeBAttr);
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Lower ZHigh QuantizedMatMul to ZLow QuantizedMatMul
//===----------------------------------------------------------------------===//

struct ZHighToZLowQuantizedMatMulOpLowering : public ConversionPattern {
  ZHighToZLowQuantizedMatMulOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(typeConverter,
            ZHighQuantizedMatMulOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    ZHighQuantizedMatMulOp matmulOp =
        llvm::dyn_cast<ZHighQuantizedMatMulOp>(op);
    ZHighQuantizedMatMulOpAdaptor operandAdaptor(operands);

    // Helper builders.
    MultiDialectBuilder<IndexExprBuilderForKrnl, KrnlBuilder, MathBuilder,
        MemRefBuilder, ZLowBuilder>
        create(rewriter, loc);

    // Compute shape.
    ZHighQuantizedMatMulOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Convert ZTensor type to MemRefType.
    ZMemRefType zMemRefType =
        convertZTensorToMemRefType(*op->result_type_begin());
    Type f32Ty = rewriter.getF32Type();
    MemRefType scalarF32MemRefTy = MemRefType::get({}, f32Ty);

    Value zero = create.math.constant(f32Ty, 0.0);
    Value one = create.math.constant(f32Ty, 1.0);

    Value alloc = insertAllocForZMemRef(
        zMemRefType, shapeHelper.getOutputDims(), op, rewriter);
    Value outRecScale = operandAdaptor.getOutRecScaleIn();
    if (mlir::isa<NoneType>(outRecScale.getType())) {
      outRecScale = create.mem.alignedAlloc(
          MemRefType::get({}, rewriter.getF32Type()), {});
      create.krnl.store(one, outRecScale);
    }
    Value outOffset = operandAdaptor.getOutOffsetIn();
    if (mlir::isa<NoneType>(outOffset.getType())) {
      outOffset = create.mem.alignedAlloc(
          MemRefType::get({}, rewriter.getF32Type()), {});
      create.krnl.store(zero, outOffset);
    }

    // Get the original shape before it is vanished by lower passes.
    // Create a 1D MemRef containing necessary dimensions for constructing
    // original shapes.
    // - In case of unstacked: X(m, n) * Y(n, p) + Bias(p)
    // shape is a 1D MemRef (memref<3xindex>) whose items are:
    //   - 1st item: m
    //   - 2nd item: n
    //   - 3rd item: p
    // - In case of stacked: X(s, m, n) * Y(s, n, p) + Bias(s, p)
    //      or broadcasting: X(s, m, n) * Y(n, p) + Bias(p)
    // shape is a 1D MemRef (memref<4xindex>) whose items are:
    //   - 1st item: s
    //   - 2nd item: m
    //   - 3rd item: n
    //   - 4th item: p
    Value shapeMemRef =
        insertShapeMemRefI64(rewriter, loc, shapeHelper.allOriginalDims);

    // Attributes.
    int64_t bcast = (shapeHelper.isBroadcasted) ? -1 : 0;
    int64_t stacked = (shapeHelper.isStacked) ? -1 : 0;
    IntegerAttr is_bcastAttr =
        rewriter.getIntegerAttr(rewriter.getIntegerType(64, true), bcast);
    IntegerAttr is_stackedAttr =
        rewriter.getIntegerAttr(rewriter.getIntegerType(64, true), stacked);
    // QuantizedType attributes.
    StringAttr xQTypeAttr = convertZTensorQuantizedTypeToStringAttr(
        rewriter, getZTensorQuantizedType(matmulOp.getX().getType()));
    StringAttr yQTypeAttr = convertZTensorQuantizedTypeToStringAttr(
        rewriter, getZTensorQuantizedType(matmulOp.getY().getType()));
    StringAttr outQTypeAttr =
        StringAttr::get(rewriter.getContext(), QTYPE_DLFLOAT16);
    StringAttr bQTypeAttr;

    // Prepare optional bias.
    SmallVector<IndexExpr, 4> resDims;
    create.krnlIE.getShapeAsDims(alloc, resDims);
    Value bias = operandAdaptor.getB();
    Value biasRecScale = operandAdaptor.getBRecScale();
    Value biasOffset = operandAdaptor.getBOffset();
    SmallVector<IndexExpr, 4> bDims;
    if (shapeHelper.isStacked) {
      // Bias type is 2DS.
      bDims.emplace_back(resDims[0]);
      bDims.emplace_back(resDims[2]);
    } else {
      // Bias type is 1D. Get the last dim size.
      bDims.emplace_back(resDims[resDims.size() - 1]);
    }
    ZTensorEncodingAttr::DataLayout bLayout;
    ZTensorEncodingAttr::QuantizedType bQType;
    if (mlir::isa<NoneType>(bias.getType())) {
      if (shapeHelper.isStacked) {
        // Bias type is 2DS.
        bLayout = ZTensorEncodingAttr::DataLayout::_2DS;
      } else {
        // Bias type is 1D. Get the last dim size.
        bLayout = ZTensorEncodingAttr::DataLayout::_1D;
      }
      bool preCompute = matmulOp.getPreComputedBias() != 0;
      // Allocate bias.
      if (preCompute)
        bQType = ZTensorEncodingAttr::QuantizedType::DLFLOAT16;
      else
        bQType = ZTensorEncodingAttr::QuantizedType::INT8;
      bQTypeAttr = convertZTensorQuantizedTypeToStringAttr(rewriter, bQType);
      bias = insertAllocOrEmitZeroConstant(
          bDims, bLayout, op, rewriter, loc, bQType);
    } else {
      Type bTensorType = matmulOp.getB().getType();
      bLayout = getZTensorLayout(bTensorType);
      ZTensorEncodingAttr::QuantizedType qtype =
          getZTensorQuantizedType(bTensorType);
      if (qtype == ZTensorEncodingAttr::QuantizedType::UNDEFINED) {
        // Bias is a non-quantized or normal ztensor. Use DLFLOAT16 type.
        qtype = ZTensorEncodingAttr::QuantizedType::DLFLOAT16;
      }
      bQTypeAttr = convertZTensorQuantizedTypeToStringAttr(rewriter, qtype);
      bQType = convertStringAttrToZTensorQuantizedType(bQTypeAttr);
    }
    if (mlir::isa<NoneType>(biasRecScale.getType())) {
      biasRecScale = create.mem.alignedAlloc(scalarF32MemRefTy);
      create.krnl.store(one, biasRecScale);
    }
    if (mlir::isa<NoneType>(biasOffset.getType())) {
      biasOffset = create.mem.alignedAlloc(scalarF32MemRefTy);
      create.krnl.store(zero, biasOffset);
    }

    // Prepare a buffer for work_area.
    // Work area has the same layout as bias but dlfloat16 type.
    if (bDims.empty())
      create.krnlIE.getShapeAsDims(bias, bDims);
    Value workArea = insertAllocForZMemRefByDim(bDims, bLayout,
        ZTensorEncodingAttr::QuantizedType::DLFLOAT16, op, rewriter);

    // Emit zlow.quantizedMatmul.
    // clang-format off
    create.zlow.quantizedMatMul(
        operandAdaptor.getX(), operandAdaptor.getXRecScale(), operandAdaptor.getXOffset(),
        operandAdaptor.getY(), operandAdaptor.getYRecScale(), operandAdaptor.getYOffset(),
        bias, biasRecScale, biasOffset,
        workArea, shapeMemRef,
        alloc, outRecScale, outOffset,
        xQTypeAttr, yQTypeAttr, bQTypeAttr, outQTypeAttr,
        is_bcastAttr, is_stackedAttr,
        matmulOp.getPreComputedBiasAttr(),
        matmulOp.getDisableClippingAttr(),
        matmulOp.getDequantizeOutputAttr());
    // clang-format on
    rewriter.replaceOp(op, {alloc, outRecScale, outOffset});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Lower ZHigh LSTM to ZLow LSTM
//===----------------------------------------------------------------------===//
struct ZHighToZLowLSTMOpLowering : public ConversionPattern {
  ZHighToZLowLSTMOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, ZHighLSTMOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    ZHighLSTMOp lstmOp = llvm::dyn_cast<ZHighLSTMOp>(op);
    ZHighLSTMOpAdaptor operandAdaptor(operands);

    // Helper builders.
    MultiDialectBuilder<IndexExprBuilderForKrnl> create(rewriter, loc);

    // Compute shape.
    ZHighLSTMOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Convert type.
    ZMemRefType hnZMemRefType =
        convertZTensorToMemRefType(lstmOp.getResults()[0].getType());
    ZMemRefType cfZMemRefType =
        convertZTensorToMemRefType(lstmOp.getResults()[1].getType());

    // Allocate result buffers.
    Value allocHnOutput = insertAllocForZMemRef(
        hnZMemRefType, shapeHelper.getOutputDims(0), op, rewriter);
    Value allocCfOutput = insertAllocForZMemRef(
        cfZMemRefType, shapeHelper.getOutputDims(1), op, rewriter);

    // Get the original shape before it is vanished by lower passes.
    // Create a 1D MemRef containing necessary dimensions for constructing
    // original shapes.
    // shapeMemRef :: memref<5xindex>
    // - 1st item: direction
    // - 2nd item: timestep
    // - 3rd item: batchSize
    // - 4th item: featureSize
    // - 5th item: hiddenSize
    Value shapeMemRef =
        insertShapeMemRefI64(rewriter, loc, shapeHelper.allOriginalDims);

    // Prepare optional values: input_bias, hidden_bias, initial_h, initial_c.
    Value initial_h = operandAdaptor.getH0();
    Value initial_c = operandAdaptor.getC0();
    Value input_bias = operandAdaptor.getInputBias();
    Value hidden_bias = operandAdaptor.getHiddenBias();
    if (mlir::isa<NoneType>(initial_h.getType())) {
      initial_h = insertAllocOrEmitZeroConstant(shapeHelper.hc0Shape,
          ZTensorEncodingAttr::DataLayout::_3DS, op, rewriter, loc);
    }
    if (mlir::isa<NoneType>(initial_c.getType())) {
      initial_c = insertAllocOrEmitZeroConstant(shapeHelper.hc0Shape,
          ZTensorEncodingAttr::DataLayout::_3DS, op, rewriter, loc);
    }
    if (mlir::isa<NoneType>(input_bias.getType())) {
      input_bias = insertAllocOrEmitZeroConstant(shapeHelper.biasShape,
          ZTensorEncodingAttr::DataLayout::FICO, op, rewriter, loc);
    }
    if (mlir::isa<NoneType>(hidden_bias.getType())) {
      hidden_bias = insertAllocOrEmitZeroConstant(shapeHelper.biasShape,
          ZTensorEncodingAttr::DataLayout::FICO, op, rewriter, loc);
    }

    // Prepare work area. Double the area for the bidirectional mode.
    bool isDouble = lstmOp.getDirectionAttr().getValue().equals_insensitive(
        "bidirectional");
    Value workArea = insertAllocForWorkAreaForRNNOps(create.krnlIE, rewriter,
        loc, operandAdaptor.getInput(), operandAdaptor.getHiddenWeights(),
        /*numOfGates=*/4,
        /*isDouble=*/isDouble);

    // Emit zlow.lstm.
    rewriter.create<ZLowLSTMOp>(loc, operandAdaptor.getInput(), initial_h,
        initial_c, operandAdaptor.getInputWeights(), input_bias,
        operandAdaptor.getHiddenWeights(), hidden_bias, workArea, shapeMemRef,
        allocHnOutput, allocCfOutput, lstmOp.getDirectionAttr(),
        lstmOp.getReturnAllStepsAttr(), rewriter.getStringAttr("none"));
    std::vector<Value> outputs = {allocHnOutput, allocCfOutput};
    rewriter.replaceOp(op, outputs);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Lower ZHigh GRU to ZLow GRU
//===----------------------------------------------------------------------===//
struct ZHighToZLowGRUOpLowering : public ConversionPattern {
  ZHighToZLowGRUOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, ZHighGRUOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    ZHighGRUOp gruOp = llvm::dyn_cast<ZHighGRUOp>(op);
    ZHighGRUOpAdaptor operandAdaptor(operands);

    // Helper builders.
    MultiDialectBuilder<IndexExprBuilderForKrnl> create(rewriter, loc);

    // Compute shape.
    ZHighGRUOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Convert type.
    ZMemRefType hnZMemRefType =
        convertZTensorToMemRefType(gruOp.getResult().getType());

    // Allocate result buffers.
    Value allocHnOutput = insertAllocForZMemRef(
        hnZMemRefType, shapeHelper.getOutputDims(), op, rewriter);

    // Get the original shape before it is vanished by lower passes.
    // Create a 1D MemRef containing necessary dimensions for constructing
    // original shapes.
    // shapeMemRef :: memref<5xindex>
    // - 1st item: direction
    // - 2nd item: timestep
    // - 3rd item: batchSize
    // - 4th item: featureSize
    // - 5th item: hiddenSize
    Value shapeMemRef =
        insertShapeMemRefI64(rewriter, loc, shapeHelper.allOriginalDims);

    // Prepare optional values: input_bias, hidden_bias, initial_h.
    Value initial_h = operandAdaptor.getH0();
    Value input_bias = operandAdaptor.getInputBias();
    Value hidden_bias = operandAdaptor.getHiddenBias();
    if (mlir::isa<NoneType>(initial_h.getType())) {
      initial_h = insertAllocOrEmitZeroConstant(shapeHelper.h0Shape,
          ZTensorEncodingAttr::DataLayout::_3DS, op, rewriter, loc);
    }
    if (mlir::isa<NoneType>(input_bias.getType())) {
      input_bias = insertAllocOrEmitZeroConstant(shapeHelper.biasShape,
          ZTensorEncodingAttr::DataLayout::ZRH, op, rewriter, loc);
    }
    if (mlir::isa<NoneType>(hidden_bias.getType())) {
      hidden_bias = insertAllocOrEmitZeroConstant(shapeHelper.biasShape,
          ZTensorEncodingAttr::DataLayout::ZRH, op, rewriter, loc);
    }

    // Prepare work area. Double the area for the bidirectional mode.
    bool isDouble =
        gruOp.getDirectionAttr().getValue().equals_insensitive("bidirectional");
    Value workArea = insertAllocForWorkAreaForRNNOps(create.krnlIE, rewriter,
        loc, operandAdaptor.getInput(), operandAdaptor.getHiddenWeights(),
        /*numOfGates=*/3,
        /*isDouble=*/isDouble);

    // Emit zlow.gru.
    rewriter.create<ZLowGRUOp>(loc, operandAdaptor.getInput(), initial_h,
        operandAdaptor.getInputWeights(), input_bias,
        operandAdaptor.getHiddenWeights(), hidden_bias, workArea, shapeMemRef,
        allocHnOutput, gruOp.getDirectionAttr(), gruOp.getReturnAllStepsAttr(),
        rewriter.getStringAttr("none"));
    rewriter.replaceOp(op, allocHnOutput);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Lower ZHigh FixGRUY to Krnl
//===----------------------------------------------------------------------===//

struct ZHighToZLowFixGRUYOpLowering : public ConversionPattern {
  ZHighToZLowFixGRUYOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, ZHighFixGRUYOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();

    ZHighFixGRUYOpAdaptor operandAdaptor(operands);

    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MathBuilder,
        MemRefBuilder>
        create(rewriter, loc);

    // create alloc
    ZHighFixGRUYOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();
    MemRefType outputMemRefType = mlir::cast<MemRefType>(
        typeConverter->convertType(op->getResults()[0].getType()));

    // Value alloc =
    //     create.mem.alignedAlloc(outputMemRefType,
    //     shapeHelper.getOutputDims(0));

    Value Y = operands[0];
    Value sequenceLens = operands[1];
    Value initialH = operands[2];
    // Let's just reuse the buffer of Y
    Value alloc = Y;

    int64_t yRank = outputMemRefType.getShape().size();

    // The loop nested for padding is as follows:
    // for (bs  from 0 to batch size ) {
    //   sequenceUB = sequence_lens[bs]
    //   for sequenceIV = max(0, sequenceUB), SEQ_LENGTH
    //     for (directionIV for directions)
    //       for (hs for hidden states)
    //         Y[sequenceIV, directionIV, bs, hs] = initValue

    // Create loop for batch
    Value iZero = create.math.constantIndex(0);
    ValueRange batchLoop = create.krnl.defineLoops(1);
    create.krnl.iterate(batchLoop, batchLoop, {iZero}, {create.mem.dim(Y, 2)},
        [&](const KrnlBuilder &createKrnl, ValueRange batchIndices) {
          MathBuilder createMath(createKrnl);
          IndexExprScope ieScope(createKrnl);
          Value bs = batchIndices[0];
          Value sequenceUB = createKrnl.load(sequenceLens, {bs});
          // The lower bound for loop of padding the batch should be
          // max(0, sequenceUB) in integer type for possible negative value
          Value seqLB = createMath.cast(rewriter.getIndexType(),
              createMath.max(
                  createMath.constant(sequenceUB.getType(), 0), sequenceUB));
          SmallVector<Value, 4> yLbs;
          SmallVector<Value, 4> yUbs;
          yLbs.emplace_back(seqLB);
          yLbs.emplace_back(iZero);
          yLbs.emplace_back(iZero);
          yUbs.emplace_back(create.mem.dim(Y, 0));
          yUbs.emplace_back(create.mem.dim(Y, 1));
          yUbs.emplace_back(create.mem.dim(Y, 3));

          KrnlRegionOp regionOp = rewriter.create<KrnlRegionOp>(loc);
          rewriter.setInsertionPointToStart(&regionOp.getBodyRegion().front());
          ValueRange loops = create.krnl.defineLoops(yRank - 1);
          create.krnl.iterate(loops, loops, yLbs, yUbs,
              [&](const KrnlBuilder &createKrnl, ValueRange indices) {
                Value sequenceIV(indices[0]);
                Value directionIV(indices[1]);
                Value hs(indices[2]);
                Value initial;
                if (isNoneValue(initialH)) {
                  initial = createMath.constant(
                      outputMemRefType.getElementType(), 0.);
                } else {
                  initial = createKrnl.load(initialH, {directionIV, bs, hs});
                }
                createKrnl.store(
                    initial, alloc, {sequenceIV, directionIV, bs, hs});
              });
        });

#if 0
    // This implementation is copied from GRU.cpp: calculateState.
    // The code is simple but may be less efficient.
    // I keep the code here for future to test the performance on real model
    SmallVector<Value, 4> yLbs(yRank, iZero);
    SmallVector<Value, 4> yUbs;
    for (unsigned r = 0; r < yRank; ++r) {
      yUbs.emplace_back(create.mem.dim(Y, r));
    }

    ValueRange loops = create.krnl.defineLoops(yRank);
    create.krnl.iterate(loops, loops, yLbs, yUbs,
        [&](const KrnlBuilder  &createKrnl, ValueRange indices) {
          MathBuilder createMath(createKrnl);
          IndexExprScope ieScope(createKrnl);
          Value sequenceIV(indices[0]);
          Value directionIV(indices[1]);
          Value bs(indices[2]), hs(indices[3]);
          Value currentV = createKrnl.load(Y, indices);
          Value sequenceUB = createKrnl.load(sequenceLens, {bs});
          Value initial;
          if (isNoneValue(initialH)) {
            initial = createMath.constant(currentV.getType(), 0.);
          } else {
            initial = createKrnl.load(initialH, {directionIV, bs, hs});
          }
          Value cond = createMath.sge(
              createMath.cast(sequenceUB.getType(), sequenceIV), sequenceUB);
          Value newV = createMath.select(cond, /*padding*/ initial, currentV);
          createKrnl.store(newV, alloc, indices);
        });
#endif

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Lower ZHigh FixGRUYh to Krnl
//===----------------------------------------------------------------------===//

struct ZHighToZLowFixGRUYhOpLowering : public ConversionPattern {
  ZHighToZLowFixGRUYhOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, ZHighFixGRUYhOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();

    ZHighFixGRUYhOpAdaptor operandAdaptor(operands);

    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MathBuilder,
        MemRefBuilder>
        create(rewriter, loc);

    // create alloc
    ZHighFixGRUYhOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();
    MemRefType outputMemRefType = mlir::cast<MemRefType>(
        typeConverter->convertType(op->getResults()[0].getType()));
    Value alloc =
        create.mem.alignedAlloc(outputMemRefType, shapeHelper.getOutputDims(0));

    Value Y = operands[0];
    Value sequenceLens = operands[1];

    // Code copied from GRU.cpp: calculateState
    int64_t htRank = 3;
    Value iZero = create.math.constantIndex(0);
    Value iOne = create.math.constantIndex(1);
    SmallVector<Value, 2> htLbs(htRank, iZero);
    SmallVector<Value, 2> htUbs;
    for (unsigned r = 0; r < htRank; ++r) {
      // skip the first two dim for sequence and batch
      htUbs.emplace_back(create.mem.dim(Y, r + 1));
    }
    Value seqSize = create.mem.dim(Y, 0);
    ValueRange loops = create.krnl.defineLoops(htRank);
    create.krnl.iterate(loops, loops, htLbs, htUbs,
        [&](const KrnlBuilder &createKrnl, ValueRange indices) {
          MathBuilder createMath(createKrnl);
          IndexExprScope ieScope(createKrnl);
          Value bs(indices[1]), hs(indices[2]);
          Value directionIV(indices[0]);
          Value sequenceUB = createKrnl.load(sequenceLens, {bs});
          Value bound = createMath.min(
              createMath.cast(seqSize.getType(), sequenceUB), seqSize);
          Value index = createMath.sub(bound, iOne);
          Value lastHt = createKrnl.load(Y, {index, directionIV, bs, hs});
          createKrnl.store(lastHt, alloc, {directionIV, bs, hs});
        });

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Lower ZHigh Conv2D to ZLow Conv2D
//===----------------------------------------------------------------------===//
struct ZHighToZLowConv2DOpLowering : public ConversionPattern {
  ZHighToZLowConv2DOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, ZHighConv2DOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    ZHighConv2DOp conv2dOp = llvm::dyn_cast<ZHighConv2DOp>(op);
    ZHighConv2DOpAdaptor operandAdaptor(operands);

    // Helper builders.
    MultiDialectBuilder<IndexExprBuilderForKrnl> create(rewriter, loc);

    // Compute shape.
    ZHighConv2DOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Convert type.
    ZMemRefType zMemRefType =
        convertZTensorToMemRefType(conv2dOp.getResult().getType());

    // Allocate result buffers.
    Value alloc = insertAllocForZMemRef(
        zMemRefType, shapeHelper.getOutputDims(), op, rewriter);

    // Create a buffer to store the original shape information.
    Value shapeMemRef =
        insertShapeMemRefI64(rewriter, loc, shapeHelper.allOriginalDims);

    // Prepare optional values: input_bias.
    Value bias = operandAdaptor.getInputBias();
    if (mlir::isa<NoneType>(bias.getType())) {
      // Bias's shape is [Channel_out].
      SmallVector<IndexExpr> dims(1, shapeHelper.allOriginalDims[4]);
      bias = insertAllocOrEmitZeroConstant(
          dims, ZTensorEncodingAttr::DataLayout::_1D, op, rewriter, loc);
    }

    // Create a zLow op.
    rewriter.create<ZLowConv2DOp>(loc, operandAdaptor.getInput(),
        operandAdaptor.getInputKernel(), bias, shapeMemRef, alloc,
        conv2dOp.getKernelShapeAttr(), conv2dOp.getStridesAttr(),
        conv2dOp.getPaddingTypeAttr(), conv2dOp.getActFuncAttr());

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Lower ZHigh BatchNorm to ZLow BatchNorm
//===----------------------------------------------------------------------===//
struct ZHighToZLowBatchNormOpLowering : public ConversionPattern {
  ZHighToZLowBatchNormOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, ZHighBatchNormOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    ZHighBatchNormOpAdaptor operandAdaptor(operands);

    // Helper builders.
    MultiDialectBuilder<IndexExprBuilderForKrnl> create(rewriter, loc);

    // Convert ZTensor type to MemRefType.
    ZMemRefType zMemRefType =
        convertZTensorToMemRefType(*op->result_type_begin());

    // Shape helper.
    ONNXUnaryOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();
    SmallVector<IndexExpr, 4> &dims = shapeHelper.getOutputDims();

    // Allocate a buffer for the result MemRef.
    Value alloc = insertAllocForZMemRef(zMemRefType, dims, op, rewriter);

    // Get the original shape before it is vanished by lower passes.
    Value shape = insertShapeMemRefI64(rewriter, loc, dims);

    rewriter.create<ZLowBatchNormOp>(loc, operandAdaptor.getInput(),
        operandAdaptor.getA(), operandAdaptor.getB(), shape, alloc);
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Lower ZHigh StickifiedConstantOfShape to ZLow
//===----------------------------------------------------------------------===//

struct ZHighToZLowStickifiedConstantOfShapeOpLowering
    : public ConversionPattern {
  ZHighToZLowStickifiedConstantOfShapeOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(typeConverter,
            ZHighStickifiedConstantOfShapeOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    MDBuilder create(rewriter, loc);

    auto stickOp = mlir::cast<ZHighStickifiedConstantOfShapeOp>(op);
    FloatAttr value = stickOp.getValueAttr();
    Type i16Ty = rewriter.getI16Type();
    Type i64Ty = rewriter.getI64Type();
    Type f16Ty = rewriter.getF16Type();

    // Convert the scalar value to dlfloat16.
    // Use uint16_t as container.
    float valueF32 = static_cast<float>(value.getValueAsDouble());
    uint16_t valueDLF16;
    fp32_to_dlf16(&valueF32, &valueDLF16, 1);

    // Convert ZTensor type to MemRefType.
    ZMemRefType zMemRefType =
        convertZTensorToMemRefType(*op->result_type_begin());

    // Compute the output shape.
    ZHighStickifiedConstantOfShapeOpShapeHelper shapeHelper(
        op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Allocate a buffer for the result MemRef.
    Value res = insertAllocForZMemRef(
        zMemRefType, shapeHelper.getOutputDims(), op, rewriter);

    // Create a f16 scalar memref to store dlfloat16.
    // Do not cast value from i16 to f16 but do copy data. Otherwise, the result
    // is wrong.
    // Scalar memref of i16.
    Value valueI16 = create.math.constant(i16Ty, valueDLF16);
    Value memrefI16 = create.mem.alignedAlloc(MemRefType::get({}, i16Ty));
    create.krnl.store(valueI16, memrefI16);
    // Scalar memref of f16.
    Value memrefF16 = create.mem.alignedAlloc(MemRefType::get({}, f16Ty));
    create.krnl.memcpy(memrefF16, memrefI16, create.math.constant(i64Ty, 1));

    // Now, broadcast the scalar value to the output tensor.
    // It's neat if we can do the two following statements.
    // ```C
    // Value valueF16 = create.krnl.load(memrefF16);
    // create.krnl.memset(res, valueF16, /*delayed=*/false);
    // ```
    // But, LLVM will think the load value is real a f16 value, and try to read
    // it to do some optimization, but it fails because s390x does not support
    // f16. Even though LLVM can read the f16 value, the result might be wrong
    // since it is dfloat16.
    //
    // The following manual loop does a trick that puts `create.krnl.load`
    // inside the loop, and LLVM does not seem to read the f16 value.
    uint64_t rank = mlir::cast<MemRefType>(res.getType()).getRank();
    ValueRange loopDef = create.krnl.defineLoops(rank);
    SmallVector<IndexExpr, 4> lbs(rank, LitIE(0));
    SmallVector<IndexExpr, 4> ubs = shapeHelper.getOutputDims();
    create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
        [&](const KrnlBuilder &createKrnl, ValueRange indices) {
          // Keep this load inside the loop to tweak LLVM.
          Value valueF16 = createKrnl.load(memrefF16);
          createKrnl.store(valueF16, res, indices);
        });

    rewriter.replaceOp(op, res);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// A template to lower ZHigh DLF16ToF32 and F32ToDLF16.
//===----------------------------------------------------------------------===//

template <typename CONVERT_OP>
struct ZHighToZLowDataConversionLowering
    : public OpConversionPattern<CONVERT_OP> {
  using OpAdaptor = typename CONVERT_OP::Adaptor;
  bool fromF32 = false;
  bool enableParallel = false;

  ZHighToZLowDataConversionLowering(TypeConverter &typeConverter,
      MLIRContext *ctx, bool fromF32, bool enableParallel)
      : OpConversionPattern<CONVERT_OP>(typeConverter, ctx), fromF32(fromF32) {
    this->enableParallel =
        enableParallel &&
        OnnxToKrnlLoweringConfiguration::enableSpecificParallelOps.isEnabled(
            CONVERT_OP::getOperationName());
  }

  LogicalResult matchAndRewrite(CONVERT_OP convertOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = convertOp.getLoc();
    MDBuilder create(rewriter, loc);

    Operation *op = convertOp.getOperation();
    ValueRange operands = adaptor.getOperands();
    Value X = operands[0];
    int64_t rank = getRank(X.getType());

    // SIMD info.
    // Fixed VL for the conversion instruction: 8 elements per instruction
    // call. Because the VL of the zlow.conversions are not "virtualized" in
    // lengths, we manually unroll the loop containing the SIMD operations.
    // Experiments on a 1024x1024 tensors shows best results with an unrolling
    // of 8 SIMD vectors.
    int64_t archVL = 8; // Vector length as defined by z arch for this type.
    int64_t archVLHalf = archVL / 2;
    int64_t unrollVL = 8;              // Manually unroll the SIMD loop.
    int64_t totVL = unrollVL * archVL; // Total numbers of values unrolled.

    // Convert the output type to MemRef.
    Type outputTensorType = convertOp.getResult().getType();
    Type convertedType = this->typeConverter->convertType(outputTensorType);
    int64_t alignment =
        KrnlTypeConverter::getDefaultAllocAlignment(outputTensorType);
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");

    // Types use archVL and archVL / 2.
    Type f16Type = rewriter.getF16Type();
    Type f32Type = rewriter.getF32Type();
    VectorType vecF16Type = VectorType::get({archVL}, f16Type);
    VectorType vecF32Type = VectorType::get({archVLHalf}, f32Type);

    // Compute output dims.
    DimsExpr outputDims;
    ONNXUnaryOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();
    IndexExprScope allocScope(create.vec, shapeHelper.getScope());
    getIndexExprList<SymIE>(shapeHelper.getOutputDims(), outputDims);

    // Alloc memory with padding for SIMD using totVL.
    MemRefType outputMemRefType = mlir::cast<MemRefType>(convertedType);
    Value alloc = create.mem.alignedAllocWithSimdPadding(
        outputMemRefType, outputDims, totVL, alignment);

    // Flatten the input to 1D.
    int64_t collapsedInnermostLoops = rank;
    DimsExpr inputDims, flattenedInputDims;
    create.krnlIE.getShapeAsSymbols(X, inputDims);
    Value flatInput = create.mem.reshapeToFlatInnermost(
        X, inputDims, flattenedInputDims, collapsedInnermostLoops);

    // Flatten the output to 1D.
    SmallVector<IndexExpr, 4> flattenedOutputDims;
    Value flatOutput = create.mem.reshapeToFlatInnermost(
        alloc, outputDims, flattenedOutputDims, collapsedInnermostLoops);

    // Create loop iteration (flattened to 1D) and block it by totVL.
    DimsExpr lbs = {LitIE(0)};
    bool useParallel = false;
    if (enableParallel) {
      int64_t parId;
      int64_t tripCount = flattenedOutputDims[0].isLiteral()
                              ? std::ceil(flattenedOutputDims[0].getLiteral() /
                                          static_cast<float>(archVL))
                              : -1;
      if (findSuitableParallelDimension(lbs, flattenedOutputDims, 0, 1, parId,
              /*min iter for going parallel*/ 1024)) {
        useParallel = true;
        onnxToKrnlParallelReport(op, /*successful*/ true, 0, tripCount,
            "dlf16-f32 conversion fully parallelized");
      } else {
        onnxToKrnlParallelReport(op, false, 0, tripCount,
            "not enough work for dlf16-f32 conversion");
      }
    }
    onnxToKrnlSimdReport(op, /*successful*/ true, archVL,
        flattenedOutputDims[0].isLiteral() ? flattenedOutputDims[0].getLiteral()
                                           : -1,
        "dlf16-f32 conversion fully flattened");

    create.krnl.forLoopIE(lbs[0], flattenedOutputDims[0], totVL, useParallel,
        [&](const KrnlBuilder &b, ValueRange loopInd) {
          MDBuilder create(b);
          // Manually unrolled loop, add archVL offset at each iterations.
          for (int64_t u = 0; u < unrollVL; ++u) {
            Value baseIdx = create.math.add(
                loopInd[0], create.math.constantIndex(u * archVL));
            Value baseIdxNext =
                create.math.add(baseIdx, create.math.constantIndex(archVLHalf));
            if (fromF32) {
              // F32 -> DLF16
              // Load archVL f32 values from the input into two vectors each
              // with archVLHalf f32 values.
              Value vecF32H = create.vec.load(vecF32Type, flatInput, {baseIdx});
              Value vecF32L =
                  create.vec.load(vecF32Type, flatInput, {baseIdxNext});
              Value vecF16 = rewriter.create<ZLowConvertF32ToDLF16VectorOp>(
                  loc, vecF32H, vecF32L);
              // Store archVL f16 values back to the output.
              create.vec.store(vecF16, flatOutput, {baseIdx});
            } else {
              // DLF16 -> F32
              // Load archVL f16 values from the input into a register.
              Value vecF16 = create.vec.load(vecF16Type, flatInput, {baseIdx});
              auto convertOp =
                  rewriter.create<ZLowConvertDLF16ToF32VectorOp>(loc, vecF16);
              Value vecF32H = convertOp.getResult(0);
              Value vecF32L = convertOp.getResult(1);
              // Store f32 values back to the output.
              create.vec.store(vecF32H, flatOutput, {baseIdx});
              create.vec.store(vecF32L, flatOutput, {baseIdxNext});
            }
          }
        });

    rewriter.replaceOp(convertOp, alloc);

    return success();
  }
};

void populateZHighToZLowConversionPattern(mlir::RewritePatternSet &patterns,
    mlir::TypeConverter &typeConverter, mlir::MLIRContext *ctx, bool enableSIMD,
    bool enableParallel) {
  // Stickify and unstickify operations.
  patterns.insert<ZHighToZLowStickifiedConstantOpLowering>(typeConverter, ctx);
  patterns.insert<ZHighToZLowStickOpLowering>(typeConverter, ctx);
  patterns.insert<ZHighToZLowStickForLSTMOpLowering>(typeConverter, ctx);
  patterns.insert<ZHighToZLowStickForGRUOpLowering>(typeConverter, ctx);
  patterns.insert<ZHighToZLowStickifiedConstantOfShapeOpLowering>(
      typeConverter, ctx);
  patterns.insert<ZHighToZLowUnstickOpLowering>(typeConverter, ctx);
  patterns.insert<ZHighToZLowDataConversionLowering<ZHighDLF16ToF32Op>>(
      typeConverter, ctx, /*fromF32=*/false, enableParallel);
  patterns.insert<ZHighToZLowDataConversionLowering<ZHighF32ToDLF16Op>>(
      typeConverter, ctx, /*fromF32=*/true, enableParallel);
  // Binary operations
  patterns.insert<ZHighToZLowBinaryOpLowering<ZHighAddOp>>(typeConverter, ctx);
  patterns.insert<ZHighToZLowBinaryOpLowering<ZHighSubOp>>(typeConverter, ctx);
  patterns.insert<ZHighToZLowBinaryOpLowering<ZHighMulOp>>(typeConverter, ctx);
  patterns.insert<ZHighToZLowBinaryOpLowering<ZHighDivOp>>(typeConverter, ctx);
  patterns.insert<ZHighToZLowBinaryOpLowering<ZHighMinOp>>(typeConverter, ctx);
  patterns.insert<ZHighToZLowBinaryOpLowering<ZHighMaxOp>>(typeConverter, ctx);
  // Activations
  patterns.insert<ZHighToZLowUnaryOpLowering<ZHighLogOp>>(typeConverter, ctx);
  patterns.insert<ZHighToZLowUnaryOpLowering<ZHighExpOp>>(typeConverter, ctx);
  patterns.insert<ZHighToZLowUnaryOpLowering<ZHighInvSqrtOp>>(
      typeConverter, ctx);
  patterns.insert<ZHighToZLowUnaryOpLowering<ZHighReluOp>>(typeConverter, ctx);
  patterns.insert<ZHighToZLowUnaryOpLowering<ZHighGeluOp>>(typeConverter, ctx);
  patterns.insert<ZHighToZLowUnaryOpLowering<ZHighSqrtOp>>(typeConverter, ctx);
  patterns.insert<ZHighToZLowUnaryOpLowering<ZHighTanhOp>>(typeConverter, ctx);
  patterns.insert<ZHighToZLowUnaryOpLowering<ZHighSigmoidOp>>(
      typeConverter, ctx);
  // Reshape operations.
  patterns.insert<ZHighToZLowReshapeOpLowering>(typeConverter, ctx);
  // Neural network operations.
  patterns.insert<ZHighToZLowReduceOpLowering<ZHighReduceMaxOp>>(
      typeConverter, ctx);
  patterns.insert<ZHighToZLowReduceOpLowering<ZHighReduceMinOp>>(
      typeConverter, ctx);
  patterns.insert<ZHighToZLowSoftmaxOpLowering>(typeConverter, ctx);
  patterns.insert<ZHighToZLowMeanReduce2DOpLowering>(typeConverter, ctx);
  patterns.insert<ZHighToZLowLeakyReluOpLowering>(typeConverter, ctx);
  patterns.insert<ZHighToZLowMatMulOpLowering>(typeConverter, ctx);
  patterns.insert<ZHighToZLowLSTMOpLowering>(typeConverter, ctx);
  patterns.insert<ZHighToZLowGRUOpLowering>(typeConverter, ctx);
  patterns.insert<ZHighToZLowFixGRUYOpLowering>(typeConverter, ctx);
  patterns.insert<ZHighToZLowFixGRUYhOpLowering>(typeConverter, ctx);
  patterns.insert<ZHighToZLowBatchNormOpLowering>(typeConverter, ctx);
  patterns.insert<ZHighToZLowConv2DOpLowering>(typeConverter, ctx);
  patterns
      .insert<ZHighToZLowPool2DOpLowering<ZHighMaxPool2DOp, ZLowMaxPool2DOp>>(
          typeConverter, ctx);
  patterns
      .insert<ZHighToZLowPool2DOpLowering<ZHighAvgPool2DOp, ZLowAvgPool2DOp>>(
          typeConverter, ctx);
  // Quantized operations.
  patterns.insert<ZHighToZLowQuantizedStickOpLowering>(
      typeConverter, ctx, enableSIMD, enableParallel);
  patterns.insert<ZHighToZLowQuantizedMatMulOpLowering>(typeConverter, ctx);
}

} // namespace zhigh
} // namespace onnx_mlir
