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

#include "src/Accelerators/NNPA/Conversion/ZHighToZLow/ZHighToZLow.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/OpHelper.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/ShapeHelper.hpp"
#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowOps.hpp"
#include "src/Accelerators/NNPA/Pass/NNPAPasses.hpp"
#include "src/Accelerators/NNPA/Runtime/zDNNExtension/zDNNExtension.h"
#include "src/Accelerators/NNPA/Support/LayoutHelper.hpp"
#include "src/Accelerators/NNPA/Support/Stickify/Convert.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

#define DEBUG_TYPE "zhigh-to-zlow"

using namespace mlir;
using namespace onnx_mlir::zlow;

namespace onnx_mlir {
namespace zhigh {

using MDBuilder = MultiDialectBuilder<IndexExprBuilderForKrnl, KrnlBuilder,
    MathBuilder, MemRefBuilder, VectorBuilder, AffineBuilder>;

//===----------------------------------------------------------------------===//
// Helper function of Zhigh to Zlow lowering
// Insert an allocation for the given dimensions and layout.
// By default, set alignment to 4K.
//===----------------------------------------------------------------------===//

Value insertAllocForZMemRefByDim(ArrayRef<IndexExpr> dims,
    ZTensorEncodingAttr::DataLayout layout, Operation *op,
    PatternRewriter &rewriter, int64_t alignment = gAlignment) {
  // Construct a MemRefType for the given dimensions and element type.
  SmallVector<int64_t, 4> shape;
  for (IndexExpr d : dims)
    shape.emplace_back((d.isLiteral() ? d.getLiteral() : ShapedType::kDynamic));
  RankedTensorType tensorType =
      RankedTensorType::get(shape, rewriter.getF32Type(),
          ZTensorEncodingAttr::get(op->getContext(), layout));
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
  IndexExpr Lit2 = LiteralIndexExpr(2);
  IndexExpr NumOfGatesLit = LiteralIndexExpr(numOfGates);
  IndexExpr dim1 = hiddenWeightDims[1];
  IndexExpr dim2 = inputDims[1];
  IndexExpr dim3 = LiteralIndexExpr(1);
  IndexExpr dim4 = NumOfGatesLit * timestepExp + NumOfGatesLit + Lit2;

  IndexExpr Lit1 = LiteralIndexExpr(1);
  IndexExpr Lit32 = LiteralIndexExpr(32);
  IndexExpr Lit64 = LiteralIndexExpr(64);
  IndexExpr Lit4K = LiteralIndexExpr(4096);
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

/// This function emits a buffer of zero elements for the given dimensions and
/// layout. If the given dimensions are static, then a stickified constant is
/// returned.
Value insertAllocOrEmitZeroConstant(ArrayRef<IndexExpr> dims,
    ZTensorEncodingAttr::DataLayout layout, Operation *op,
    PatternRewriter &rewriter, Location loc) {
  Value res;
  bool allStaticDims =
      llvm::all_of(dims, [](IndexExpr ie) { return ie.isLiteral(); });
  if (allStaticDims) {
    // Construct a MemRefType for the given dimensions and element type.
    SmallVector<int64_t, 4> shape;
    for (IndexExpr d : dims)
      shape.emplace_back(d.getLiteral());
    RankedTensorType tensorType =
        RankedTensorType::get(shape, rewriter.getF32Type(),
            ZTensorEncodingAttr::get(op->getContext(), layout));
    ZMemRefType zMemRefType = convertZTensorToMemRefType(tensorType);
    MemRefType resType =
        affine::normalizeMemRefType(zMemRefType.value.cast<MemRefType>());

    // Create a ZHighStickifiedConstantOp.
    ZHighStickifiedConstantOp stickifiedConstant =
        rewriter.create<ZHighStickifiedConstantOp>(loc, resType,
            /*value=*/nullptr,
            /*alignment=*/rewriter.getI64IntegerAttr(4096));

    // Use an dense resource attribute to store stickified data.
    // Attribute type: tensor<sizeInBytes x i8>
    int64_t sizeInBytes =
        affine::getIntOrFloatMemRefSizeInBytes(resType).value();
    char *rawData = (char *)malloc(sizeInBytes);
    memset(rawData, 0, sizeInBytes);
    DenseResourceElementsAttr valueAttr = DenseUI8ResourceElementsAttr::get(
        RankedTensorType::get({sizeInBytes}, rewriter.getI8Type()),
        stickifiedConstant.getOperation()
            ->getDialect()
            ->getNamespace(), // use the dialect as the blob "hint"
        HeapAsmResourceBlob::allocateAndCopyWithAlign(
            llvm::ArrayRef(rawData, sizeInBytes), alignof(char)));
    stickifiedConstant.setValueAttr(valueAttr);
    free(rawData);

    res = stickifiedConstant.getResult();
  } else {
    MultiDialectBuilder<KrnlBuilder, MathBuilder> create(rewriter, loc);
    res = insertAllocForZMemRefByDim(dims, layout, op, rewriter);
    Value initValue = create.math.constant(rewriter.getF16Type(), 0);
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
      {(int64_t)originalDims.size()}, rewriter.getIntegerType(64));
  Value shapeMemRef = create.mem.alignedAlloc(shapeMemRefType);
  for (uint64_t i = 0; i < originalDims.size(); ++i) {
    Value dim =
        create.math.cast(rewriter.getI64Type(), originalDims[i].getValue());
    create.krnl.storeIE(dim, shapeMemRef, {LiteralIndexExpr(i)});
  }
  return shapeMemRef;
}

/// Get the corresponding MemRefType and layout of a given ZTensorType.
ZMemRefType convertZTensorToMemRefType(Type type) {
  ZMemRefType resZMemRefType;
  if (type.isa<TensorType>()) {
    OpBuilder b(type.getContext());
    RankedTensorType tensorType = type.dyn_cast<RankedTensorType>();
    assert(tensorType && "expected only ranked shapes");
    ArrayRef<int64_t> shape = tensorType.getShape();
    Type elementType = tensorType.getElementType();
    int64_t rank = shape.size();
    if (tensorType.getEncoding()) {
      // Obtain element type and affine map.
      AffineExpr constExpr0 = getAffineConstantExpr(0, b.getContext());
      AffineExpr constExpr31 = getAffineConstantExpr(31, b.getContext());
      AffineExpr constExpr32 = getAffineConstantExpr(32, b.getContext());
      AffineExpr constExpr64 = getAffineConstantExpr(64, b.getContext());
      unsigned e4, e3, e2, e1;
      AffineExpr n, c, h, w, res32, res64;
      SmallVector<AffineExpr, 6> dimExpr;

      ZTensorEncodingAttr::DataLayout layout = getZTensorLayout(tensorType);
      if (layout == ZTensorEncodingAttr::DataLayout::_1D) {
        // (e1) -> (1, 1, 1, e1) -> (1, ceil(e1/64), 1, 1, 32, 64)
        e1 = 0;
        n = constExpr0;
        h = b.getAffineDimExpr(e1).floorDiv(constExpr64);
        w = constExpr0;
        c = constExpr0;
        res32 = constExpr31;
        res64 = b.getAffineDimExpr(e1) % constExpr64;
      } else if (layout == ZTensorEncodingAttr::DataLayout::_2D) {
        // (e2, e1) -> (1, 1, e2, e1) -> (1, ceil(e1/64), 1, ceil(e2/32), 32
        // 64)
        e2 = 0;
        e1 = 1;
        n = constExpr0;
        h = b.getAffineDimExpr(e1).floorDiv(constExpr64);
        w = constExpr0;
        c = b.getAffineDimExpr(e2).floorDiv(constExpr32);
        res32 = b.getAffineDimExpr(e2) % constExpr32;
        res64 = b.getAffineDimExpr(e1) % constExpr64;
      } else if (layout == ZTensorEncodingAttr::DataLayout::_3D) {
        // (e3, e2, e1) -> (1, e3, e2, e1)
        // -> (1, ceil(e1/64), e3, ceil(e2/32), 32, 64)
        e3 = 0;
        e2 = 1;
        e1 = 2;
        n = constExpr0;
        h = b.getAffineDimExpr(e1).floorDiv(constExpr64);
        w = b.getAffineDimExpr(e3);
        c = b.getAffineDimExpr(e2).floorDiv(constExpr32);
        res32 = b.getAffineDimExpr(e2) % constExpr32;
        res64 = b.getAffineDimExpr(e1) % constExpr64;
      } else if (layout == ZTensorEncodingAttr::DataLayout::_4D) {
        // (e4, e3, e2, e1) -> (e4, ceil(e1/64), e3, ceil(e2/32), 32, 64)
        e4 = 0;
        e3 = 1;
        e2 = 2;
        e1 = 3;
        n = b.getAffineDimExpr(e4);
        h = b.getAffineDimExpr(e1).floorDiv(constExpr64);
        w = b.getAffineDimExpr(e3);
        c = b.getAffineDimExpr(e2).floorDiv(constExpr32);
        res32 = b.getAffineDimExpr(e2) % constExpr32;
        res64 = b.getAffineDimExpr(e1) % constExpr64;
      } else if (layout == ZTensorEncodingAttr::DataLayout::_2DS) {
        // (e4, e1) -> (e4, 1, 1, e1) -> (e4, ceil(e1/64), 1, 1, 32, 64)
        e4 = 0;
        e1 = 1;
        n = b.getAffineDimExpr(e4);
        h = b.getAffineDimExpr(e1).floorDiv(constExpr64);
        w = constExpr0;
        c = constExpr0;
        res32 = constExpr31;
        res64 = b.getAffineDimExpr(e1) % constExpr64;
      } else if (layout == ZTensorEncodingAttr::DataLayout::_3DS) {
        // (e4, e2, e1) -> (e4, 1, e2, e1)
        // -> (e4, ceil(e1/64), 1, ceil(e2/32), 32, 64)
        e4 = 0;
        e2 = 1;
        e1 = 2;
        n = b.getAffineDimExpr(e4);
        h = b.getAffineDimExpr(e1).floorDiv(constExpr64);
        w = constExpr0;
        c = b.getAffineDimExpr(e2).floorDiv(constExpr32);
        res32 = b.getAffineDimExpr(e2) % constExpr32;
        res64 = b.getAffineDimExpr(e1) % constExpr64;
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
          h = b.getAffineDimExpr(e1).floorDiv(constExpr64);
        } else {
          AffineExpr padded_e1 =
              b.getAffineDimExpr(e1).ceilDiv(constExpr64) * constExpr64;
          h = (2 * padded_e1).floorDiv(constExpr64);
        }
        w = b.getAffineDimExpr(e3);
        c = b.getAffineDimExpr(e2).floorDiv(constExpr32);
        res32 = b.getAffineDimExpr(e2) % constExpr32;
        res64 = b.getAffineDimExpr(e1) % constExpr64;
      } else if (layout == ZTensorEncodingAttr::DataLayout::NHWC) {
        // (e4, e3, e2, e1) -> (e4, ceil(e1/64), e3, ceil(e2/32), 32, 64)
        e4 = 0;
        e3 = 1;
        e2 = 2;
        e1 = 3;
        n = b.getAffineDimExpr(e4);
        h = b.getAffineDimExpr(e1).floorDiv(constExpr64);
        w = b.getAffineDimExpr(e3);
        c = b.getAffineDimExpr(e2).floorDiv(constExpr32);
        res32 = b.getAffineDimExpr(e2) % constExpr32;
        res64 = b.getAffineDimExpr(e1) % constExpr64;
      } else if (layout == ZTensorEncodingAttr::DataLayout::NCHW) {
        // (e4, e3, e2, e1) -> (e4, ceil(e2/64), e1, ceil(e3/32), 32, 64)
        llvm_unreachable("Not tested yet");
        e4 = 0;
        e3 = 1;
        e2 = 2;
        e1 = 3;
        n = b.getAffineDimExpr(e4);
        h = b.getAffineDimExpr(e2).floorDiv(constExpr64);
        w = b.getAffineDimExpr(e1);
        c = b.getAffineDimExpr(e3).floorDiv(constExpr32);
        res32 = b.getAffineDimExpr(e3) % constExpr32;
        res64 = b.getAffineDimExpr(e2) % constExpr64;
      } else if (layout == ZTensorEncodingAttr::DataLayout::HWCK) {
        // HWCK (e4, e3, e2, e1) -> KHWC (ceil(e1/64), e4,, e3, ceil(e2/32),
        // 32, 64)
        e4 = 0;
        e3 = 1;
        e2 = 2;
        e1 = 3;
        n = b.getAffineDimExpr(e1).floorDiv(constExpr64);
        h = b.getAffineDimExpr(e4);
        w = b.getAffineDimExpr(e3);
        c = b.getAffineDimExpr(e2).floorDiv(constExpr32);
        res32 = b.getAffineDimExpr(e2) % constExpr32;
        res64 = b.getAffineDimExpr(e1) % constExpr64;
      } else if (layout == ZTensorEncodingAttr::DataLayout::FICO) {
        // (e4, e3, e2, e1) -> (e4, 4*ceil(e1/4/64), e3, ceil(e2/32), 32, 64)
        assert(!ShapedType::isDynamic(shape[rank - 1]) &&
               (shape[rank - 1] % 4) == 0 &&
               "wrong concatenated dimension size");
        int64_t s = shape[rank - 1] / 4;
        // ((s + 64 - 1) / 64) * 64;
        int64_t s_pad = ceil((double)s / 64) * 64;
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
                .floorDiv(constExpr64);
        c = b.getAffineDimExpr(e2).floorDiv(constExpr32);
        res32 = b.getAffineDimExpr(e2) % constExpr32;
        res64 = (b.getAffineDimExpr(e1) +
                    pad_size * (b.getAffineDimExpr(e1).floorDiv(constExprS))) %
                constExpr64;
      } else if (layout == ZTensorEncodingAttr::DataLayout::ZRH) {
        // (e4, e3, e2, e1) -> (e4, 3*ceil(e1/4/64), e3, ceil(e2/32), 32, 64)
        int64_t hidden_size = shape[rank - 1];
        assert(hidden_size > 0 &&
               "Dynamic dimension in hidden_size not supported "
               "in affine_map generation.");
        assert((hidden_size % 3) == 0 && "wrong concatenated dimension size.");
        int64_t s = hidden_size / 3;
        int64_t s_pad = ceil((float)s / 64) * 64; // ((s + 64 - 1) / 64) * 64;
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
                .floorDiv(constExpr64);
        c = b.getAffineDimExpr(e2).floorDiv(constExpr32);
        res32 = b.getAffineDimExpr(e2) % constExpr32;
        res64 = (b.getAffineDimExpr(e1) +
                    pad_size * (b.getAffineDimExpr(e1).floorDiv(constExprS))) %
                constExpr64;
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
      MemRefType outType = MemRefType::get(shape, b.getF16Type());
      resZMemRefType.value =
          MemRefType::Builder(outType).setLayout(AffineMapAttr::get(smap));
      resZMemRefType.layout = convertZTensorDataLayoutToStringAttr(b, layout);
    } else { // Does not have tensorType.getEncoding().
      resZMemRefType.value = MemRefType::get(shape, elementType);
    }
  } else { // Not type.isa<TensorType>().
    resZMemRefType.value = type.dyn_cast<MemRefType>();
  }
  return resZMemRefType;
}

//===----------------------------------------------------------------------===//
// Lower ZHigh Stick to ZLow Stick
//===----------------------------------------------------------------------===//

// Initial support for flatten ztensor

struct ZHighToZLowStickOpLowering : public ConversionPattern {
  ZHighToZLowStickOpLowering(TypeConverter &typeConverter, MLIRContext *ctx,
      bool enableParallel, bool enableCompilerStickUnstickCodeGen)
      : ConversionPattern(
            typeConverter, ZHighStickOp::getOperationName(), 1, ctx),
        enableParallel(enableParallel),
        enableCompilerCodeGen(enableCompilerStickUnstickCodeGen) {}
  bool enableParallel;
  bool enableCompilerCodeGen;

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();

    ZHighStickOpAdaptor operandAdaptor(operands);
    StringAttr layout = cast<ZHighStickOp>(op).getLayoutAttr();

    IndexExprBuilderForKrnl createKrnlIE(rewriter, loc);
    ZHighStickOpShapeHelper shapeHelper(op, operands, &createKrnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Convert ZTensor type to MemRefType.
    ZMemRefType zMemRefType =
        convertZTensorToMemRefType(*op->result_type_begin());

    // Allocate a buffer for the result MemRef.
    Value alloc = insertAllocForZMemRef(
        zMemRefType, shapeHelper.getOutputDims(), op, rewriter);
    // Set pre-transformed layout: if NHWC, we can directly stickify from NCHW.
    if (isNHWCLayout(layout))
      layout = getNCHWLayoutAttr(rewriter);

    if (enableCompilerCodeGen) {
#if 0
      // Think we only come in here when condition below is true.
      if (layout.getValue().equals_insensitive("4D") ||
          layout.getValue().equals_insensitive("3D") ||
          layout.getValue().equals_insensitive("2D") ||
          layout.getValue().equals_insensitive("1D") ||
          layout.getValue().equals_insensitive("3DS") ||
          layout.getValue().equals_insensitive("2DS")) {
        return generateStickCode(
            rewriter, op, shapeHelper, alloc, operandAdaptor.getIn(), layout);
      }

#else
      if (layout.getValue().equals_insensitive("3DS")) {
        return generateStickCode3D4D(
            rewriter, op, shapeHelper, alloc, operandAdaptor.getIn(), layout);
      }
#endif
    }

    // Emit a ZLow operation.
    rewriter.create<ZLowStickOp>(loc, operandAdaptor.getIn(), alloc, layout);
    rewriter.replaceOp(op, alloc);
    return success();
  }

  LogicalResult generateStickCode3D4D(ConversionPatternRewriter &rewriter,
      Operation *op, ZHighStickOpShapeHelper &shapeHelper, Value alloc,
      Value input, StringAttr layout) const {
    fprintf(stderr, "hi alex with memcpy\n");
    // generate code (unoptimized)
    int64_t rank = shapeHelper.getOutputDims().size();
    assert((rank == 3 || rank == 4) && "3D/4D expected");
    // Dims: e2 is second to last, e1 is last.
    int64_t e2Ind = rank - 2, e1Ind = rank - 1;

    Location loc = op->getLoc();
    MDBuilder create(rewriter, loc);

    // Compute output dims.
    IndexExprScope allocScope(create.krnl, shapeHelper.getScope());
    DimsExpr outputDims;
    getIndexExprList<SymbolIndexExpr>(shapeHelper.getOutputDims(), outputDims);

    // Special dimensions of output
    int64_t D1 = AIU_2BYTE_CELLS_PER_STICK;
    int64_t D2 = AIU_STICKS_PER_PAGE;
    IndexExpr E1ByD1 = outputDims[e1Ind].ceilDiv(D1);
    IndexExpr E2ByD2 = outputDims[e2Ind].ceilDiv(D2);

    // Tiling in the e2 x e1 dim: N x M.
    int64_t N = 2;
    assert(D2 % N == 0 && "Tiling by N (along E2) must divide 32");
    int64_t M = 4; // Tiling along E1 by M * 64 values.
    // Info for SIMD.
    int64_t VL = 8;          // FP16 VL.
    int64_t VLHalf = VL / 2; // FP32 VL.
    assert(D1 % VL == 0 && "SIMD vector length must divide 64");
    int64_t D1ByVL = D1 / VL;

    // Literals for further use.
    IndexExpr litZero = LiteralIndexExpr(0);
    IndexExpr litVL = LiteralIndexExpr(VL);
    IndexExpr litVLHalf = LiteralIndexExpr(VLHalf);
    IndexExpr litN = LiteralIndexExpr(N);
    IndexExpr litM = LiteralIndexExpr(M);
    IndexExpr litD1ByVL = LiteralIndexExpr(D1ByVL);
    IndexExpr litD1 = LiteralIndexExpr(D1);
    IndexExpr litD2 = LiteralIndexExpr(D2);
    IndexExpr litNxD1 = LiteralIndexExpr(N * D1);
    IndexExpr litD1xD2 = LiteralIndexExpr(D1 * D2);

    // Types use the SIMD unrolling VL and VLHalf.
    Type f16Type = rewriter.getF16Type();
    Type f32Type = rewriter.getF32Type();
    VectorType vecF32Type = VectorType::get({VLHalf}, f32Type);

    // Type for buffer
    MemRefType bufferType = MemRefType::get({M, N, D1}, f16Type);

    // Cast the alloc output buffer: reduces all but last 2 mapped dimensions to
    // one (which are set to 32x64). Dx = E4 * E3 * Ceil(E2, 32) * Ceil(E1, 64)
    IndexExpr Dx = LiteralIndexExpr(1);
    for (int64_t i = 0; i < rank - 2; ++i)
      Dx = Dx * outputDims[i];
    Dx = Dx * E2ByD2;
    Dx = Dx * E1ByD1;
    DimsExpr outputDimsAs32x64 = {Dx, litD2, litD1};
    Value allocAs32x64 = create.mem.reinterpretCast(alloc, outputDimsAs32x64);

    // Create loop iterations.
    ValueRange loopDefs = create.krnl.defineLoops(rank);
    Value defE2 = loopDefs[e2Ind]; // Second to last dim.
    Value defE1 = loopDefs[e1Ind]; // Last dim.
    ValueRange tiledDefE2 = create.krnl.block(defE2, N);
    ValueRange tiledDefE1 = create.krnl.block(defE1, M * D1);
    ValueRange innerTiledDefE1 = create.krnl.block(tiledDefE1[1], D1);
    // Permute loop defs
    if (rank == 3) {
      // Final order: E3, E2 tiled by N, E1 tiled by M*D1, followed by E1 tiled
      // by D1 (inside M), E2 (inside N), E1 (inside D1). The last 2 form a N x
      // 64 bytes and are mem-copied, so these loops are not used.
      create.krnl.permute(
          {/*E3*/ loopDefs[0],
              /*E2*/ tiledDefE2[0], tiledDefE2[1],
              /*E1*/ tiledDefE1[0], innerTiledDefE1[0], innerTiledDefE1[1]},
          {/*E3*/ 0, /*E2*/ 1, 4, /*E1*/ 2, 3, 5});
    } else {
      // Same, but with E4 first, all other indices increased by one.
      create.krnl.permute(
          {/*E4*/ loopDefs[0], /*E3*/ loopDefs[1],
              /*E2*/ tiledDefE2[0], tiledDefE2[1],
              /*E1*/ tiledDefE1[0], innerTiledDefE1[0], innerTiledDefE1[1]},
          {/*E4*/ 0, /*E3*/ 1, /*E2*/ 2, 5, /*E1*/ 3, 4, 6});
    }

    // Bounds and optimized loop defs.
    DimsExpr lbs(rank, litZero);
    DimsExpr ubs = outputDims;
    SmallVector<Value, 4> optLoopDefs;
    // Outer dims: possibly e4, e3.
    for (int64_t i = 0; i < rank - 2; ++i)
      optLoopDefs.emplace_back(loopDefs[i]);
    // Tiled dims e2, which iterates over N elements.
    optLoopDefs.emplace_back(tiledDefE2[0]);
    // Tiled dims: for e1, which iterates over M*D1 elements.
    optLoopDefs.emplace_back(tiledDefE1[0]);

    // Parallel...
    if (enableParallel) {
      int64_t parId;
      if (findSuitableParallelDimension(lbs, ubs, 0, rank - 2, parId, 8)) {
        create.krnl.parallel(optLoopDefs[parId]);
        onnxToKrnlParallelReport(op, true, parId, lbs[parId], ubs[parId],
            "compiler-generated stickify");
      } else {
        onnxToKrnlParallelReport(op, false, -1, -1,
            "no dim with enough work in compiler-generated stickify");
      }
    }

    // Outer loop (E4, E3, E2 tiled by N, E1 tiled by M*D1)
    create.krnl.iterateIE(loopDefs, optLoopDefs, lbs, ubs,
        [&](KrnlBuilder &b, ValueRange loopInd) {
          MDBuilder create(b);
          // Create access function using the outer indices;
          SmallVector<Value, 4> inputAF, outputAF;
          for (int i = 0; i < rank - 2; ++i) {
            inputAF.emplace_back(loopInd[i]);
            outputAF.emplace_back(loopInd[i]);
          }
          Value tiledE2ByN = loopInd[e2Ind];
          Value tiledE1ByMxD1 = loopInd[e1Ind];
          // Create buffer
          Value buffer = create.mem.alignedAlloc(bufferType, {});
          // Iterate over N, M, and D1. Manage iterations individually.
          DimsExpr lbs2(3, litZero);
          DimsExpr ubs2 = {litN, litM, litD1};
          SmallVector<int64_t, 3> steps2 = {1, 1, VL};
          // Analysis of assembly showed that the inner loop was fully unrolled.
          create.affine.forIE(
              lbs2, ubs2, steps2, [&](AffineBuilder &b, ValueRange loopInd) {
                MDBuilder create(b);
                Value n = loopInd[0], m = loopInd[1], l = loopInd[2];
                // Create the e2, e1 for the input access function.
                Value e2 = create.math.add(tiledE2ByN, n);
                inputAF.emplace_back(e2);
                Value mTimesD1 = create.math.mul(m, litD1.getValue());
                Value e1 = create.math.add(tiledE1ByMxD1, mTimesD1);
                e1 = create.math.add(e1, l);
                inputAF.emplace_back(e1);
                Value vecF32H = create.vec.load(vecF32Type, input, inputAF);
                // Get the second half of VL in the input access function.
                Value e1Next = create.math.add(e1, litVLHalf.getValue());
                inputAF[e1Ind] = e1Next;
                Value vecF32L = create.vec.load(vecF32Type, input, inputAF);
                Value vecF16 = rewriter.create<ZLowConvertF32ToDLF16VectorOp>(
                    loc, vecF32H, vecF32L);
                create.vec.store(vecF16, buffer, {m, n, l});
              });
          // Perform copy: E1 Tiled by D1 (inside tile by M)
          create.krnl.iterate({}, {innerTiledDefE1[0]}, {}, {},
              [&](KrnlBuilder &b, ValueRange loopInd) {
                MDBuilder create(b);
                // Loop over the inner tiles data points (e1by64, e2)
                Value tiledE1ByD1 = loopInd[0];
                // Compute m as address within buffer.
                Value m = create.math.sub(tiledE1ByD1, tiledE1ByMxD1);
                m = create.math.floorDiv(m, litD1.getValue());
                // Compute offset x inside allocAs32x64
                Value e1DivD1 =
                    create.math.floorDiv(tiledE1ByD1, litD1.getValue());
                Value e2DivD2 =
                    create.math.floorDiv(tiledE2ByN, litD2.getValue());
                Value e2ModD2 = create.math.rem(tiledE2ByN, litD2.getValue());
                Value x;
                if (layout.getValue().equals_insensitive("3DS")) {
                  // x = (e3 * Ceil(E1/64) + floor(e1/64)) * Ceil(E2/32) +
                  // floor(e2/32).
                  Value e3 = outputAF[0];
                  x = create.math.mul(e3, E1ByD1.getValue());
                  x = create.math.add(x, e1DivD1);
                  x = create.math.mul(x, E2ByD2.getValue());
                  x = create.math.add(x, e2DivD2);
                } else {
                  llvm_unreachable("missing layout");
                }
                Value tileOffsetAlloc = create.math.mul(x, litD1xD2.getValue());
                Value offsetInTile = create.math.mul(e2ModD2, litD1.getValue());
                Value offsetAlloc =
                    create.math.add(tileOffsetAlloc, offsetInTile);
                // now I have to add the offset to get to the right

                // create offset in buffer
                Value offsetBuffer = create.math.mul(m, litNxD1.getValue());
                Type intType = rewriter.getIntegerType(64);
                Value num = create.math.constant(intType, N * D1);
                create.krnl.memcpy(
                    allocAs32x64, buffer, num, offsetAlloc, offsetBuffer);
              });
        });
    fprintf(stderr, "bye alex with memcpy\n");
    rewriter.replaceOp(op, alloc);
    return success();
  }

  void prepareReinterpretZTensor(DimsExpr &origE, StringAttr layout,
      DimsExpr &normE, DimsExpr &normT, IndexExpr &numTiles,
      DimsExpr &coeff) const {
    int64_t rank = origE.size();
    assert(rank >= 1 && rank <= 4 && "expected a format between 1 to 4D");
    IndexExpr lit0 = LiteralIndexExpr(0);
    IndexExpr lit1 = LiteralIndexExpr(1);
    IndexExpr lit64 = LiteralIndexExpr(AIU_2BYTE_CELLS_PER_STICK);
    IndexExpr lit32 = LiteralIndexExpr(AIU_STICKS_PER_PAGE);

    // E indices to access origE.
    int64_t E4 = rank - 4, E3 = rank - 3, E2 = rank - 2, E1 = rank - 1;
    // The extended E has always 4 dims; values depend on layout.
    bool isDS = false;
    if (layout.getValue().equals_insensitive("4D")) {
      normE = {origE[E4], origE[E3], origE[E2], origE[E1]};
    } else if (layout.getValue().equals_insensitive("3D")) {
      normE = {lit1, origE[E3], origE[E2], origE[E1]};
    } else if (layout.getValue().equals_insensitive("2D")) {
      normE = {lit1, lit1, origE[E2], origE[E1]};
    } else if (layout.getValue().equals_insensitive("1D")) {
      normE = {lit1, lit1, lit1, origE[E1]};
    } else if (layout.getValue().equals_insensitive("3DS")) {
      normE = {origE[E3], lit1, origE[E2], origE[E1]};
      isDS = true;
    } else if (layout.getValue().equals_insensitive("2DS")) {
      normE = {origE[E2], lit1, lit1, origE[E1]};
      isDS = true;
    } else {
      llvm_unreachable("format cannot be processed here");
    }
    // From here on, we only used the extended E, so update the E access indices
    // to reflect that we always have 4 values.
    E4 = 0, E3 = 1, E2 = 2, E1 = 3;
    IndexExpr t1 = normE[E1].ceilDiv(64);
    IndexExpr t2 = normE[E2].ceilDiv(32);
    normT = {normE[E4], normE[E3], t2, t1};
    // Compute the tiled dimensions Ceil(E1, 64) and Ceil(E2, 32).
    numTiles = normT[E4] * normT[E3];
    numTiles = numTiles * normT[E2];
    numTiles = numTiles * normT[E1];

    // Compute access coefficients. Below T4 is a shortcut for normT[E4].
    // Coefficients assume indices in tiles. Thus we use the normT instead of
    // the normE.
    if (!isDS) {
      // Data layout is: T4(c4), T1(c3), T3(c2), T2(c1)
      IndexExpr c1 = lit1;
      IndexExpr c2 = c1 * normT[E2];
      IndexExpr c3 = c2 * normT[E3];
      IndexExpr c4 = c3 * normT[E1];
      if (rank == 4) // 4D
        coeff = {/*t4*/ c4, /*t3*/ c2, /*t2*/ c1, /*t1*/ c3};
      else if (rank == 3) // 3D
        coeff = {/*t4*/ lit0, /*t3*/ c2, /*t2*/ c1, /*t1*/ c3};
      else if (rank == 2) // 2D
        coeff = {/*t4*/ lit0, /*t3*/ lit0, /*t2*/ c1, /*t1*/ c2};
      else // 1D
        coeff = {/*t4*/ lit0, /*t3*/ lit0, /*t2*/ lit0, /*t1*/ c1};
    } else {
      if (rank == 3) { // 3DS
        // Order is T4(c3), T1(c2), T2(c1)
        IndexExpr c1 = lit1;
        IndexExpr c2 = c1 * normT[E2];
        IndexExpr c3 = c2 * normT[E1];
        coeff = {/*t4*/ c3, /*t3*/ lit0, /*t2*/ c1, /*t1*/ c2};
      } else { // 2DS
        // Order is T4(c2), T1(c1)
        IndexExpr c1 = lit1;
        IndexExpr c2 = c1 * normT[E1];
        coeff = {/*t4*/ c2, /*t3*/ lit0, /*t2*/ lit0, /*t1*/ c1};
      }
    }
  }

  IndexExpr tileOffsetForReinterpretZTensor(
      DimsExpr &indices, DimsExpr &coeff) const {
    IndexExpr t4 = indices[0] * coeff[0];
    IndexExpr t3 = indices[1] * coeff[1];
    IndexExpr t2 = indices[2] * coeff[2];
    IndexExpr t1 = indices[3] * coeff[3];
    IndexExpr res = t4 * t3;
    res = res * t2;
    res = res * t1;
    return res;
  }

  LogicalResult generateStickCode(ConversionPatternRewriter &rewriter,
      Operation *op, ZHighStickOpShapeHelper &shapeHelper, Value alloc,
      Value input, StringAttr layout) const {
    fprintf(stderr, "hi alex with generic memcpy\n");
    // generate code (unoptimized)
    Location loc = op->getLoc();
    MDBuilder create(rewriter, loc);

    // Tiling in the e2 x e1 dim: N x M.
    int64_t N = 2;
    int64_t M = 4; // Tiling along E1 by M * 64 values.
    assert(32 % N == 0 && "Tiling by N (along E2) must divide 32");
    // Info for SIMD.
    int64_t VL = 8;          // FP16 VL.
    int64_t VLHalf = VL / 2; // FP32 VL.
    assert(64 % VL == 0 && "SIMD vector length must divide 64");

    // Compute output dims.
    IndexExprScope allocScope(create.krnl, shapeHelper.getScope());
    DimsExpr outputDims;
    getIndexExprList<SymbolIndexExpr>(shapeHelper.getOutputDims(), outputDims);

    // Prepare for cast_reinterpret
    DimsExpr normE, normT, coeff;
    IndexExpr numTiles;
    prepareReinterpretZTensor(
        outputDims, layout, normE, normT, numTiles, coeff);
    int64_t E4 = 0, E3 = 1, E2 = 2, E1 = 3;
    IndexExpr lit32 = LiteralIndexExpr(32);
    IndexExpr lit64 = LiteralIndexExpr(64);
    DimsExpr reallocOutputDims = {numTiles, lit32, lit64};
    Value allocAs32x64 = create.mem.reinterpretCast(alloc, reallocOutputDims);

    // Types use the SIMD unrolling VL and VLHalf.
    Type f16Type = rewriter.getF16Type();
    Type f32Type = rewriter.getF32Type();
    VectorType vecF32Type = VectorType::get({VLHalf}, f32Type);

    // Type for buffer
    MemRefType bufferType = MemRefType::get({M, N, 64}, f16Type);

    // Literals for further use.
    IndexExpr litZero = LiteralIndexExpr(0);
    IndexExpr litN = LiteralIndexExpr(N);
    IndexExpr litM = LiteralIndexExpr(M);

    // Create loop iterations. Use normalized iterations, which are
    // defined as a 4D vector. So use 4 loops here. The `for(i=0; i<1; ++i)`
    // loops should simplify nicely.
    // Note that we iterate over E1's of 64 elements.
    ValueRange loopDefs = create.krnl.defineLoops(4);
    ValueRange tiledDefE2 = create.krnl.block(loopDefs[E2], N);
    ValueRange tiledDefE1 = create.krnl.block(loopDefs[E1], M);
    // Final order: E4, E3, E2 tiled by N, E1 tiled by M, followed by E1 (inside
    // M), E2 (inside N).
    create.krnl.permute({/*E4*/ loopDefs[E4], /*E3*/ loopDefs[E3],
                            /*E2*/ tiledDefE2[0], tiledDefE2[1],
                            /*E1*/ tiledDefE1[0], tiledDefE1[1]},
        {/*E4*/ 0, /*E3*/ 1, /*E2*/ 2, 5, /*E1*/ 3, 4});

    // Bounds and optimized loop defs.
    DimsExpr lbs = {litZero, litZero, litZero, litZero};
    DimsExpr ubs = {normE[E4], normE[E3], normE[E2], normT[E1]};
    SmallVector<Value, 4> optLoopDefs = {
        loopDefs[E4], loopDefs[E3], tiledDefE2[0], tiledDefE1[0]};

    // Parallel...
    if (enableParallel) {
      int64_t parId;
      if (findSuitableParallelDimension(lbs, ubs, 0, 2, parId, 8)) {
        create.krnl.parallel(optLoopDefs[parId]);
        onnxToKrnlParallelReport(op, true, parId, lbs[parId], ubs[parId],
            "compiler-generated stickify");
      } else {
        onnxToKrnlParallelReport(op, false, -1, -1,
            "no dim with enough work in compiler-generated stickify");
      }
    }

    // Outer loop (E4, E3, E2 tiled by N, E1 tiled by M)
    create.krnl.iterateIE(loopDefs, optLoopDefs, lbs, ubs,
        [&](KrnlBuilder &b, ValueRange loopInd) {
          MDBuilder create(b);
          IndexExprScope outerScope(create.krnl, &allocScope);
          DimsExpr outerIndices;
          getIndexExprList<SymbolIndexExpr>(loopInd, outerIndices);

          // Create buffer
          Value buffer = create.mem.alignedAlloc(bufferType, {});
          // Iterate over N, M, and D1. Manage iterations individually.
          DimsExpr lbs2(3, litZero);
          DimsExpr ubs2 = {litN, litM, lit64};
          SmallVector<int64_t, 3> steps2 = {1, 1, VL};
          // Analysis of assembly showed that the inner loop was fully unrolled.
          create.affine.forIE(
              lbs2, ubs2, steps2, [&](AffineBuilder &b, ValueRange loopInd) {
                MDBuilder create(b);
                IndexExprScope innerScope(create.krnl, &outerScope);
                SymbolIndexExpr e4(outerIndices[E4]);
                SymbolIndexExpr e3(outerIndices[E3]);
                SymbolIndexExpr e2ByN(outerIndices[E2]);
                SymbolIndexExpr t1ByM(outerIndices[E1]);
                SymbolIndexExpr n(loopInd[0]);
                SymbolIndexExpr m(loopInd[1]);
                SymbolIndexExpr l(loopInd[2]);
                IndexExpr e2 = e2ByN + n;
                IndexExpr t1 = t1ByM + m;
                IndexExpr e1 = (t1 * 64) + l;
                Value vecF32H =
                    create.vec.loadIE(vecF32Type, input, {e4, e3, e2, e1}, {});
                IndexExpr e1Next = e1 + VLHalf;
                Value vecF32L = create.vec.loadIE(
                    vecF32Type, input, {e4, e3, e2, e1Next}, {});
                Value vecF16 = rewriter.create<ZLowConvertF32ToDLF16VectorOp>(
                    loc, vecF32H, vecF32L);
                create.vec.storeIE(vecF16, buffer, {m, n, l}, {});
              });
          // Perform copy: E1 Tiled by 64 (inside tile by M)
          create.krnl.iterate({}, {tiledDefE1[1]}, {}, {},
              [&](KrnlBuilder &b, ValueRange loopInd) {
                MDBuilder create(b);
                IndexExprScope innerScope(create.krnl, &outerScope);
                SymbolIndexExpr e4(outerIndices[E4]);
                SymbolIndexExpr e3(outerIndices[E3]);
                SymbolIndexExpr e2ByN(outerIndices[E2]);
                SymbolIndexExpr t1ByM(outerIndices[E1]);
                SymbolIndexExpr m(loopInd[0]);
                IndexExpr t1 = t1ByM + m;
                IndexExpr t2 = e2ByN.floorDiv(32);
                IndexExpr e2Mod32 = e2ByN % 32;

                // Re-actualize coefficients, and calculate alloc offset.
                DimsExpr localCoeff;
                getIndexExprList<SymbolIndexExpr>(coeff, localCoeff);
                DimsExpr tileIndices = {e4, e3, t2, t1};
                IndexExpr allocTileOffset =
                    tileOffsetForReinterpretZTensor(tileIndices, localCoeff);
                IndexExpr allocOffset = allocTileOffset * (32 * 64);
                allocOffset = allocOffset + (e2Mod32 * 64);
                // Calculate buffer offset
                IndexExpr bufferOffset = m * (N * 64);
                // Amount of values to copy
                IndexExpr num = LiteralIndexExpr(N * 64);
                // Mem copy
                create.krnl.memcpy(allocAs32x64, buffer, num.getValue(),
                    allocOffset.getValue(), bufferOffset.getValue());
              });
        });
    fprintf(stderr, "bye alex with generic memcpy\n");
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

    IndexExprBuilderForKrnl createKrnlIE(rewriter, loc);
    ZHighUnstickOpShapeHelper shapeHelper(op, operands, &createKrnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Convert ZTensor type to MemRefType.
    ZMemRefType zMemRefType =
        convertZTensorToMemRefType(*op->result_type_begin());

    // Allocate a buffer for the result MemRef.
    Value alloc = insertAllocForZMemRef(
        zMemRefType, shapeHelper.getOutputDims(), op, rewriter);

    // Set layout: if NHWC, we can directly unstickify to NCHW.
    if (isNHWCLayout(layout))
      layout = getNCHWLayoutAttr(rewriter);

    // Emit a ZLow operation.
    rewriter.create<ZLowUnstickOp>(loc, input, alloc, layout);
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
    assert(zMemRefType.value.cast<MemRefType>().getNumDynamicDims() == 0 &&
           "MemRefType has dynamic dimensions");
    MemRefType normalizedType =
        affine::normalizeMemRefType(zMemRefType.value.cast<MemRefType>());
    ArrayRef<int64_t> normalizedShape = normalizedType.getShape();

    // Get dense resource attribute.
    auto blob = stickifiedConstOp.getValue()
                    .value()
                    .cast<DenseResourceElementsAttr>()
                    .getRawHandle()
                    .getBlob();
    assert(blob && "Expecting dense resource with a valid blob");
    ArrayRef<char> data = blob->getData();

    // Validate the stickified tensor.
    int64_t memRefSizeInBytes = getMemRefEltSizeInBytes(normalizedType);
    memRefSizeInBytes *= normalizedType.getNumElements();
    assert((data.size() == (uint64_t)memRefSizeInBytes) &&
           "The stickified tensor's buffer size and MemRef's size mismatched");

    // Create a KrnlGlobalOp.
    KrnlGlobalOp constantGlobal =
        rewriter.create<KrnlGlobalOp>(loc, zMemRefType.value,
            /*shape=*/
            rewriter.getI64ArrayAttr(normalizedShape),
            /*name=*/
            rewriter.getStringAttr(
                "constant_stickify_" + std::to_string(constantID)),
            /*value=*/stickifiedConstOp.getValueAttr(),
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
struct ZLowOpFor<ZHighReluOp> {
  using Op = ZLowReluOp;
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
// Lower ZHigh MatMul to ZLow MatMul
//===----------------------------------------------------------------------===//

struct ZHighToZLowMatMulOpLowering : public ConversionPattern {
  ZHighToZLowMatMulOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, ZHighMatMulOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
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
    //      or broadcasting: X(s, m, n) * Y(n, p) + Bias(p)
    // shape is a 1D MemRef (memref<4xindex>) whose items are:
    //   - 1st item: s
    //   - 2nd item: m
    //   - 3rd item: n
    //   - 4th item: p

    Value shapeMemRef =
        insertShapeMemRefI64(rewriter, loc, shapeHelper.allOriginalDims);

    // Prepare optional bias.
    Value bias = operandAdaptor.getB();
    if (bias.getType().isa<NoneType>()) {
      SmallVector<IndexExpr, 4> resDims, biasDims;
      create.krnlIE.getShapeAsDims(alloc, resDims);
      ZTensorEncodingAttr::DataLayout biasLayout;
      if (shapeHelper.isStacked) {
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
    int64_t bcast = (shapeHelper.isBroadcasted) ? -1 : 0;
    int64_t stacked = (shapeHelper.isStacked) ? -1 : 0;
    IntegerAttr is_bcastAttr =
        rewriter.getIntegerAttr(rewriter.getIntegerType(64, true), bcast);
    IntegerAttr is_stackedAttr =
        rewriter.getIntegerAttr(rewriter.getIntegerType(64, true), stacked);

    // Emit zlow.matmul.
    rewriter.create<ZLowMatMulOp>(loc, operandAdaptor.getX(),
        operandAdaptor.getY(), bias, shapeMemRef, alloc, is_bcastAttr,
        is_stackedAttr);
    rewriter.replaceOp(op, alloc);
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
    if (initial_h.getType().isa<NoneType>()) {
      initial_h = insertAllocOrEmitZeroConstant(shapeHelper.hc0Shape,
          ZTensorEncodingAttr::DataLayout::_3DS, op, rewriter, loc);
    }
    if (initial_c.getType().isa<NoneType>()) {
      initial_c = insertAllocOrEmitZeroConstant(shapeHelper.hc0Shape,
          ZTensorEncodingAttr::DataLayout::_3DS, op, rewriter, loc);
    }
    if (input_bias.getType().isa<NoneType>()) {
      input_bias = insertAllocOrEmitZeroConstant(shapeHelper.biasShape,
          ZTensorEncodingAttr::DataLayout::FICO, op, rewriter, loc);
    }
    if (hidden_bias.getType().isa<NoneType>()) {
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
    if (initial_h.getType().isa<NoneType>()) {
      initial_h = insertAllocOrEmitZeroConstant(shapeHelper.h0Shape,
          ZTensorEncodingAttr::DataLayout::_3DS, op, rewriter, loc);
    }
    if (input_bias.getType().isa<NoneType>()) {
      input_bias = insertAllocOrEmitZeroConstant(shapeHelper.biasShape,
          ZTensorEncodingAttr::DataLayout::ZRH, op, rewriter, loc);
    }
    if (hidden_bias.getType().isa<NoneType>()) {
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
    MemRefType outputMemRefType =
        typeConverter->convertType(op->getResults()[0].getType())
            .cast<MemRefType>();

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
    //     for (directioIV for directions)
    //       for (hs for hidden states)
    //         Y[sequenceIV, directionIV, bs, hs] = initValue

    // Create loop for batch
    Value iZero = create.math.constantIndex(0);
    ValueRange batchLoop = create.krnl.defineLoops(1);
    create.krnl.iterate(batchLoop, batchLoop, {iZero}, {create.mem.dim(Y, 2)},
        [&](KrnlBuilder &createKrnl, ValueRange batchIndices) {
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
              [&](KrnlBuilder &createKrnl, ValueRange indices) {
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
        [&](KrnlBuilder &createKrnl, ValueRange indices) {
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
    MemRefType outputMemRefType =
        typeConverter->convertType(op->getResults()[0].getType())
            .cast<MemRefType>();
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
        [&](KrnlBuilder &createKrnl, ValueRange indices) {
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
    if (bias.getType().isa<NoneType>()) {
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

    auto stickOp = cast<ZHighStickifiedConstantOfShapeOp>(op);
    FloatAttr value = stickOp.getValueAttr();
    Type i16Ty = rewriter.getI16Type();
    Type i64Ty = rewriter.getI64Type();
    Type f16Ty = rewriter.getF16Type();

    // Convert the scalar value to dlfloat16.
    // Use uint16_t as container.
    float valueF32 = (float)value.getValueAsDouble();
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
    uint64_t rank = res.getType().cast<MemRefType>().getRank();
    ValueRange loopDef = create.krnl.defineLoops(rank);
    SmallVector<IndexExpr, 4> lbs(rank, LiteralIndexExpr(0));
    SmallVector<IndexExpr, 4> ubs = shapeHelper.getOutputDims();
    create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
        [&](KrnlBuilder &createKrnl, ValueRange indices) {
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
    // Fixed VL for the conversion instruction: 8 elements per instruction call.
    // Because the VL of the zlow.conversions are not "virtualized" in lengths,
    // we manually unroll the loop containing the SIMD operations manually.
    // Experiments on a 1024x1024 tensors shows best results with an unrolling
    // of 8 SIMD vectors.
    int64_t VL = 8;
    int64_t VLHalf = VL / 2;
    int64_t unrollSIMD = 8;             // Manually unroll the SIMD loop.
    int64_t unrollVL = unrollSIMD * VL; // Total numbers of values unrolled.

    // Convert the output type to MemRef.
    Type outputTensorType = convertOp.getResult().getType();
    Type convertedType = this->typeConverter->convertType(outputTensorType);
    int64_t alignment =
        KrnlTypeConverter::getDefaultAllocAlignment(outputTensorType);
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");

    // Types use the SIMD unrolling VL and VLHalf.
    Type f16Type = rewriter.getF16Type();
    Type f32Type = rewriter.getF32Type();
    VectorType vecF16Type = VectorType::get({VL}, f16Type);
    VectorType vecF32Type = VectorType::get({VLHalf}, f32Type);

    // Compute output dims.
    DimsExpr outputDims;
    ONNXUnaryOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();
    IndexExprScope allocScope(create.vec, shapeHelper.getScope());
    getIndexExprList<SymbolIndexExpr>(shapeHelper.getOutputDims(), outputDims);

    // Alloc memory with padding for SIMD. Padding and loop unrolling use
    // unrollVL.
    MemRefType outputMemRefType = convertedType.cast<MemRefType>();
    Value alloc = create.mem.alignedAllocWithSimdPadding(
        outputMemRefType, outputDims, unrollVL, alignment);

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

    // Create loop iteration (flattened to 1D) and block it by unrollVL.
    ValueRange loopDef = create.krnl.defineLoops(1);
    ValueRange blockedLoopDef = create.krnl.block(loopDef[0], unrollVL);
    SmallVector<Value, 1> optimizedLoopDef(1, blockedLoopDef[0]);

    if (enableParallel) {
      create.krnl.parallel(blockedLoopDef[0]);
      onnxToKrnlParallelReport(op, /*successful*/ true, 0,
          flattenedOutputDims[0].isLiteral()
              ? std::ceil(flattenedOutputDims[0].getLiteral() / (float)VL)
              : -1,
          "dlf16-f32 conversion fully parallelized");
    }

    onnxToKrnlSimdReport(op, /*successful*/ true, VL,
        flattenedOutputDims[0].isLiteral() ? flattenedOutputDims[0].getLiteral()
                                           : -1,
        "dlf16-f32 conversion fully flattened");

    IndexExpr zero = LiteralIndexExpr(0);
    create.krnl.iterateIE(loopDef, optimizedLoopDef, {zero},
        flattenedOutputDims, [&](KrnlBuilder &b, ValueRange loopInd) {
          MDBuilder create(b);
          // Manually unrolled loop, add VL offset at each iterations.
          for (int64_t u = 0; u < unrollSIMD; ++u) {
            Value baseIdx =
                create.math.add(loopInd[0], create.math.constantIndex(u * VL));
            Value baseIdxNext =
                create.math.add(baseIdx, create.math.constantIndex(VLHalf));
            if (fromF32) {
              // F32 -> DLF16
              // Load VL f32 values from the input into two vectors each
              // with VLHalf f32 values.
              Value vecF32H = create.vec.load(vecF32Type, flatInput, {baseIdx});
              Value vecF32L =
                  create.vec.load(vecF32Type, flatInput, {baseIdxNext});
              Value vecF16 = rewriter.create<ZLowConvertF32ToDLF16VectorOp>(
                  loc, vecF32H, vecF32L);
              // Store VL f16 values back to the output.
              create.vec.store(vecF16, flatOutput, {baseIdx});
            } else {
              // DLF16 -> F32
              // Load VL f16 values from the input into a register.
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
    mlir::TypeConverter &typeConverter, mlir::MLIRContext *ctx,
    bool enableParallel, bool enableCompilerStickUnstickCodeGen) {
  // Stickify and unstickify operations.
  patterns.insert<ZHighToZLowStickifiedConstantOpLowering>(typeConverter, ctx);
  patterns.insert<ZHighToZLowStickOpLowering>(
      typeConverter, ctx, enableParallel, enableCompilerStickUnstickCodeGen);
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
  patterns.insert<ZHighToZLowUnaryOpLowering<ZHighReluOp>>(typeConverter, ctx);
  patterns.insert<ZHighToZLowUnaryOpLowering<ZHighTanhOp>>(typeConverter, ctx);
  patterns.insert<ZHighToZLowUnaryOpLowering<ZHighSigmoidOp>>(
      typeConverter, ctx);
  // Neural network operations.
  patterns.insert<ZHighToZLowSoftmaxOpLowering>(typeConverter, ctx);
  patterns.insert<ZHighToZLowMeanReduce2DOpLowering>(typeConverter, ctx);
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
}

} // namespace zhigh
} // namespace onnx_mlir
