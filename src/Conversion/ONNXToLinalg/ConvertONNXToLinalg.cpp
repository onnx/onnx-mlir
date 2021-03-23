//===- ConvertONNXToLinalg.cpp - ONNX to Linalg conversion ----------------===//
//
// Convert Onnx.matmul operators to Linalg dialect. This will prevent
// Onnx.matmul lowering to affine loops and allow linalg.matmul conversion
// to NEPAL dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

#define DEBUG_TYPE "convert-onnx-to-linalg"

//
// tensors are converted to memrefs as a part of this pass, inline
// with how ONNX lowering to affine+std is currently implemented.
//
class ONNXToLinalgMatMulConverter : public ConversionPattern {

  // ISSUE-MAKUDRYA-TODO: #244 Create target description structure which can
  // be used across all passes. Use Apollo 0.5 (Artemis) value for now.
  static const int64_t kTargetDimensionDenominator = 16;

  // Each matrix multiply utilizes the innermost two dimensions. Given a
  // MemRefType, this functions creates the loops to iterate over the N-2 outer
  // dimensions. It returns the list of resulting IVs that can be used to create
  // sub-views of the matmul arguments.
  //
  // This function also sets the instruction insertion point into the body of
  // the innermost loop.
  void createLoops(MemRefType type, Value input,
      ConversionPatternRewriter &rewriter, Location loc,
      SmallVector<Value> &resultIVs) const {
    assert(resultIVs.size() == 0);
    std::vector<Value> originalLoops;

    // the body of the loop is at a 2D granularity
    auto rank = type.getRank() - 2;
    assert(rank > 0);
    defineLoops(rewriter, loc, originalLoops, rank);

    std::vector<Value> loops;
    for (int i = 0; i < rank; ++i) {
      loops.push_back(originalLoops[i]);
    }

    KrnlIterateOperandPack pack(rewriter, loops);
    auto shape = type.getShape();
    for (int i = 0; i < rank; ++i) {
      pack.pushConstantBound(0);
      if (shape[i] < 0) {
        pack.pushOperandBound(
            rewriter.create<memref::DimOp>(loc, input, i).getResult());
      } else {
        pack.pushConstantBound(shape[i]);
      }
    }

    auto outerIterateOp = rewriter.create<KrnlIterateOp>(loc, pack);

    // Insert instructions inside the outer loop.
    Block &outerIterationBlock = outerIterateOp.bodyRegion().front();
    rewriter.setInsertionPointToStart(&outerIterationBlock);

    // Return the induction variables.
    for (auto arg : outerIterationBlock.getArguments()) {
      resultIVs.push_back(arg);
    }
  }

  // Take a memref and get a subview into its last two dimensions. E.g.
  //        %opA : memref<5x10x3072x512xbf16>
  //           =>  subview %opA[%iv1, %iv2, 0, 0] [1, 1, 3072, 512] [1, 1, 1, 1]
  //                : memref<3072x512xbf16, #map1>
  //
  // This gets interesting when the memref is simulataneously being broadcast
  // (both to a higher rank and for dimensions of size 1). We follow numpy
  // semantics, so this entails prepending the matrix with the 1s when the
  // matrix needs to be broadcast to a higher rank, and using the 0th element
  // repeatedly when broadcasting across a dimension.
  Value getOperandSubView(MemRefType type, Value operand,
      SmallVector<Value> &ivs, ConversionPatternRewriter &rewriter,
      Location loc) const {

    auto iterationDims = type.getRank() - 2;
    if (iterationDims < 1)
      return operand;

    SmallVector<int64_t> offsets(type.getRank(), 0);
    SmallVector<Value> iterationIvs;
    auto shape = type.getShape();
    auto dimsToPad = ivs.size() - iterationDims;
    auto ivItr = ivs.begin() + dimsToPad;

    // skip the last two indices, they're always 0
    for (int i = 0; i < type.getRank() - 2; i++) {
      // we broadcast the 0th value when the dimension is 1
      if (shape[i] == 1) {
        offsets[i] = 0;
      } else {
        offsets[i] = (ShapedType::kDynamicStrideOrOffset);
        iterationIvs.push_back(*ivItr);
      }
      ivItr++;
    }
    auto offsetsAttr = rewriter.getI64ArrayAttr(offsets);

    SmallVector<int64_t> sizes(iterationDims, 1);
    auto resultShape = shape.take_back(2);
    sizes.append(resultShape.begin(), resultShape.end());
    auto sizesAttr = rewriter.getI64ArrayAttr(sizes);

    SmallVector<int64_t> strides(type.getRank(), 1);
    auto stridesAttr = rewriter.getI64ArrayAttr(strides);

    // Subtensor ops don't play well with downstream passes. Normalize
    // everything to be memrefs.
    if (operand.getType().isa<RankedTensorType>()) {
      operand = rewriter.create<KrnlDummyCastOp>(loc, operand, type);
    }

    assert(operand.getType().isa<MemRefType>());
    SmallVector<Value> empty;
    auto resType =
        memref::SubViewOp::inferRankReducedResultType(2, type, offsets, sizes, strides);
    return rewriter.create<memref::SubViewOp>(loc, resType, operand,
        llvm::makeArrayRef(iterationIvs), llvm::makeArrayRef(empty),
        llvm::makeArrayRef(empty), offsetsAttr, sizesAttr, stridesAttr);
  }

  // Verify the last two dimensions of each type are divisible by
  // kTargetDimensionDenominator and that the N-2 outermost dimensions match
  // numpy broadcast semantics for two arguments: so their dimensions match, or
  // one of them has dimension of size 1.
  bool verifyDims(MemRefType typeA, MemRefType typeB) const {
    auto shapeA = typeA.getShape();
    auto shapeB = typeB.getShape();
    assert(shapeA.size() > 1 && shapeB.size() > 1);

    // if the shapes don't match in rank, start out at an offset in the larger
    // shape so that the two matrices line up along the rightmost dimension.
    // E.g. if A is 2D and B is 4D this is how the dimensions should line up
    // when verifying the two arguments match:
    //                         1 2 3 4
    //                      A:     |-|
    //                      B: |-----|
    int64_t a = 0, b = 0;
    if (shapeA.size() > shapeB.size()) {
      a = shapeB.size() - shapeA.size();
    } else if (shapeA.size() < shapeB.size()) {
      b = shapeA.size() - shapeB.size();
    }

    for (; a < shapeA.size() - 2 && b < shapeB.size() - 2; a++, b++) {
      if (shapeA[a] != 1 && shapeB[b] != 1 && shapeA[a] != shapeB[b]) {
        return false;
      }
    }

    // Convert to linalg.matmul only when both arguments are matrices
    // with the last two dimensions perfectly divisible by
    // kTargetDimensionDenominator.
    auto divisibleByTargetDim = [](int64_t dim) {
      return (dim >= 0) &&
             (dim % ONNXToLinalgMatMulConverter::kTargetDimensionDenominator ==
                 0);
    };

    return llvm::all_of(shapeA.take_back(2), divisibleByTargetDim) &&
           llvm::all_of(shapeB.take_back(2), divisibleByTargetDim);
  }

public:
  ONNXToLinalgMatMulConverter(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXMatMulOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    ONNXMatMulOpAdaptor operandAdaptor(operands);
    auto AType = convertToMemRefType(operandAdaptor.A().getType());
    auto BType = convertToMemRefType(operandAdaptor.B().getType());
    auto AShape = AType.getShape();
    auto BShape = BType.getShape();

    // Result type
    auto memRefType = convertToMemRefType(*op->result_type_begin());

    // Insert an allocation and deallocation for the result of this operation.
    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);
    if (hasAllConstantDimensions(memRefType)) {
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    } else {
      op->emitWarning("This operation produces unsupported by "
                      "current target dynamically sized tensor.");
      return failure();
    }

    if (!verifyDims(AType, BType)) {
      op->emitWarning("This operation takes tensors with unsupported by "
                      "current target sizes.");
      return failure();
    }

    auto cachedInsertionPt = rewriter.saveInsertionPoint();
    auto operandA = operandAdaptor.A();
    auto operandB = operandAdaptor.B();
    auto result = alloc;

    // When performing matmul between two arguments with more than 2 dimensions,
    // treat them like a stack of 2D matrices and generate loops over the
    // outermost N-2 dimensions.
    if (AShape.size() > 2 || BShape.size() > 2) {
      SmallVector<Value> ivs;
      if (AShape.size() > BShape.size()) {
        createLoops(AType, operandA, rewriter, loc, ivs);
      } else {
        createLoops(BType, operandB, rewriter, loc, ivs);
      }

      operandA = getOperandSubView(AType, operandA, ivs, rewriter, loc);
      operandB = getOperandSubView(BType, operandB, ivs, rewriter, loc);
      result = getOperandSubView(memRefType, alloc, ivs, rewriter, loc);
    }

    SmallVector<Value> ops{operandA, operandB};
    rewriter.create<linalg::MatmulOp>(loc, ops, result);
    rewriter.restoreInsertionPoint(cachedInsertionPt);
    rewriter.replaceOp(op, alloc);

    return success();
  }
};

namespace {
//
//  Function pass that performs Onnx.matmul operators conversion to Linalg
//  dialect.
//
class ConvertONNXToLinalgPass
    : public PassWrapper<ConvertONNXToLinalgPass, FunctionPass> {
public:
  void runOnFunction() override {
    OwningRewritePatternList patterns(&getContext());
    ConversionTarget target(getContext());
    target.addLegalDialect<linalg::LinalgDialect, memref::MemRefDialect,
        StandardOpsDialect, KrnlOpsDialect, ONNXOpsDialect>();

    // Signal pass failure (for now) if there are any unconverted matrix
    // multiplications.
    target.addIllegalOp<ONNXMatMulOp>();

    auto func = getFunction();
    patterns.insert<ONNXToLinalgMatMulConverter>(func.getContext());
    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createConvertONNXToLinalgPass() {
  return std::make_unique<ConvertONNXToLinalgPass>();
}