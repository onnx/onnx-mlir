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

//
// Conversion is currently limited to 2Dx2D shaped ONNX matmuls.
// ISSUE-MAKUDRYA-TODO: support all possible ONNX matmul shapes.
//
// tensors are converted to memrefs as a part of this pass, inline
// with how ONNX lowering to affine+std is currently implemented.
//
class ONNXToLinalgMatMulConverter : public ConversionPattern {

  // ISSUE-MAKUDRYA-TODO: #244 Create target description structure which can
  // be used across all passes. Use Apollo 0.5 (Artemis) value for now.
  static const int64_t kTargetDimensionDenominator = 16;

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
    auto elementType = memRefType.getElementType();
    auto memRefShape = memRefType.getShape();

    // Insert an allocation and deallocation for the result of this operation.
    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);
    if (hasAllConstantDimensions(memRefType))
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    else {
      op->emitWarning("This operation produces unsupported by "
                      "current target dynamically sized tensor.");
      return failure();
    }

    // Convert to linalg.matmul only when both arguments are 2-D matrices
    // with dimansions perfectly divisible by kTargetDimensionDenominator.
    if (!hasAllDimensionsDivisibleBy(AType, kTargetDimensionDenominator) ||
        !hasAllDimensionsDivisibleBy(BType, kTargetDimensionDenominator)) {
      op->emitWarning("This operation takes tensors with unsupported by "
                      "current target sizes.");
      return failure();
    }

    if (AShape.size() == 2 || BShape.size() == 2) {
      // Special handling for Apollo. Please refer to the link below for more
      // details.
      // https://dev.azure.com/mltools/llvm-project/_wiki/wikis/llvm-project.wiki/76/Handling-matrix-multiplication

      rewriter.create<linalg::MatmulOp>(loc, operands, alloc);
    } else {
      op->emitWarning("This operation takes tensors with unsupported by "
                      "current target shape.");
      return failure();
    }

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
    OwningRewritePatternList patterns;
    ConversionTarget target(getContext());
    target.addLegalDialect<linalg::LinalgDialect, memref::MemRefDialect,
      StandardOpsDialect, ONNXOpsDialect>();

    // Signal pass failure (for now) if there are any unconverted matrix
    // multiplications.
    // ISSUE-MAKUDRYA-TODO: cover all types of matmul
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