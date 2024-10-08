/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ KrnlNone.cpp - Lower KrnlRoundEvenOp
//-------------------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the KrnlRoundEvenOp operator.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "krnl_to_llvm"

using namespace mlir;

namespace onnx_mlir {
namespace krnl {

class KrnlRoundEvenOpLowering : public ConversionPattern {
public:
  explicit KrnlRoundEvenOpLowering(
      LLVMTypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(
            typeConverter, KrnlRoundEvenOp::getOperationName(), 1, context) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    KrnlRoundEvenOp::Adaptor operandAdaptor(operands);
    Value input = operandAdaptor.getIn();

    // Scalar or Vector?
    Type inputType = input.getType();
    Type inputElemType = getElementTypeOrSelf(inputType);
    assert(mlir::isa<FloatType>(inputElemType) && "expected float");
    int64_t inputBitWidth = inputElemType.getIntOrFloatBitWidth();
    assert(inputBitWidth == 32 && "expected 32bit float");
    VectorType inputVecType = mlir::dyn_cast<VectorType>(inputType);

    // Common between scalar and vector
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);
    Type i32Ty = rewriter.getI32Type();
    Type f32Ty = rewriter.getF32Type();

    if (inputVecType) {
      // Vector of 4 elements.
      Type vecTypeI32 = LLVM::getFixedVectorType(i32Ty, 4);
      Type vecTypeF32 = LLVM::getFixedVectorType(f32Ty, 4);
      // Use integer as container for inputs.
      Value inputVecI32 = create.llvm.bitcast(vecTypeI32, input);
      SmallVector<Value> asmVals{inputVecI32};
      // SIMD ASM op
      const char *asmStr = "VFISB $0,$1,0,4";
      const char *asmConstraints = "=v,v";
      Value outVecI32 =
          rewriter
              .create<LLVM::InlineAsmOp>(loc, vecTypeI32,
                  /*operands=*/asmVals,
                  /*asm_string=*/asmStr,
                  /*constraints=*/asmConstraints, /*has_side_effects=*/false,
                  /*is_align_stack=*/false,
                  /*asm_dialect=*/LLVM::AsmDialectAttr(),
                  /*operand_attrs=*/ArrayAttr())
              .getResult(0);
      // Cast output back to float.
      Value outVecF32 = create.llvm.bitcast(vecTypeF32, outVecI32);
      rewriter.replaceOp(op, {outVecF32});
      return success();
    } else {
      // Scalar types.
      Type typeI32 = rewriter.getI32Type();
      Type typeF32 = rewriter.getF32Type();
      // Use integer as container for inputs.
      Value inputI32 = create.llvm.bitcast(typeI32, input);
      SmallVector<Value> asmVals{inputI32};
      // SIMD ASM op
      const char *asmStr = "FIEBR $0,$1,4";
      const char *asmConstraints = "=f,f";
      Value outI32 =
          rewriter
              .create<LLVM::InlineAsmOp>(loc, typeI32,
                  /*operands=*/asmVals,
                  /*asm_string=*/asmStr,
                  /*constraints=*/asmConstraints, /*has_side_effects=*/false,
                  /*is_align_stack=*/false,
                  /*asm_dialect=*/LLVM::AsmDialectAttr(),
                  /*operand_attrs=*/ArrayAttr())
              .getResult(0);
      // Cast output back to float.
      Value outF32 = create.llvm.bitcast(typeF32, outI32);
      rewriter.replaceOp(op, {outF32});
      return success();
    }
    llvm_unreachable("not supported");
  }
};

void populateLoweringKrnlRoundEvenOpPattern(LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<KrnlRoundEvenOpLowering>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
