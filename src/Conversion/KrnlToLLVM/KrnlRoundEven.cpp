/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ KrnlRoundEven.cpp - Lower KrnlRoundEvenOp ---------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the KrnlRoundEvenOp operator.
//
// Currently limited to fp32 integers, instructions supports other data types.
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
    assert(VectorMachineSupport::requireCustomASM(
               GenericOps::roundEvenGop, inputElemType) &&
           "expected custom requirement");
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
      // SIMD ASM round to nearest even (M5=4) op
      // Note the spaces are required by the z/OS assembler.
      const char *asmStr = "       VFISB $0,$1,0,4         \n\t";
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
      Type typeF32 = rewriter.getF32Type();
      SmallVector<Value> asmVals{input};
      // Scalar ASM round to the nearest even (M3=4) op.
      // Note the spaces are required by the z/OS assembler.
      const char *asmStr = "       FIEBR $0,4,$1         \n\t";
      const char *asmConstraints = "=f,f";
      Value outF32 =
          rewriter
              .create<LLVM::InlineAsmOp>(loc, typeF32,
                  /*operands=*/asmVals,
                  /*asm_string=*/asmStr,
                  /*constraints=*/asmConstraints, /*has_side_effects=*/false,
                  /*is_align_stack=*/false,
                  /*asm_dialect=*/LLVM::AsmDialectAttr(),
                  /*operand_attrs=*/ArrayAttr())
              .getResult(0);
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
