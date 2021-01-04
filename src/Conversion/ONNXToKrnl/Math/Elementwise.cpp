//===---------------- Elementwise.cpp - Elementwise Ops -------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers ONNX element-wise operators to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXShapeHelper.hpp"

using namespace mlir;

template <>
struct ScalarOp<ONNXTanhOp> {
  using FOp = TanhOp;
  using IOp = TanhOp; // not use
};

template <>
struct ScalarOp<ONNXAddOp> {
  using FOp = AddFOp;
  using IOp = AddIOp;
};

template <>
struct ScalarOp<ONNXMulOp> {
  using FOp = MulFOp;
  using IOp = MulIOp;
};

template <>
struct ScalarOp<ONNXDivOp> {
  using FOp = DivFOp;
  using IOp = SignedDivIOp;
};

template <>
struct ScalarOp<ONNXSubOp> {
  using FOp = SubFOp;
  using IOp = SubIOp;
};

template <>
struct ScalarOp<ONNXAndOp> {
  using FOp = AndOp; // not use
  using IOp = AndOp;
};

template <>
struct ScalarOp<ONNXOrOp> {
  using FOp = OrOp; // not use
  using IOp = OrOp;
};

template <>
struct ScalarOp<ONNXXorOp> {
  using FOp = XOrOp; // not use
  using IOp = XOrOp;
};

template <>
struct ScalarOp<ONNXExpOp> {
  using FOp = ExpOp;
  using IOp = ExpOp; // not use
};

template <>
struct ScalarOp<ONNXSumOp> {
  using FOp = AddFOp;
  using IOp = AddIOp;
};

template <>
struct ScalarOp<ONNXCosOp> {
  using FOp = CosOp;
  using IOp = CosOp; // not use
};

template <>
struct ScalarOp<ONNXLogOp> {
  using FOp = LogOp;
  using IOp = LogOp; // not use
};

template <>
struct ScalarOp<ONNXSqrtOp> {
  using FOp = SqrtOp;
  using IOp = SqrtOp; // not use
};

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXCastOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXCastOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  ONNXCastOp castOp = llvm::dyn_cast<ONNXCastOp>(op);
  auto mlirtype = castOp.toAttr().getValue();
  Value operand = scalarOperands[0];
  auto origtype = operand.getType();

  // check output type is the same as expected output type
  if (elementType != mlirtype)
    llvm_unreachable("output type different from expected output type");

  // if same input and output type, return input
  if (origtype == elementType)
    return operand;

  if (origtype.isa<FloatType>()) {
    // cast from floating-point type to integer type
    if (elementType.isa<IntegerType>())
      return rewriter.create<FPToSIOp>(loc, elementType, operand);
    // cast from floating-point type to other floating-point type
    else if (elementType.isa<FloatType>()) {
      // cast from floating-point to wider floating-point
      if (origtype.getIntOrFloatBitWidth() <
          elementType.getIntOrFloatBitWidth())
        return rewriter.create<FPExtOp>(loc, elementType, operand);
      // cast from floating-point to narrower floating-point
      else
        return rewriter.create<FPTruncOp>(loc, elementType, operand);
    }
  } else if (origtype.isa<IntegerType>()) {
    // cast from integer type to floating-point type
    if (elementType.isa<FloatType>())
      return rewriter.create<SIToFPOp>(loc, elementType, operand);
    else if (elementType.isa<IntegerType>())
      // cast from integer to wider integer
      if (origtype.getIntOrFloatBitWidth() <
          elementType.getIntOrFloatBitWidth())
        return rewriter.create<SignExtendIOp>(loc, operand, elementType);
      // cast from integer to narrower integer
      else
        return rewriter.create<TruncateIOp>(loc, operand, elementType);
    else
      llvm_unreachable("unsupported element type");
  }
  llvm_unreachable("unsupported element type");
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXSinhOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXSinhOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  // ONNXSinhOp(%X) = DivFOp(SubFOp(ExpOp(%X), ExpOp(NegFOp(%X))),
  //                         ConstantOp 2)
  Value operand = scalarOperands[0];

  auto zero = emitConstantOp(rewriter, loc, elementType, 0);
  auto two = emitConstantOp(rewriter, loc, elementType, 2);
  auto neg = rewriter.create<SubFOp>(loc, zero, operand);
  auto exp = rewriter.create<ExpOp>(loc, operand);
  auto negExp = rewriter.create<ExpOp>(loc, neg);
  auto result = rewriter.create<DivFOp>(
      loc, rewriter.create<SubFOp>(loc, exp, negExp), two);

  return result;
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXCoshOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXCoshOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  // ONNXCoshOp(%X) = DivFOp(AddFOp(ExpOp(%X), ExpOp(NegFOp(%X))),
  //                         ConstantOp 2)
  Value operand = scalarOperands[0];

  auto zero = emitConstantOp(rewriter, loc, elementType, 0);
  auto two = emitConstantOp(rewriter, loc, elementType, 2);
  auto neg = rewriter.create<SubFOp>(loc, zero, operand);
  auto exp = rewriter.create<ExpOp>(loc, operand);
  auto negExp = rewriter.create<ExpOp>(loc, neg);
  auto result = rewriter.create<DivFOp>(
      loc, rewriter.create<AddFOp>(loc, exp, negExp), two);

  return result;
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXSigmoidOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXSigmoidOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  // ONNXSigmoidOp(%X) = DivFOp(ConstantOp 1,
  //                            AddFOp(ConstantOp 1, ExpOp(NegFOp(%X))))
  Value operand = scalarOperands[0];

  auto zero = emitConstantOp(rewriter, loc, elementType, 0);
  auto one = emitConstantOp(rewriter, loc, elementType, 1);
  auto neg = rewriter.create<SubFOp>(loc, zero, operand);
  auto negExp = rewriter.create<ExpOp>(loc, neg);
  auto result = rewriter.create<DivFOp>(
      loc, one, rewriter.create<AddFOp>(loc, one, negExp));

  return result;
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXHardSigmoidOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXHardSigmoidOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  // %Y = AddFOp(MulFOp(alpha, %X), beta)
  // %Z = SelectOp(CmpFOp(OGT, %Y, Constant 0),
  //               %Y,
  //               Constant 0)
  // ONNXHardSigmoidOp(%X) = SelectOp(CmpFOp(OLT, %Z, Constant 1),
  //                                  %Z,
  //                                  Constant 1)
  Value operand = scalarOperands[0];
  auto alphaAttribute = FloatAttr::get(rewriter.getF32Type(),
      llvm::dyn_cast<ONNXHardSigmoidOp>(op).alpha().convertToFloat());
  auto betaAttribute = FloatAttr::get(rewriter.getF32Type(),
      llvm::dyn_cast<ONNXHardSigmoidOp>(op).beta().convertToFloat());

  auto zero = emitConstantOp(rewriter, loc, elementType, 0);
  auto one = emitConstantOp(rewriter, loc, elementType, 1);
  auto alpha = rewriter.create<ConstantOp>(loc, alphaAttribute);
  auto beta = rewriter.create<ConstantOp>(loc, betaAttribute);

  auto add = rewriter.create<AddFOp>(
      loc, rewriter.create<MulFOp>(loc, alpha, operand), beta);
  auto maxPredicate =
      rewriter.create<CmpFOp>(loc, CmpFPredicate::OGT, add, zero);
  auto max = rewriter.create<SelectOp>(loc, maxPredicate, add, zero);
  auto minPredicate =
      rewriter.create<CmpFOp>(loc, CmpFPredicate::OLT, max, one);
  auto result = rewriter.create<SelectOp>(loc, minPredicate, max, one);

  return result;
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXEluOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXEluOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  // ONNXEluOp(%X) = SelectOp(CmpFOp(OLT, %X, ConstantOp 0),
  //                          MulFOp(alpha, SubFOp(ExpOp(%X), 1)),
  //                          %X)
  Value operand = scalarOperands[0];

  auto alphaAttribute = FloatAttr::get(rewriter.getF32Type(),
      llvm::dyn_cast<ONNXEluOp>(op).alpha().convertToFloat());
  auto zero = emitConstantOp(rewriter, loc, elementType, 0);
  auto one = emitConstantOp(rewriter, loc, elementType, 1);
  auto alpha = rewriter.create<ConstantOp>(loc, alphaAttribute);
  auto exp = rewriter.create<ExpOp>(loc, operand);
  auto lessThanZero =
      rewriter.create<CmpFOp>(loc, CmpFPredicate::OLT, operand, zero);
  auto result = rewriter.create<SelectOp>(loc, lessThanZero,
      rewriter.create<MulFOp>(
          loc, alpha, rewriter.create<SubFOp>(loc, exp, one)),
      operand);

  return result;
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXReluOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXReluOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  // ONNXReluOp(%X) = SelectOp(CmpFOp(OLT, %X, ConstantOp 0),
  //                           ConstantOp 0,
  //                           %X)
  Value operand = scalarOperands[0];

  auto zero = emitConstantOp(rewriter, loc, elementType, 0);
  auto lessThanZero =
      rewriter.create<CmpFOp>(loc, CmpFPredicate::OLT, operand, zero);
  auto result = rewriter.create<SelectOp>(loc, lessThanZero, zero, operand);

  return result;
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXLeakyReluOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXLeakyReluOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  // ONNXLeakyReluOp(%X) = SelectOp(CmpFOp(OLT, %X, ConstantOp 0),
  //                                MulFOp(alpha, %X),
  //                                %X)
  Value operand = scalarOperands[0];

  auto alphaAttribute = FloatAttr::get(rewriter.getF32Type(),
      llvm::dyn_cast<ONNXLeakyReluOp>(op).alpha().convertToFloat());
  auto zero = emitConstantOp(rewriter, loc, elementType, 0);
  auto alpha = rewriter.create<ConstantOp>(loc, alphaAttribute);
  auto lessThanZero =
      rewriter.create<CmpFOp>(loc, CmpFPredicate::OLT, operand, zero);
  auto result = rewriter.create<SelectOp>(
      loc, lessThanZero, rewriter.create<MulFOp>(loc, alpha, operand), operand);

  return result;
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXSeluOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXSeluOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  // ONNXSeluOp(%X) = SelectOp(CmpFOp(OGT, %X, ConstantOp 0),
  //                           MulFOp(gamma, %X),
  //                           MulFOp(gamma,
  //                                  SubFOp(MulFOp(alpha, ExpOp(%X)),
  //                                         alpha)))
  Value operand = scalarOperands[0];
  auto alphaAttribute = FloatAttr::get(rewriter.getF32Type(),
      llvm::dyn_cast<ONNXSeluOp>(op).alpha().convertToFloat());
  auto gammaAttribute = FloatAttr::get(rewriter.getF32Type(),
      llvm::dyn_cast<ONNXSeluOp>(op).gamma().convertToFloat());

  auto zero = emitConstantOp(rewriter, loc, elementType, 0);
  auto alpha = rewriter.create<ConstantOp>(loc, alphaAttribute);
  auto gamma = rewriter.create<ConstantOp>(loc, gammaAttribute);
  auto exp = rewriter.create<ExpOp>(loc, operand);
  auto greaterThanZero =
      rewriter.create<CmpFOp>(loc, CmpFPredicate::OGT, operand, zero);
  auto select = rewriter.create<SelectOp>(loc, greaterThanZero, operand,
      rewriter.create<SubFOp>(
          loc, rewriter.create<MulFOp>(loc, alpha, exp), alpha));
  auto result = rewriter.create<MulFOp>(loc, gamma, select);

  return result;
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXReciprocalOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXReciprocalOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  // ONNXReciprocalOp(%X) = DivFOp(ConstantOp 1, %X)
  Value operand = scalarOperands[0];
  auto one = emitConstantOp(rewriter, loc, elementType, 1);
  auto result = rewriter.create<DivFOp>(loc, one, operand);

  return result;
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXSoftplusOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXSoftplusOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  // ONNXSoftplusOp(%X) = LogOp(AddFOp(ExpOp(%X), ConstantOp 1))
  Value operand = scalarOperands[0];

  auto exp = rewriter.create<ExpOp>(loc, operand);
  auto one = emitConstantOp(rewriter, loc, elementType, 1);
  auto add = rewriter.create<AddFOp>(loc, exp, one);
  auto result = rewriter.create<LogOp>(loc, add);

  return result;
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXSoftsignOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXSoftsignOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  // ONNXSoftsignOp(%X) = DivFOp(ConstantOp 1, %X)
  Value operand = scalarOperands[0];

  auto abs = rewriter.create<AbsFOp>(loc, operand);
  auto one = emitConstantOp(rewriter, loc, elementType, 1);
  auto add = rewriter.create<AddFOp>(loc, abs, one);
  auto result = rewriter.create<DivFOp>(loc, operand, add);

  return result;
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXSignOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXSignOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  Value operand = scalarOperands[0];
  // TODO: unsigned int should be supported separately?
  if (elementType.isa<IntegerType>()) {
    // %Y = SelectOP(CmpIOp(GT, %X, ConstantOp 0),
    //               ConstantOp 1,
    //               COnstantOp -1)
    // ONNXSignOp(%X) = SelectOP(CmpIOp(EQ, %X, ConstantOp 0),
    //                           ConstantOp 0,
    //                           %Y)
    auto zero = emitConstantOp(rewriter, loc, elementType, 0);
    auto one = emitConstantOp(rewriter, loc, elementType, 1);
    auto minusOne = emitConstantOp(rewriter, loc, elementType, -1);
    auto plusPredicate =
        rewriter.create<CmpIOp>(loc, CmpIPredicate::sgt, operand, zero);
    auto plusSelect =
        rewriter.create<SelectOp>(loc, plusPredicate, one, minusOne);
    auto zeroPredicate =
        rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, operand, zero);
    auto result =
        rewriter.create<SelectOp>(loc, zeroPredicate, zero, plusSelect);
    return result;
  } else if (elementType.isa<FloatType>()) {
    // %Y = SelectOP(CmpFOp(OGT, %X, ConstantOp 0),
    //               ConstantOp 1,
    //               ConstantOp -1)
    // ONNXSignOp(%X) = SelectOP(CmpFOp(OEQ, %X, ConstantOp 0),
    //                           ConstantOp 0,
    //                           %Y)
    auto zero = emitConstantOp(rewriter, loc, elementType, 0);
    auto one = emitConstantOp(rewriter, loc, elementType, 1);
    auto minusOne = emitConstantOp(rewriter, loc, elementType, -1);
    auto plusPredicate =
        rewriter.create<CmpFOp>(loc, CmpFPredicate::OGT, operand, zero);
    auto plusSelect =
        rewriter.create<SelectOp>(loc, plusPredicate, one, minusOne);
    auto zeroPredicate =
        rewriter.create<CmpFOp>(loc, CmpFPredicate::OEQ, operand, zero);
    auto result =
        rewriter.create<SelectOp>(loc, zeroPredicate, zero, plusSelect);
    return result;
  } else {
    llvm_unreachable("unsupported element type");
  }
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXMaxOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXMaxOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  // ONNXMaxOp(%X, %Y) = SelectOp(CmpFOp(OGT, %X, %Y),
  //                              %X,
  //                              %Y)
  Value lhs = scalarOperands[0];
  Value rhs = scalarOperands[1];
  auto max = rewriter.create<CmpFOp>(loc, CmpFPredicate::OGT, lhs, rhs);
  auto result = rewriter.create<SelectOp>(loc, max, lhs, rhs);
  return result;
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXMinOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXMinOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  // ONNXMinOp(%X, %Y) = SelectOp(CmpFOp(OLT, %X, %Y),
  //                              %X,
  //                              %Y)
  Value lhs = scalarOperands[0];
  Value rhs = scalarOperands[1];
  auto min = rewriter.create<CmpFOp>(loc, CmpFPredicate::OLT, lhs, rhs);
  auto result = rewriter.create<SelectOp>(loc, min, lhs, rhs);
  return result;
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXAbsOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXAbsOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  Value operand = scalarOperands[0];

  if (elementType.isa<FloatType>()) {
    return rewriter.create<AbsFOp>(loc, operand);
  } else if (elementType.isa<IntegerType>()) {
    auto zero = emitConstantOp(rewriter, loc, elementType, 0);
    auto lessThanZero =
        rewriter.create<CmpIOp>(loc, CmpIPredicate::slt, operand, zero);
    auto negativeOperand = rewriter.create<SubIOp>(loc, zero, operand);
    return rewriter.create<SelectOp>(
        loc, lessThanZero, negativeOperand, operand);
  } else {
    llvm_unreachable("unsupported element type");
  }
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXNegOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXNegOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  Value operand = scalarOperands[0];

  if (elementType.isa<FloatType>()) {
    return rewriter.create<mlir::NegFOp>(loc, operand);
  } else if (elementType.isa<IntegerType>()) {
    auto zero = emitConstantOp(rewriter, loc, elementType, 0);
    return rewriter.create<mlir::SubIOp>(loc, zero, operand); // 0 - X = -X
  } else {
    llvm_unreachable("unsupported element type");
  }
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXLessOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXLessOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  Value lhs = scalarOperands[0];
  Value rhs = scalarOperands[1];

  Type inputType = lhs.getType();
  if (inputType.isa<FloatType>()) {
    return rewriter.create<CmpFOp>(loc, CmpFPredicate::OLT, lhs, rhs);
  } else if (inputType.isa<IntegerType>()) {
    return rewriter.create<CmpIOp>(loc, CmpIPredicate::slt, lhs, rhs);
  } else {
    llvm_unreachable("unsupported element type");
  }
}

// Element-wise unary ops lowering to Krnl dialect.
//===----------------------------------------------------------------------===//
template <typename ElementwiseUnaryOp>
struct ONNXElementwiseUnaryOpLowering : public ConversionPattern {
  ONNXElementwiseUnaryOpLowering(MLIRContext *ctx)
      : ConversionPattern(ElementwiseUnaryOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    auto X = operands[0];

    // Insert an allocation and deallocation for the result of this operation.
    auto memRefType = convertToMemRefType(*op->result_type_begin());

    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);

    if (hasAllConstantDimensions(memRefType))
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    else
      alloc =
          insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc, X);

    SmallVector<Value, 4> loopIVs;
    // Only create krnl.iterate if one of the operands is not scalar tensor.
    if (!hasAllScalarValues(operands)) {
      // Create iterateOp & get block within iterate op.
      BuildKrnlLoop loops(rewriter, loc, memRefType.getRank());
      loops.createDefineAndIterateOp(X);
      Block *iterationBlock = loops.getIterateBlock();

      // Insert instructions inside the KernelIterateOp body.
      rewriter.setInsertionPointToStart(iterationBlock);

      // Handle the operation:
      for (auto arg : iterationBlock->getArguments())
        loopIVs.push_back(arg);
    }

    auto loadedVal = rewriter.create<AffineLoadOp>(loc, X, loopIVs);
    auto loweredOpResult = emitScalarOpFor<ElementwiseUnaryOp>(
        rewriter, loc, op, memRefType.getElementType(), {loadedVal});
    // Store result in the resulting array.
    rewriter.create<AffineStoreOp>(loc, loweredOpResult, alloc, loopIVs);

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

// Element-wise binary ops lowering to Krnl dialect.
// This template can be used for binary ops that return a result whose type is
// different from the input type.
//===----------------------------------------------------------------------===//
template <typename ElementwiseBinaryOp>
struct ONNXElementwiseBinaryOpLowering : public ConversionPattern {
  ONNXElementwiseBinaryOpLowering(MLIRContext *ctx)
      : ConversionPattern(ElementwiseBinaryOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    auto numArgs = op->getNumOperands();
    auto outputMemRefType = convertToMemRefType(*op->result_type_begin());
    auto outputElementType = outputMemRefType.getElementType();
    auto outputRank = outputMemRefType.getRank();

    // Shape helper.
    ONNXOpBroadcastedShapeHelper shapeHelper(&rewriter, loc);
    assert(succeeded(shapeHelper.Compute(operands)));
    IndexExprContext outerContext(shapeHelper.context);

    // Insert an allocation and deallocation for the result of this operation.
    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, outputMemRefType, loc, shapeHelper.outputDims);

    // Emit main computation.
    SmallVector<IndexExpr, 4> outputAccessExprs;
    // Only create krnl.iterate if one of the operands is not scalar tensor.
    if (!hasAllScalarValues(operands)) {
      // Create iterateOp & get block within iterate op.
      BuildKrnlLoop loops(rewriter, loc, outputRank);
      loops.createDefineAndIterateOp(alloc);
      Block *iterationBlock = loops.getIterateBlock();
      // Insert instructions inside the KernelIterateOp body.
      rewriter.setInsertionPointToStart(iterationBlock);
      // Handle the operation:
      for (auto arg : iterationBlock->getArguments())
        outputAccessExprs.emplace_back(
            outerContext.createLoopInductionIndex(arg));
    }

    // Load the first value.
    SmallVector<IndexExpr, 4> lhsAccessExprs;
    shapeHelper.GetAccessExprs(
        outerContext, operands[0], 0, outputAccessExprs, lhsAccessExprs);
    Value lhs = outerContext.createLoadOp(operands[0], lhsAccessExprs);

    // Load the sencond value.
    SmallVector<IndexExpr, 4> rhsAccessExprs;
    shapeHelper.GetAccessExprs(
        outerContext, operands[1], 1, outputAccessExprs, rhsAccessExprs);
    Value rhs = outerContext.createLoadOp(operands[1], rhsAccessExprs);

    // Apply the element-wise function.
    Value result = emitScalarOpFor<ElementwiseBinaryOp>(
        rewriter, loc, op, outputElementType, {lhs, rhs});

    // Store result in the resulting array.
    outerContext.createStoreOp(result, alloc, outputAccessExprs);

    rewriter.replaceOp(op, alloc);

    return success();
  }
};

// Element-wise variadic ops lowering to Krnl dialect.
//===----------------------------------------------------------------------===//
template <typename ElementwiseVariadicOp>
struct ONNXElementwiseVariadicOpLowering : public ConversionPattern {
  ONNXElementwiseVariadicOpLowering(MLIRContext *ctx)
      : ConversionPattern(ElementwiseVariadicOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    auto numArgs = op->getNumOperands();
    auto outputMemRefType = convertToMemRefType(*op->result_type_begin());
    auto outputElementType = outputMemRefType.getElementType();
    auto outputRank = outputMemRefType.getRank();

    // Shape helper.
    ONNXOpBroadcastedShapeHelper shapeHelper(&rewriter, loc);
    assert(succeeded(shapeHelper.Compute(operands)));
    IndexExprContext outerContext(shapeHelper.context);

    // Insert an allocation and deallocation for the result of this operation.
    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, outputMemRefType, loc, shapeHelper.outputDims);

    // Emit main computation.
    SmallVector<IndexExpr, 4> outputAccessExprs;
    // Only create krnl.iterate if one of the operands is not scalar tensor.
    if (!hasAllScalarValues(operands)) {
      // Create iterateOp & get block within iterate op.
      BuildKrnlLoop loops(rewriter, loc, outputRank);
      loops.createDefineAndIterateOp(alloc);
      Block *iterationBlock = loops.getIterateBlock();
      // Insert instructions inside the KernelIterateOp body.
      rewriter.setInsertionPointToStart(iterationBlock);
      // Handle the operation:
      for (auto arg : iterationBlock->getArguments())
        outputAccessExprs.emplace_back(
            outerContext.createLoopInductionIndex(arg));
    }

    // Fold over operands for each of their scalar values.
    // Obtain the first operand.
    SmallVector<IndexExpr, 4> oprdAccessExprs;
    shapeHelper.GetAccessExprs(
        outerContext, operands[0], 0, outputAccessExprs, oprdAccessExprs);
    Value accumulated = outerContext.createLoadOp(operands[0], oprdAccessExprs);

    // Iterate over the remaining operands.
    for (unsigned i = 1; i < numArgs; i++) {
      // Obtain the next operand.
      SmallVector<IndexExpr, 4> oprdAccessExprs;
      shapeHelper.GetAccessExprs(
          outerContext, operands[i], i, outputAccessExprs, oprdAccessExprs);
      Value next = outerContext.createLoadOp(operands[i], oprdAccessExprs);
      // Fold.
      accumulated = emitScalarOpFor<ElementwiseVariadicOp>(
          rewriter, loc, op, outputElementType, {accumulated, next});
    }

    // Store result in the resulting array.
    outerContext.createStoreOp(accumulated, alloc, outputAccessExprs);

    rewriter.replaceOp(op, alloc);

    return success();
  }
};

void populateLoweringONNXElementwiseOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXElementwiseUnaryOpLowering<mlir::ONNXAbsOp>,
      ONNXElementwiseVariadicOpLowering<mlir::ONNXAddOp>,
      ONNXElementwiseVariadicOpLowering<mlir::ONNXAndOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXCastOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXCosOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXCoshOp>,
      ONNXElementwiseVariadicOpLowering<mlir::ONNXDivOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXEluOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXExpOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXHardSigmoidOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXLeakyReluOp>,
      ONNXElementwiseBinaryOpLowering<mlir::ONNXLessOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXLogOp>,
      ONNXElementwiseVariadicOpLowering<mlir::ONNXMaxOp>,
      ONNXElementwiseVariadicOpLowering<mlir::ONNXMinOp>,
      ONNXElementwiseVariadicOpLowering<mlir::ONNXMulOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXNegOp>,
      ONNXElementwiseVariadicOpLowering<mlir::ONNXOrOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXReciprocalOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXReluOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXSeluOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXSigmoidOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXSignOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXSinhOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXSoftplusOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXSoftsignOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXSqrtOp>,
      ONNXElementwiseVariadicOpLowering<mlir::ONNXSubOp>,
      ONNXElementwiseVariadicOpLowering<mlir::ONNXSumOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXTanhOp>,
      ONNXElementwiseVariadicOpLowering<mlir::ONNXXorOp>>(ctx);
}
