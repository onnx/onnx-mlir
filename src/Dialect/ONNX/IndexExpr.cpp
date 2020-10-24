//===----------------IndexExpr.cpp - Index expression---------------------=== //
//
// Copyright 2020 The IBM Research Authors.
//
// =============================================================================
//
// This file handle index expressions using indices and calcualtions using
// literals, affine expressions, and values.
//
//===----------------------------------------------------------------------===//

// both debug variables will be removed once debugging is complete.
#define DEBUG 0
#define CEIL_FLOOR_IN_STD 0

#include "src/Dialect/ONNX/IndexExpr.hpp"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/MathExtras.h"
#include "src/Dialect/ONNX/ONNXShapeHelper.hpp"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// IndexExprContext constructors.
//===----------------------------------------------------------------------===//

IndexExprContext::IndexExprContext(
    ConversionPatternRewriter *rewriter, Location loc)
    : rewriter(rewriter), loc(loc), dims(), symbols(), parentContext(nullptr) {}

IndexExprContext::IndexExprContext(
    IndexExprContext &newParentContext, bool mayReuseContext)
    : rewriter(newParentContext.rewriter), loc(newParentContext.loc), dims(),
      symbols(), parentContext(nullptr) {
  // When we cannot reuse the parent context, then there is nothing to do but
  // reusing the same rewriter and code location. We are done.
  if (!mayReuseContext)
    return;
  // Otherwise, we can resue the parent context, and in particuliar its affine
  // functions. Now because the affine functions of the parent context have
  // "ids" embedded in the AffineExpr, we must reuse the same mix of Dims and
  // Symbols here. I don't believe there is any sideeffects in considering a Dim
  // from the parent's context as a Dim in the child's context, even though the
  // parent's dim is supposed to be constant in the child's context.
  for (Value parentDim : newParentContext.dims)
    AddDim(parentDim);
  for (Value parentSymbol : newParentContext.symbols)
    AddSymbol(parentSymbol);
  // Save reference to parent context so that we may detect the reuse.
  parentContext = &newParentContext;
}

//===----------------------------------------------------------------------===//
// IndexExprContext builder for IndexExpr.
//===----------------------------------------------------------------------===//

IndexExpr IndexExprContext::CreateUndefinedIndexExpr() {
  IndexExpr res;
  return res.InitAsUndefined();
}

IndexExpr IndexExprContext::CreateQuestionmarkIndexExpr() {
  IndexExpr res;
  return res.InitAsQuestionmark(this);
}

IndexExpr IndexExprContext::CreateLiteralIndexExpr(int64_t val) {
  IndexExpr res;
  return res.InitAsIntLit(this, val);
}

IndexExpr IndexExprContext::CreateDimIndexExpr(Value val) {
  IndexExpr res;
  return res.InitAsDim(this, val);
}

IndexExpr IndexExprContext::CreateDimIndexExpr(
    Value memref, ArrayRef<int64_t> memrefShape, int index) {
  IndexExpr res;
  return res.InitAsDim(this, memref, memrefShape, index);
}

IndexExpr IndexExprContext::CreateSymbolIndexExpr(Value val) {
  IndexExpr res;
  return res.InitAsSymbol(this, val);
}

// Additional builder for repurposing IndexExpr from parent context.
IndexExpr IndexExprContext::CreateSymbolFromParentContext(
    IndexExpr &parentIndexExpr) {
  if (!IsReusingParentContext()) {
    // We are not reusing the parent context; we wil just extract the value from
    // the parent index expression and inject it here as a symbol. By creating
    // it using the normal CreateSymbolIndexExpr, it will get associated with
    // the child's context.
    return CreateSymbolIndexExpr(parentIndexExpr.GetValue());
  }
  // We are now reusing the parent IndexExpr, but we must reset it's context to
  // the child's one.
  IndexExpr childIndexExpr(parentIndexExpr);
  childIndexExpr.SetContext(*this);
  return childIndexExpr;
}

//===----------------------------------------------------------------------===//
// IndexExprContext support for dim and symbol lists in affine exprs.
//===----------------------------------------------------------------------===//

int IndexExprContext::AddDim(Value value) {
  dims.emplace_back(value);
  return dims.size() - 1;
  ;
}
int IndexExprContext::AddSymbol(Value value) {
  symbols.emplace_back(value);
  return symbols.size() - 1;
}

//===----------------------------------------------------------------------===//
// IndexExprContext getters.
//===----------------------------------------------------------------------===//

void IndexExprContext::GetDimAndSymbolList(SmallVectorImpl<Value> &list) const {
  list.clear();
  for (auto dim : dims)
    list.emplace_back(dim);
  for (auto sym : symbols)
    list.emplace_back(sym);
}

ConversionPatternRewriter &IndexExprContext::GetRewriter() const {
  assert(rewriter);
  return *rewriter;
}

//===----------------------------------------------------------------------===//
// IndexExpr constructors, initializers, and copy.
//===----------------------------------------------------------------------===//

IndexExpr::IndexExpr()
    : isDefined(false), isIntLit(false), isAffine(false), isSymbol(false),
      isDim(false), intLit(0), affineExpr(nullptr), value(nullptr),
      context(nullptr) {}

IndexExpr &IndexExpr::InitAsUndefined() {
  return Init(/*context*/ nullptr, /*isDefined*/ false, /*isIntLit*/ false,
      /*isAffine*/ false, /*isSymbol*/ false, /*isDim*/ false, 0,
      AffineExpr(nullptr), Value(nullptr));
}

IndexExpr &IndexExpr::InitAsQuestionmark(IndexExprContext *newContext) {
  return Init(newContext, /*isDefined*/ true, /*isIntLit*/ false,
      /*isAffine*/ true, /*isSymbol*/ false, /*isDim*/ false, 0,
      AffineExpr(nullptr), Value(nullptr));
}

IndexExpr &IndexExpr::InitAsIntLit(IndexExprContext *newContext, int64_t val) {
  return Init(newContext, /*isDefined*/ true, /*isIntLit*/ true,
      /*isAffine*/ true, /*isSymbol*/ false, /*isDim*/ false, val,
      AffineExpr(nullptr), Value(nullptr));
}

IndexExpr &IndexExpr::InitAsDim(IndexExprContext *newContext, Value val) {
  return InitAsValueOrIntLit(
      newContext, val, /*isAffine*/ true, /*isSymbol*/ false, /*isDim*/ true);
}

IndexExpr &IndexExpr::InitAsSymbol(IndexExprContext *newContext, Value val) {
  return InitAsValueOrIntLit(
      newContext, val, /*isAffine*/ true, /*isSymbol*/ true, /*isDim*/ false);
}

IndexExpr &IndexExpr::InitAsAffineExpr(
    IndexExprContext *newContext, AffineExpr val) {
  return Init(newContext, /*isDefined*/ true, /*isIntLit*/ false,
      /*isAffine*/ true, /*isSymbol*/ false, /*isDim*/ false, 0,
      AffineExpr(val), Value(nullptr));
}

IndexExpr &IndexExpr::InitAsValue(IndexExprContext *newContext, Value val) {
  return InitAsValueOrIntLit(newContext, val, /*isAffine*/ false,
      /*isSymbol*/ false, /*isDim*/ false);
}

IndexExpr &IndexExpr::InitAsDim(IndexExprContext *newContext, Value memref,
    ArrayRef<int64_t> memrefShape, int index) {
  if (memrefShape[index] < 0) {
    // We have a dynamic dimension.
    assert(newContext && "expected a context");
    Value dynVal = newContext->GetRewriter().create<DimOp>(
        newContext->GetLocation(), memref, index);
    return InitAsDim(newContext, dynVal);
  }
  // We have a constant dimension.
  int64_t intVal = memrefShape[index];
  return InitAsIntLit(newContext, intVal);
}

IndexExpr &IndexExpr::Init(IndexExprContext *newContext, bool newIsDefined,
    bool newIsIntLit, bool newIsAffine, bool newIsSymbol, bool newIsDim,
    int newIntLit, AffineExpr newAffineExpr, Value newValue) {
  if (newIsDefined)
    assert(newContext && "defined expressions need a context");
  context = newContext;
  isDefined = newIsDefined;
  isIntLit = newIsIntLit;
  isAffine = newIsAffine;
  isSymbol = newIsSymbol;
  isDim = newIsDim;
  intLit = newIntLit;
  affineExpr = newAffineExpr;
  value = newValue;
  return *this;
}

IndexExpr &IndexExpr::InitAsValueOrIntLit(IndexExprContext *newContext,
    Value val, bool newIsAfine, bool newIsSymbol, bool newIsDim) {
  // Do we have a literal integer, if we do, handle it now.
  int64_t valIntLit;
  if (getIntegerLiteralFromValue(val, valIntLit)) {
    // We have an integer. No need for symbol or dim. It is by default affine.
    return InitAsIntLit(newContext, valIntLit);
  }
  // We have a value, check if it is of the right type.
  auto type = val.getType();
  if (type.isa<IntegerType>()) {
    // We need to convert the int into an index, since we are dealing with index
    // expressions.
    assert(newContext && "defined expressions need a context");
    val =
        newContext->GetRewriter().create<IndexCastOp>(newContext->GetLocation(),
            newContext->GetRewriter().getIndexType(), val);
  } else {
    assert(type.isa<IndexType>() && "unsupported element type");
  }
  // Now record the value. Affine Expr will be created on demand by
  // GetAffineExpr.
  assert(!(newIsDim && newIsSymbol) &&
         "cannot have dim and symbol at the same time");
  return Init(newContext, /*isDefined*/ true, /*isIntLit*/ false,
      /*isAffine*/ newIsAfine, /*isSymbol*/ newIsSymbol, /*isDim*/ newIsDim, 0,
      AffineExpr(nullptr), val);
}

//===----------------------------------------------------------------------===//
// IndexExpr copy and setters.
//===----------------------------------------------------------------------===//

IndexExpr &IndexExpr::Copy(IndexExpr &a) {
  // If we go to a model like Values & AffineExpr with a pointer to the actual
  // data, we should just make the indirection here. Copy info in the meanwhile.
  *this = a;
  return *this;
}

void IndexExpr::SetContext(IndexExprContext &newContext) {
  context = &newContext;
}

//===----------------------------------------------------------------------===//
// IndexExpr list querries.
//===----------------------------------------------------------------------===//

bool IndexExpr::IsDefined() const {
  assert(!isDefined || HasContext());
  return isDefined;
}

bool IndexExpr::IsLiteral() const {
  assert(IsDefined());
  return isIntLit;
}

bool IndexExpr::IsQuestionmark() const {
  assert(IsDefined());
  return !IsLiteral();
}

bool IndexExpr::IsAffine() const {
  assert(IsDefined());
  return isAffine;
}

bool IndexExpr::IsSymbol() const {
  assert(IsDefined());
  return isSymbol;
}

bool IndexExpr::IsDim() const {
  assert(IsDefined());
  return isDim;
}

bool IndexExpr::IsShapeInferencePass() const {
  assert(HasContext());
  return context->IsShapeInferencePass();
}

bool IndexExpr::HasContext() const { return context != nullptr; }

bool IndexExpr::HasAffineExpr() const {
  assert(IsDefined());
  return !(!affineExpr);
}

bool IndexExpr::HasValue() const {
  assert(IsDefined());
  return !(!value);
}

bool IndexExpr::AreAllLiteral(SmallVectorImpl<IndexExpr> &list) {
  for (auto index : list) {
    if (!index.IsLiteral())
      return false;
  }
  return true;
}

bool IndexExpr::AreAllAffine(SmallVectorImpl<IndexExpr> &list) {
  for (auto index : list) {
    if (!index.IsAffine())
      return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// IndexExpr Getters.
//===----------------------------------------------------------------------===//

int64_t IndexExpr::GetLiteral() const {
  assert(IsLiteral());
  return intLit;
}

AffineExpr IndexExpr::GetAffineExpr() {
  assert(!IsShapeInferencePass() && "cannot get affine during shape inference");
  if (IsLiteral()) {
    // Create a literal.
    affineExpr = context->GetRewriter().getAffineConstantExpr(intLit);
  } else if (IsSymbol()) {
    // Create a symbol value expr and register its value in the
    // array of symbols. Has value because symbols are gen on demand from
    // values.
    assert(HasValue());
    int id = context->AddSymbol(value);
    affineExpr = context->GetRewriter().getAffineSymbolExpr(id);
  } else if (IsDim()) {
    // Create a dim/index value expr and register its value in the
    // array of dims/indices. Has value because dims are gen on demand from
    // values.
    assert(HasValue());
    int id = context->AddDim(value);
    affineExpr = context->GetRewriter().getAffineDimExpr(id);
  } else {
    assert(
        HasAffineExpr() && "requesting affine expr of incompatible IndexExpr");
  }
  return affineExpr;
}

Value IndexExpr::GetValue() {
  assert(!IsShapeInferencePass() && "cannot get affine during shape inference");
  if (IsLiteral()) {
    // Create a litteral constant.
    value = context->GetRewriter().create<ConstantIndexOp>(
        context->GetLocation(), intLit);
  } else if (HasAffineExpr()) {
    // Has an affine expression: need to build a map, and then perform an
    // affine.apply.
    int dimNum = context->GetDimSize();
    int symNum = context->GetSymbolSize();
    AffineMap map = AffineMap::get(
        dimNum, symNum, {affineExpr}, context->GetRewriter().getContext());
    // We need to concatenate the dims and symbol into a single
    // list, and then use the apply.
    SmallVector<Value, 4> list;
    context->GetDimAndSymbolList(list);
    value = context->GetRewriter().create<AffineApplyOp>(
        context->GetLocation(), map, list);
  } else {
    assert(HasValue());
  }
  return value;
}

IndexExprContext *IndexExpr::GetContext() const {
  assert(HasContext());
  return context;
}

Location IndexExpr::GetLocation() const { return GetContext()->GetLocation(); }

void IndexExpr::DebugPrint(const std::string &msg) {
#if DEBUG
  printf("%s:", msg.c_str());
  if (IsIntLit())
    printf(" val(%lli)", GetIntLit());
  if (HasAffineExpr())
    printf(" hasAffine");
  if (HasValue())
    printf(" hasValue");
  if (IsAffine())
    printf(" is affine");
  printf(" context(0x%llx)\n", (long long unsigned)context);
#endif
}

//===----------------------------------------------------------------------===//
// IndexExpr Op Support.
//===----------------------------------------------------------------------===//

// Used for Add/Sub/Mult/CeilDiv/FloorDiv
IndexExpr &IndexExpr::BinaryOp(IndexExpr &a, IndexExpr &b, bool affineWithLitB,
    bool canBeAffine, F2 litFct, F2 affineExprFct, F2 valueFct) {
  assert(a.GetContext() == b.GetContext() && "incompatible contexts");
  // Literal integer if a and b are literals. Affine if canBeAffine is true,
  // both a and b are affine, and possibly a and/or b are also constant.
  bool resIsLit = a.IsLiteral() && b.IsLiteral();
  bool resIsAffine = resIsLit || (canBeAffine && a.IsAffine() && b.IsAffine() &&
                                     (!affineWithLitB || b.IsLiteral()));

  // We use now use the result of the above determination on whether the new
  // index is literal and/or affine.
  if (resIsLit) {
    // Constant, use constant computations.
    litFct(*this, a, b);
  } else if (a.IsShapeInferencePass()) {
    // In shape analysis, if not constant: do noting, aka leave Values & Affine
    // expr undefined.
    InitAsQuestionmark(a.GetContext());
  } else if (resIsAffine) {
    // Use affine values.
    affineExprFct(*this, a, b);
  } else {
    // Use values.
    valueFct(*this, a, b);
  }
  return *this;
}

// Used for Clamp.
IndexExpr &IndexExpr::TernaryOp(
    IndexExpr &a, IndexExpr &b, IndexExpr &c, F3 litFct, F3 valueFct) {
  assert(a.GetContext() == b.GetContext() && a.GetContext() == c.GetContext() &&
         "incompatible contexts");
  // Literal integer if a, b, and c are literals. Output is not affine (unless
  // all 3 are literals).
  bool resIsLit = a.IsLiteral() && b.IsLiteral() && c.IsLiteral();
  // We use now use the result of the above determination on whether the new
  // index is literal and/or affine.
  if (resIsLit) {
    // Constant, use constant computations.
    litFct(*this, a, b, c);
  } else if (a.IsShapeInferencePass()) {
    // In shape analysis, if not constant: do noting, aka leave Values & Affine
    // expr undefined.
    InitAsQuestionmark(a.GetContext());
  } else {
    // Use values.
    valueFct(*this, a, b, c);
  }
  return *this;
}

IndexExpr &IndexExpr::QuaternarySelectOp(IndexExpr &compA, IndexExpr &compB,
    IndexExpr &trueVal, IndexExpr &falseVal, F4 litFct, F4 valueFct) {
  assert(compA.GetContext() == compB.GetContext() &&
         compA.GetContext() == trueVal.GetContext() &&
         compA.GetContext() == falseVal.GetContext() &&
         "incompatible contexts");
  // Check first if the test (ca & cb) can be evaluated at compile time.
  if (compA.IsLiteral() && compB.IsLiteral()) {
    // Comparison will set the right const/affine depending on the input
    // selected, as the compare can be evaluated at compile time.
    litFct(*this, compA, compB, trueVal, falseVal);
  } else if (compA.IsShapeInferencePass()) {
    // Just set as undefined
    InitAsQuestionmark(compA.GetContext());
  } else {
    // We cannot represent this as an affine expression, so go directly
    // to values.
    valueFct(*this, compA, compB, trueVal, falseVal);
  }
  return *this;
}

// The affine reduction labda function processes the whole list and must init
// the result.
IndexExpr &IndexExpr::ReductionOp(
    SmallVectorImpl<IndexExpr> &vals, F2 litRed, Flist affineRed, F2 valueRed) {
  // If no values, result is undefined.
  int size = vals.size();
  if (size == 0) {
    InitAsUndefined();
    return *this;
  }
  // Set the output to the first value.
  Copy(vals[0]);
  // If list has one element, we are done. Literal/Affine... will be the same as
  // this single element.
  if (vals.size() == 1)
    return *this;
  // Have multiple values, need to do some checks.
  bool resIsLit = true;
  bool resIsAffine = true;
  for (int i = 0; i < size; ++i) {
    if (!vals[i].IsLiteral())
      resIsLit = false;
    if (!vals[i].IsAffine())
      resIsAffine = false;
    assert(vals[0].GetContext() == vals[i].GetContext() &&
           "incompatible contexts");
  }
  if (resIsLit) {
    // Process int literals, if we only have literal values.
    // Result was set to first element, which by default is literal/affine. This
    // will be the correct result for the output.
    for (int i = 1; i < size; ++i) {
      litRed(*this, vals[i], *this);
    }
  } else if (vals[0].IsShapeInferencePass()) {
    // Just set as undefined
    InitAsQuestionmark(vals[0].GetContext());
  } else if (resIsAffine) {
    affineRed(*this, vals);
  } else {
    for (int i = 1; i < size; ++i) {
      valueRed(*this, vals[i], *this);
    }
  }
  return *this;
}

//===----------------------------------------------------------------------===//
// IndexExpr Ops.
//===----------------------------------------------------------------------===//

IndexExpr &IndexExpr::Add(IndexExpr &a, IndexExpr &b) {
  F2 litFct = [](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    res.InitAsIntLit(aa.GetContext(), aa.GetLiteral() + bb.GetLiteral());
  };
  F2 affineExprFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    res.InitAsAffineExpr(
        aa.GetContext(), aa.GetAffineExpr() + bb.GetAffineExpr());
  };
  F2 valueFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    res.InitAsValue(
        aa.GetContext(), aa.GetContext()->GetRewriter().create<AddIOp>(
                             aa.GetLocation(), aa.GetValue(), bb.GetValue()));
  };
  return BinaryOp(a, b, false, true, litFct, affineExprFct, valueFct);
}

IndexExpr &IndexExpr::Sub(IndexExpr &a, IndexExpr &b) {
  F2 litFct = [](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    res.InitAsIntLit(aa.GetContext(), aa.GetLiteral() - bb.GetLiteral());
  };
  F2 affineExprFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    res.InitAsAffineExpr(
        aa.GetContext(), aa.GetAffineExpr() - bb.GetAffineExpr());
  };
  F2 valueFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    res.InitAsValue(
        aa.GetContext(), aa.GetContext()->GetRewriter().create<SubIOp>(
                             aa.GetLocation(), aa.GetValue(), bb.GetValue()));
  };
  return BinaryOp(a, b, false, true, litFct, affineExprFct, valueFct);
}

IndexExpr &IndexExpr::Mult(IndexExpr &a, IndexExpr &b) {
  // In the lambda function below, if one is literal, it is assumed that it is
  // in the second position (b).
  F2 litFct = [](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    res.InitAsIntLit(aa.GetContext(), aa.GetLiteral() * bb.GetLiteral());
  };
  F2 affineExprFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    // Operand aa must be a literal.
    res.InitAsAffineExpr(aa.GetContext(), aa.GetAffineExpr() * bb.GetLiteral());
  };
  F2 valueFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    if (bb.IsLiteral() && bb.GetLiteral() == 1) {
      res.Copy(aa);
    } else {
      res.InitAsValue(
          aa.GetContext(), aa.GetContext()->GetRewriter().create<MulIOp>(
                               aa.GetLocation(), aa.GetValue(), bb.GetValue()));
    }
  };
  // Literal should be place in second argument; do so if a is a lit.
  if (a.IsLiteral())
    return BinaryOp(b, a, true, true, litFct, affineExprFct, valueFct);
  return BinaryOp(a, b, true, true, litFct, affineExprFct, valueFct);
}

IndexExpr &IndexExpr::FloorDiv(IndexExpr &a, IndexExpr &b) {
  F2 litFct = [](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    int64_t rval = floor((1.0 * aa.GetLiteral()) / (1.0 * bb.GetLiteral()));
    res.InitAsIntLit(aa.GetContext(), rval);
  };
  F2 affineExprFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    // Operand bb must be a literal.
    int64_t bval = bb.GetLiteral();
    if (bval == 1) {
      res.Copy(aa);
    } else if (bval > 1) {
      res.InitAsAffineExpr(aa.GetContext(), aa.GetAffineExpr().floorDiv(bval));
    } else {
#if CEIL_FLOOR_IN_STD
      res.InitAsValue(aa.GetContext(),
          aa.GetContext()->GetRewriter().create<SignedFloorDivIOp>(
              aa.GetLocation(), aa.GetValue(), bb.GetValue()));
#else
      llvm_unreachable("not implemented yet, wait for the new LLVM/MLIR "
                       "support in std");
#endif
    }
  };
  F2 valueFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    if (bb.IsLiteral() && bb.GetLiteral() == 1) {
      res.Copy(aa);
    } else {
#if CEIL_FLOOR_IN_STD
      res.InitAsValue(aa.GetContext(),
          aa.GetContext()->GetRewriter().create<SignedFloorDivIOp>(
              aa.GetLocation(), aa.GetValue(), bb.GetValue()));
#else
      llvm_unreachable("not implemented yet, wait for the new LLVM/MLIR "
                       "support in std");
#endif
    }
  };
  // Index b must be a literal.
  return BinaryOp(a, b, true, true, litFct, affineExprFct, valueFct);
}

IndexExpr &IndexExpr::CeilDiv(IndexExpr &a, IndexExpr &b) {
  F2 litFct = [](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    int64_t rval = ceil((1.0 * aa.GetLiteral()) / (1.0 * bb.GetLiteral()));
    res.InitAsIntLit(aa.GetContext(), rval);
  };
  F2 affineExprFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    // Operand bb must be a literal.
    int64_t bval = bb.GetLiteral();
    if (bval == 1) {
      res.Copy(aa);
    } else if (bval > 1) {
      res.InitAsAffineExpr(aa.GetContext(), aa.GetAffineExpr().ceilDiv(bval));
    } else {
#if CEIL_FLOOR_IN_STD
      res.InitAsValue(aa.GetContext(),
          aa.GetContext()->GetRewriter().create<SignedCeilDivIOp>(
              aa.GetLocation(), aa.GetValue(), bb.GetValue()));
#else
      llvm_unreachable(
          "not implemented yet, wait for the new LLVM/MLIR support in std");
#endif
    }
  };
  F2 valueFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    if (bb.IsLiteral() && bb.GetLiteral() == 1) {
      res.Copy(aa);
    } else {
#if CEIL_FLOOR_IN_STD
      res.InitAsValue(aa.GetContext(),
          aa.GetContext()->GetRewriter().create<SignedCeilDivIOp>(
              aa.GetLocation(), aa.GetValue(), bb.GetValue()));
#else
      llvm_unreachable(
          "not implemented yet, wait for the new LLVM/MLIR support in std");
#endif
    }
  };
  // Index b must be a literal.
  return BinaryOp(a, b, true, true, litFct, affineExprFct, valueFct);
}

IndexExpr &IndexExpr::Mod(IndexExpr &a, IndexExpr &b) {
  F2 litFct = [](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    res.InitAsIntLit(aa.GetContext(), mod(aa.GetLiteral(), bb.GetLiteral()));
  };
  F2 affineExprFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    // Operand bb must be a literal.
    int64_t bval = bb.GetLiteral();
    if (bval >= 0) {
      res.InitAsAffineExpr(aa.GetContext(), aa.GetAffineExpr() % bval);
    } else {
      res.InitAsValue(
          aa.GetContext(), aa.GetContext()->GetRewriter().create<SignedRemIOp>(
                               aa.GetLocation(), aa.GetValue(), bb.GetValue()));
    }
  };
  F2 valueFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    res.InitAsValue(
        aa.GetContext(), aa.GetContext()->GetRewriter().create<SignedRemIOp>(
                             aa.GetLocation(), aa.GetValue(), bb.GetValue()));
  };
  // Index b must be a literal.
  return BinaryOp(a, b, true, true, litFct, affineExprFct, valueFct);
}

IndexExpr &IndexExpr::Clamp(IndexExpr &val, IndexExpr &min, IndexExpr &max) {
  // Functions below uncoditionally override rr with the clipped value of val.
  F3 litFct = [&](IndexExpr &res, IndexExpr &val, IndexExpr &min,
                  IndexExpr &max) {
    // assume signed compares
    int64_t smin = min.GetLiteral();
    int64_t smax = max.GetLiteral();
    int64_t sval = val.GetLiteral();
    if (sval < smin)
      sval = smin;
    if (sval > smax)
      sval = smax;
    res.InitAsIntLit(val.GetContext(), sval);
  };
  F3 valueFct = [&](IndexExpr &res, IndexExpr &val, IndexExpr &min,
                    IndexExpr &max) {
    // Copy min, max, and val as we don't want to change the original values.
    IndexExpr minBound(min), newVal(val), maxBound(max);
    newVal.Select(val, CmpIPredicate::slt, minBound, minBound, val);
    res.Select(newVal, CmpIPredicate::sgt, maxBound, maxBound, newVal);
  };
  return TernaryOp(val, min, max, litFct, valueFct);
}

IndexExpr &IndexExpr::Select(IndexExpr &condA, CmpIPredicate comparePred,
    IndexExpr &condB, IndexExpr &trueVal, IndexExpr &falseVal) {
  F4 litFct = [&](IndexExpr &res, IndexExpr &ca, IndexExpr &cb, IndexExpr &tv,
                  IndexExpr &fv) {
    int64_t sca = ca.GetLiteral();
    int64_t scb = cb.GetLiteral();
    uint64_t uca = (uint64_t)sca;
    uint64_t ucb = (uint64_t)scb;
    switch (comparePred) {
    case CmpIPredicate::eq:
      if (sca == scb)
        res.Copy(tv);
      else
        res.Copy(fv);
      break;
    case CmpIPredicate::ne:
      if (sca != scb)
        res.Copy(tv);
      else
        res.Copy(fv);
      break;
    case CmpIPredicate::slt:
      if (sca < scb)
        res.Copy(tv);
      else
        res.Copy(fv);
      break;
    case CmpIPredicate::sle:
      if (sca <= scb)
        res.Copy(tv);
      else
        res.Copy(fv);
      break;
    case CmpIPredicate::sgt:
      if (sca > scb)
        res.Copy(tv);
      else
        res.Copy(fv);
      break;
    case CmpIPredicate::sge:
      if (sca >= scb)
        res.Copy(tv);
      else
        res.Copy(fv);
      break;
    case CmpIPredicate::ult:
      if (uca < ucb)
        res.Copy(tv);
      else
        res.Copy(fv);
      break;
    case CmpIPredicate::ule:
      if (uca <= ucb)
        res.Copy(tv);
      else
        res.Copy(fv);
      break;
    case CmpIPredicate::ugt:
      if (uca > ucb)
        res.Copy(tv);
      else
        res.Copy(fv);
      break;
    case CmpIPredicate::uge:
      if (uca >= ucb)
        res.Copy(tv);
      else
        res.Copy(fv);
      break;
    default:
      llvm_unreachable("unknown compare opeartor");
    }
  };
  F4 valueFct = [&](IndexExpr &res, IndexExpr &ca, IndexExpr &cb, IndexExpr &tv,
                    IndexExpr &fv) {
    Value compare = ca.GetContext()->GetRewriter().create<CmpIOp>(
        ca.GetLocation(), comparePred, ca.GetValue(), cb.GetValue());
    Value results = ca.GetContext()->GetRewriter().create<SelectOp>(
        ca.GetLocation(), compare, tv.GetValue(), fv.GetValue());
    res.InitAsValue(ca.GetContext(), results);
  };
  return QuaternarySelectOp(condA, condB, trueVal, falseVal, litFct, valueFct);
}

IndexExpr &IndexExpr::Min(SmallVectorImpl<IndexExpr> &vals) {
  F2 litFct = [](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    auto aaa = aa.GetLiteral();
    auto bbb = bb.GetLiteral();
    res.InitAsIntLit(aa.GetContext(), (aaa < bbb) ? aaa : bbb);
  };
  Flist affineExprFct = [&](IndexExpr &res, SmallVectorImpl<IndexExpr> &vvals) {
    // Create a list of affine expression
    assert(vvals.size() > 1 && "come here only with 2 or more values");
    SmallVector<AffineExpr, 4> affineExprs;
    for (IndexExpr &vv : vvals) {
      affineExprs.emplace_back(vv.GetAffineExpr());
    }
    // Compute a map including the list of affine expressions.
    IndexExprContext *currContext = vvals[0].GetContext();
    int dimNum = currContext->GetDimSize();
    int symNum = currContext->GetSymbolSize();
    auto mapContext = currContext->GetRewriter().getContext();
    AffineMap map = AffineMap::get(dimNum, symNum, affineExprs, mapContext);
    // Compute the min value out of this map.
    SmallVector<Value, 4> dimAndSymList;
    currContext->GetDimAndSymbolList(dimAndSymList);
    Value minVal = currContext->GetRewriter().create<AffineMinOp>(
        vvals[0].GetLocation(), map, dimAndSymList);
    res.InitAsValue(context, minVal);
  };
  F2 valueFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    Value compareVal = aa.GetContext()->GetRewriter().create<CmpIOp>(
        aa.GetLocation(), CmpIPredicate::slt, aa.GetValue(), bb.GetValue());
    Value resVal = aa.GetContext()->GetRewriter().create<SelectOp>(
        aa.GetLocation(), compareVal, aa.GetValue(), bb.GetValue());
    res.InitAsValue(aa.GetContext(), resVal);
  };
  return ReductionOp(vals, litFct, affineExprFct, valueFct);
}

IndexExpr &IndexExpr::Max(SmallVectorImpl<IndexExpr> &vals) {
  F2 litFct = [](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    auto aaa = aa.GetLiteral();
    auto bbb = bb.GetLiteral();
    res.InitAsIntLit(aa.GetContext(), (aaa > bbb) ? aaa : bbb);
  };
  Flist affineExprFct = [&](IndexExpr &res, SmallVectorImpl<IndexExpr> &vvals) {
    // Create a list of affine expression
    assert(vvals.size() > 1 && "come here only with 2 or more values");
    SmallVector<AffineExpr, 4> affineExprs;
    for (IndexExpr &vv : vvals) {
      affineExprs.emplace_back(vv.GetAffineExpr());
    }
    // Compute a map including the list of affine expressions.
    IndexExprContext *currContext = vvals[0].GetContext();
    int dimNum = currContext->GetDimSize();
    int symNum = currContext->GetSymbolSize();
    auto mapContext = currContext->GetRewriter().getContext();
    AffineMap map = AffineMap::get(dimNum, symNum, affineExprs, mapContext);
    // Compute the min value out of this map.
    SmallVector<Value, 4> dimAndSymList;
    currContext->GetDimAndSymbolList(dimAndSymList);
    Value minVal = currContext->GetRewriter().create<AffineMaxOp>(
        vvals[0].GetLocation(), map, dimAndSymList);
    res.InitAsValue(context, minVal);
  };
  F2 valueFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    Value compareVal = aa.GetContext()->GetRewriter().create<CmpIOp>(
        aa.GetLocation(), CmpIPredicate::sgt, aa.GetValue(), bb.GetValue());
    Value resVal = aa.GetContext()->GetRewriter().create<SelectOp>(
        aa.GetLocation(), compareVal, aa.GetValue(), bb.GetValue());
    res.InitAsValue(aa.GetContext(), resVal);
  };
  return ReductionOp(vals, litFct, affineExprFct, valueFct);
}

//===----------------------------------------------------------------------===//
// IndexExpr Ops Derivatives
//===----------------------------------------------------------------------===//

IndexExpr &IndexExpr::Add(IndexExpr &a, int64_t b) {
  IndexExpr bIndex = a.GetContext()->CreateLiteralIndexExpr(b);
  return Add(a, bIndex);
}

IndexExpr &IndexExpr::IncBy(IndexExpr &b) { return Add(*this, b); }

IndexExpr &IndexExpr::IncBy(int64_t b) { return Add(*this, b); }

IndexExpr &IndexExpr::Sub(IndexExpr &a, int64_t b) {
  IndexExpr bIndex = a.GetContext()->CreateLiteralIndexExpr(b);
  return Sub(a, bIndex);
}

IndexExpr &IndexExpr::Sub(int64_t a, IndexExpr &b) {
  IndexExpr aIndex = b.GetContext()->CreateLiteralIndexExpr(a);
  return Sub(aIndex, b);
}

IndexExpr &IndexExpr::DecBy(IndexExpr &b) { return Sub(*this, b); }

IndexExpr &IndexExpr::DecBy(int64_t b) { return Sub(*this, b); }

IndexExpr &IndexExpr::Mult(IndexExpr &a, int64_t b) {
  IndexExpr bIndex = a.GetContext()->CreateLiteralIndexExpr(b);
  return Mult(a, bIndex);
}

IndexExpr &IndexExpr::MultBy(IndexExpr &b) { return Mult(*this, b); }

IndexExpr &IndexExpr::MultBy(int64_t b) { return Mult(*this, b); }

IndexExpr &IndexExpr::Clamp(IndexExpr &val, int64_t min, IndexExpr &max) {
  IndexExpr minIndex = val.GetContext()->CreateLiteralIndexExpr(min);
  return Clamp(val, minIndex, max);
}

IndexExpr &IndexExpr::FloorDivBy(IndexExpr &b) { return FloorDiv(*this, b); }
IndexExpr &IndexExpr::CeilDivBy(IndexExpr &b) { return CeilDiv(*this, b); }

IndexExpr &IndexExpr::Select(IndexExpr &condA, CmpIPredicate comparePred,
    int64_t condB, IndexExpr &trueVal, IndexExpr &falseVal) {
  IndexExpr condBIndex = condA.GetContext()->CreateLiteralIndexExpr(condB);
  return Select(condA, comparePred, condBIndex, trueVal, falseVal);
}

IndexExpr &IndexExpr::Select(IndexExpr &condA, CmpIPredicate comparePred,
    int64_t condB, int64_t trueVal, IndexExpr &falseVal) {
  IndexExpr condBIndex = condA.GetContext()->CreateLiteralIndexExpr(condB);
  IndexExpr trueValIndex = condA.GetContext()->CreateLiteralIndexExpr(trueVal);
  return Select(condA, comparePred, condBIndex, trueValIndex, falseVal);
}

IndexExpr &IndexExpr::AssignIf(IndexExpr &condA, CmpIPredicate comparePred,
    int64_t condB, IndexExpr &trueVal) {
  return Select(condA, comparePred, condB, trueVal, *this);
}

IndexExpr &IndexExpr::AssignIf(IndexExpr &condA, CmpIPredicate comparePred,
    int64_t condB, int64_t trueVal) {
  return Select(condA, comparePred, condB, trueVal, *this);
}
