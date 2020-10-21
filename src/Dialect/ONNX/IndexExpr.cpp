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
#define CEIL_FLOOR_IN_STD 1

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
// IndexExprContainer utils.
//===----------------------------------------------------------------------===//

IndexExprContainer::IndexExprContainer(
    ConversionPatternRewriter *rewriter, Location loc)
    : rewriter(rewriter), loc(loc), dims(), symbols() {}

int IndexExprContainer::AddDim(Value value) {
  dims.emplace_back(value);
  return dims.size() - 1;
  ;
}
int IndexExprContainer::AddSymbol(Value value) {
  symbols.emplace_back(value);
  return symbols.size() - 1;
}

void IndexExprContainer::DimAndSymbolList(SmallVectorImpl<Value> &list) const {
  list.clear();
  for (auto dim : dims)
    list.emplace_back(dim);
  for (auto sym : symbols)
    list.emplace_back(sym);
}

ConversionPatternRewriter &IndexExprContainer::GetRewriter() const {
  assert(rewriter);
  return *rewriter;
}

//===----------------------------------------------------------------------===//
// IndexExpr constructors, initializers, and copy.
//===----------------------------------------------------------------------===//

IndexExpr::IndexExpr()
    : isIntLit(false), isAffine(false), isSymbol(false), isDim(false),
      isDefined(false), intLit(0), affineExpr(nullptr), value(nullptr) {}

IndexExpr::IndexExpr(int64_t val)
    : isIntLit(true), isAffine(true), isSymbol(false), isDim(false),
      isDefined(true), intLit(val), affineExpr(nullptr), value(nullptr) {}

IndexExpr::IndexExpr(IndexExprContainer &container, Value memref,
    ArrayRef<int64_t> memrefShape, int index) {
  InitAsDim(container, memref, memrefShape, index);
}

void IndexExpr::InitAsUndefined() {
  Init(/*isIntLit*/ false, /*isAffine*/ false, /*isSymbol*/ false,
      /*isDim*/ false, /*isDefined*/ false, 0, AffineExpr(nullptr),
      Value(nullptr));
}

void IndexExpr::InitAsQuestionmark() {
  Init(/*isIntLit*/ false, /*isAffine*/ true, /*isSymbol*/ false,
      /*isDim*/ false, /*isDefined*/ true, 0, AffineExpr(nullptr),
      Value(nullptr));
}

void IndexExpr::InitAsIntLit(int64_t val) {
  Init(/*isIntLit*/ true, /*isAffine*/ true, /*isSymbol*/ false,
      /*isDim*/ false, /*isDefined*/ true, val, AffineExpr(nullptr),
      Value(nullptr));
}

void IndexExpr::InitAsDim(IndexExprContainer &container, Value val) {
  InitAsValueOrIntLit(container, val, /*isAffine*/ true, /*isSymbol*/ false,
      /*isDim*/ true);
}

void IndexExpr::InitAsSymbol(IndexExprContainer &container, Value val) {
  InitAsValueOrIntLit(container, val, /*isAffine*/ true, /*isSymbol*/ true,
      /*isDim*/ false);
}

void IndexExpr::InitAsDim(IndexExprContainer &container, Value memref,
    ArrayRef<int64_t> memrefShape, int index) {
  if (memrefShape[index] < 0) {
    // We have a dynamic dimension.
    Value dynVal = container.GetRewriter().create<DimOp>(
        container.GetLocation(), memref, index);
    InitAsDim(container, dynVal);
  } else {
    // We have a consant dimension.
    int64_t intVal = memrefShape[index];
    InitAsIntLit(intVal);
  }
}

void IndexExpr::InitAsValue(IndexExprContainer &container, Value val) {
  InitAsValueOrIntLit(container, val, /*isAffine*/ false, /*isSymbol*/ false,
      /*isDim*/ false);
}

// Private inits.
void IndexExpr::Init(bool newIsIntLit, bool newIsAffine, bool newIsSymbol,
    bool newIsDim, bool newIsDefined, int newIntLit, AffineExpr newAffineExpr,
    Value newValue) {
  isIntLit = newIsIntLit;
  isAffine = newIsAffine;
  isSymbol = newIsSymbol;
  isDim = newIsDim;
  isDefined = newIsDefined;
  intLit = newIntLit;
  affineExpr = newAffineExpr;
  value = newValue;
}

void IndexExpr::Init(bool newIsIntLit, bool newIsAffine) {
  // The result is affine if the inputs are affine or if the result is a
  // litteral int, as any operations on literals result in a new literal that is
  // by definition affine.
  // Can only use specialized constructor/init to get symbols or dims.
  Init(/*isIntLit*/ newIsIntLit, /*isAffine*/ newIsAffine || newIsIntLit,
      /*isSymbol*/ false, /*isDim*/ false, /*isDefined*/ true, 0,
      AffineExpr(nullptr), Value(nullptr));
}

void IndexExpr::InitAsValueOrIntLit(IndexExprContainer &container, Value val,
    bool newIsAfine, bool newIsSymbol, bool newIsDim) {
  // Do we have a literal integer, if we do, handle it now.
  int64_t valIntLit;
  if (getIntegerLiteralFromValue(val, valIntLit)) {
    // We have an integer. No need for symbol or dim. It is by default affine.
    printf("init symbol as an integer");
    InitAsIntLit(valIntLit);
    return;
  }
  // We have a value, check if it is of the right type.
  auto type = val.getType();
  if (type.isa<IntegerType>()) {
    printf("add an int symbol, convert to index\n");
    // We need to convert the int into an index, since we are dealing with index
    // expressions.
    val = container.GetRewriter().create<IndexCastOp>(
        container.GetLocation(), container.GetRewriter().getIndexType(), val);
  } else if (!type.isa<IndexType>()) {
    llvm_unreachable("unsupported element type");
  }
  // Now record the value.
  assert(!(newIsDim && newIsSymbol) &&
         "cannot have dim and symbol at the same time");
  Init(/*isIntLit*/ false, /*isAffine*/ newIsAfine, /*isSymbol*/ newIsSymbol,
      /*isDim*/ newIsDim, /*isDefined*/ true, 0, AffineExpr(nullptr), val);
}

void IndexExpr::Copy(IndexExpr &a) {
  // If we go to a model like Values & AffineExpr with a pointer to the actual
  // data, we should just make the indirection here. Copy info in the meanwhile.
  *this = a;
}

//===----------------------------------------------------------------------===//
// IndexExpr list querries.
//===----------------------------------------------------------------------===//

bool IndexExpr::IsDefined() const { return isDefined; }
bool IndexExpr::IsIntLit() const {
  assert(IsDefined());
  return isIntLit;
}
bool IndexExpr::IsQuestionmark() const {
  assert(IsDefined());
  return !IsIntLit();
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
bool IndexExpr::HasAffineExpr() const {
  assert(IsDefined());
  return !(!affineExpr);
}
bool IndexExpr::HasValue() const {
  assert(IsDefined());
  return !(!value);
}

bool IndexExpr::AreAllIntLit(SmallVectorImpl<IndexExpr> &list) {
  for (auto index : list) {
    if (!index.IsIntLit())
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
// IndexExpr Getters / Setters.
//===----------------------------------------------------------------------===//

int64_t IndexExpr::GetIntLit() const {
  assert(IsIntLit());
  return intLit;
}

void IndexExpr::SetIntLiteral(int64_t val) {
  assert(IsIntLit()); // Should have been set properly by Init.
  intLit = val;
}

AffineExpr IndexExpr::GetAffineExpr(IndexExprContainer &container) {
  if (IsIntLit()) {
    // Create a literal.
    affineExpr = container.GetRewriter().getAffineConstantExpr(intLit);
  } else if (IsSymbol()) {
    // Create a symbol value expr and register its value in the
    // aresay of symbols.
    assert(HasValue());
    int id = container.AddSymbol(value);
    affineExpr = container.GetRewriter().getAffineSymbolExpr(id);
  } else if (IsDim()) {
    // Create a dim/index value expr and register its value in the
    // aresay of dims/indices.
    assert(HasValue());
    int id = container.AddDim(value);
    affineExpr = container.GetRewriter().getAffineDimExpr(id);
  } else {
    assert(HasAffineExpr());
  }
  return affineExpr;
}

Value IndexExpr::GetValue(IndexExprContainer &container) {
  if (IsIntLit()) {
    // Create a litteral constant.
    value = container.GetRewriter().create<ConstantIndexOp>(
        container.GetLocation(), intLit);
  } else if (HasAffineExpr()) {
    // Has an affine expression: need to build a map, and then perform an
    // affine.apply.
    int dimNum = container.GetDimSize();
    int symNum = container.GetSymbolSize();
    AffineMap map = AffineMap::get(
        dimNum, symNum, {affineExpr}, container.GetRewriter().getContext());
    // We need to concatenate the dims and symbol into a single
    // list, and then use the apply.
    SmallVector<Value, 4> list;
    container.DimAndSymbolList(list);
    value = container.GetRewriter().create<AffineApplyOp>(
        container.GetLocation(), map, list);
  } else {
    assert(HasValue());
  }
  return value;
}

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
  printf("\n");
#endif
}

//===----------------------------------------------------------------------===//
// IndexExpr Op Support.
//===----------------------------------------------------------------------===//

void IndexExpr::BinaryOp(IndexExprContainer &container, IndexExpr &a,
    IndexExpr &b, bool affineWithLitA, bool affineWithLitB,
    bool affineExprCompatible, F2 litFct, F2 affineExprFct, F2 valueFct) {
  // Literal integer if a and b are literals. Affine if both a and b are affine
  // (and possibly b is also constant)
  Init(a.IsIntLit() && b.IsIntLit(), affineExprCompatible && a.IsAffine() &&
                                         b.IsAffine() &&
                                         (!affineWithLitA || a.IsIntLit()) &&
                                         (!affineWithLitB || b.IsIntLit()));
  // We use now use the result of the above determination on whether the new
  // index is literal and/or affine.
  if (IsIntLit()) {
    // Constant, use constant computations.
    litFct(*this, a, b);
  } else if (container.IsShapeInferencePass()) {
    // In shape analysis, if not constant: do noting, aka leave Values & Affine
    // expr undefined.
    assert(IsQuestionmark());
  } else if (IsAffine()) {
    // Use affine values.
    affineExprFct(*this, a, b);
  } else {
    // Use values.
    valueFct(*this, a, b);
  }
}

void IndexExpr::TernaryOp(IndexExprContainer &container, IndexExpr &a,
    IndexExpr &b, IndexExpr &c, F3 litFct, F3 valueFct) {
  // Literal integer if a and b are literals. Affine if both a and b are affine
  // (and possibly b is also constant)
  Init(a.IsIntLit() && b.IsIntLit() && c.IsIntLit(), false);
  // We use now use the result of the above determination on whether the new
  // index is literal and/or affine.
  if (IsIntLit()) {
    // Constant, use constant computations.
    litFct(*this, a, b, c);
  } else if (container.IsShapeInferencePass()) {
    // In shape analysis, if not constant: do noting, aka leave Values & Affine
    // expr undefined.
    assert(IsQuestionmark());
  } else {
    // Use values.
    valueFct(*this, a, b, c);
  }
}

void IndexExpr::QuaternarySelectOp(IndexExprContainer &container,
    IndexExpr &compA, IndexExpr &compB, IndexExpr &trueVal, IndexExpr &falseVal,
    F4 litFct, F4 valueFct) {
  // Check first if the test (ca & cb) can be evaluated at compile time.
  if (compA.IsIntLit() && compB.IsIntLit()) {
    // Comparison will set the right const/affine depending on the input
    // selected, as the compare can be evaluated at compile time.
    litFct(*this, compA, compB, trueVal, falseVal);
  } else if (container.IsShapeInferencePass()) {
    // Just set as undefined
    Init(false, false);
    assert(IsQuestionmark());
  } else {
    // We cannot represent this as an affine expression, so go directly
    // to values.
    Init(false, false);
    valueFct(*this, compA, compB, trueVal, falseVal);
  }
}

void IndexExpr::ReductionOp(IndexExprContainer &container,
    SmallVectorImpl<IndexExpr> &vals, F2 litRed, Flist affineRed, F2 valueRed) {
  auto size = vals.size();
  if (size == 0) {
    InitAsUndefined();
    return;
  }
  // Set the output to the first value.
  Copy(vals[0]);
  // If list has one element, we are done.
  if (vals.size() == 1)
    return;
  // process int literals
  if (AreAllIntLit(vals)) {
    for (int i = 1; i < size; ++i) {
      litRed(*this, vals[i], *this);
    }
  } else if (AreAllAffine(vals)) {
    affineRed(*this, vals);
  } else {
    for (int i = 1; i < size; ++i) {
      valueRed(*this, vals[i], *this);
    }
  }
}

//===----------------------------------------------------------------------===//
// IndexExpr Ops.
//===----------------------------------------------------------------------===//

void IndexExpr::Add(IndexExprContainer &container, IndexExpr &a, IndexExpr &b) {
  F2 litFct = [](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    res.SetIntLiteral(aa.GetIntLit() + bb.GetIntLit());
  };
  F2 affineExprFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    res.SetAffineExpr(
        aa.GetAffineExpr(container) + bb.GetAffineExpr(container));
  };
  F2 valueFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    res.SetValue(container.GetRewriter().create<AddIOp>(container.GetLocation(),
        aa.GetValue(container), bb.GetValue(container)));
  };
  BinaryOp(
      container, a, b, false, false, true, litFct, affineExprFct, valueFct);
}

void IndexExpr::Sub(IndexExprContainer &container, IndexExpr &a, IndexExpr &b) {
  F2 litFct = [](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    res.SetIntLiteral(aa.GetIntLit() - bb.GetIntLit());
  };
  F2 affineExprFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    res.SetAffineExpr(
        aa.GetAffineExpr(container) - bb.GetAffineExpr(container));
  };
  F2 valueFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    res.SetValue(container.GetRewriter().create<SubIOp>(container.GetLocation(),
        aa.GetValue(container), bb.GetValue(container)));
  };
  BinaryOp(
      container, a, b, false, false, true, litFct, affineExprFct, valueFct);
}

void IndexExpr::Mult(
    IndexExprContainer &container, IndexExpr &a, IndexExpr &b) {
  F2 litFct = [](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    res.SetIntLiteral(aa.GetIntLit() * bb.GetIntLit());
  };
  F2 affineExprFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    // Operand aa must be a literal.
    res.SetAffineExpr(bb.GetAffineExpr(container) * aa.GetIntLit());
  };
  F2 valueFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    res.SetValue(container.GetRewriter().create<MulIOp>(container.GetLocation(),
        aa.GetValue(container), bb.GetValue(container)));
  };
  // Index a must be a literal.
  BinaryOp(container, a, b, true, false, true, litFct, affineExprFct, valueFct);
}

void IndexExpr::FloorDiv(
    IndexExprContainer &container, IndexExpr &a, IndexExpr &b) {
  F2 litFct = [](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    int64_t rval = floor((1.0 * aa.GetIntLit()) / (1.0 * bb.GetIntLit()));
    res.SetIntLiteral(rval);
  };
  F2 affineExprFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    // Operand bb must be a literal.
    int64_t bval = bb.GetIntLit();
    if (bval == 1)
      res.Copy(aa);
    else
      res.SetAffineExpr(aa.GetAffineExpr(container).floorDiv(bval));
  };
  F2 valueFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    if (bb.IsIntLit() && bb.GetIntLit() == 1)
      res.Copy(aa);
    else
#if CEIL_FLOOR_IN_STD
      res.SetValue(container.GetRewriter().create<SignedFloorDivIOp>(
          container.GetLocation(), aa.GetValue(container),
          bb.GetValue(container)));
#else
      llvm_unreachable(
          "not implemented yet, wait for the new LLVM/MLIR support in std");
#endif
  };
  // Index b must be a literal.
  BinaryOp(container, a, b, false, true, true, litFct, affineExprFct, valueFct);
}

void IndexExpr::CeilDiv(
    IndexExprContainer &container, IndexExpr &a, IndexExpr &b) {
  F2 litFct = [](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    int64_t rval = ceil((1.0 * aa.GetIntLit()) / (1.0 * bb.GetIntLit()));
    res.SetIntLiteral(rval);
  };
  F2 affineExprFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    // Operand bb must be a literal.
    int64_t bval = bb.GetIntLit();
    if (bval == 1)
      res.Copy(aa);
    else
      res.SetAffineExpr(aa.GetAffineExpr(container).ceilDiv(bval));
  };
  F2 valueFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    if (bb.IsIntLit() && bb.GetIntLit() == 1) {
      res.Copy(aa);
    } else {
#if CEIL_FLOOR_IN_STD
      res.SetValue(container.GetRewriter().create<SignedCeilDivIOp>(
          container.GetLocation(), aa.GetValue(container),
          bb.GetValue(container)));
#else
      llvm_unreachable(
          "not implemented yet, wait for the new LLVM/MLIR support in std");
#endif
    }
  };
  // Index b must be a literal.
  BinaryOp(container, a, b, false, true, true, litFct, affineExprFct, valueFct);
}

void IndexExpr::Mod(IndexExprContainer &container, IndexExpr &a, IndexExpr &b) {
  F2 litFct = [](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    res.SetIntLiteral(mod(aa.GetIntLit(), bb.GetIntLit()));
  };
  F2 affineExprFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    // Operand bb must be a literal.
    res.SetAffineExpr(aa.GetAffineExpr(container) % bb.GetIntLit());
  };
  F2 valueFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    res.SetValue(
        container.GetRewriter().create<SignedRemIOp>(container.GetLocation(),
            aa.GetValue(container), bb.GetValue(container)));
  };
  // Index b must be a literal.
  BinaryOp(container, a, b, false, true, true, litFct, affineExprFct, valueFct);
}

void IndexExpr::Clamp(IndexExprContainer &container, IndexExpr &val,
    IndexExpr &min, int64_t minInc, IndexExpr &max, int64_t maxInc) {
  // Functions below uncoditionally override rr with the clipped value of val.
  F3 litFct = [&](IndexExpr &res, IndexExpr &val, IndexExpr &min,
                  IndexExpr &max) {
    // assume signed compares
    int64_t smin = min.GetIntLit() + minInc;
    int64_t smax = max.GetIntLit() + maxInc;
    int64_t sval = val.GetIntLit();
    if (sval < smin)
      sval = smin;
    if (sval > smax)
      sval = smax;
    res.SetIntLiteral(sval);
  };
  F3 valueFct = [&](IndexExpr &res, IndexExpr &val, IndexExpr &min,
                    IndexExpr &max) {
    IndexExpr minBound(min);
    if (minInc != 0) {
      IndexExpr increment(minInc);
      minBound.Add(container, min, increment);
    }
    IndexExpr newVal(val);
    newVal.Select(container, val, CmpIPredicate::slt, minBound, minBound, val);
    // Copy because don't want to modify the original max.
    IndexExpr maxBound(max);
    if (maxInc != 0) {
      IndexExpr increment(maxInc);
      maxBound.Add(container, max, increment);
    }
    res.Select(
        container, newVal, CmpIPredicate::sgt, maxBound, maxBound, newVal);
  };
  TernaryOp(container, val, min, max, litFct, valueFct);
}

void IndexExpr::Select(IndexExprContainer &container, IndexExpr &condA,
    CmpIPredicate comparePred, IndexExpr &condB, IndexExpr &trueVal,
    IndexExpr &falseVal) {
  F4 litFct = [&](IndexExpr &res, IndexExpr &ca, IndexExpr &cb, IndexExpr &tv,
                  IndexExpr &fv) {
    int64_t sca = ca.GetIntLit();
    int64_t scb = cb.GetIntLit();
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
    auto compareVal =
        container.GetRewriter().create<CmpIOp>(container.GetLocation(),
            comparePred, ca.GetValue(container), cb.GetValue(container));
    auto resVal =
        container.GetRewriter().create<SelectOp>(container.GetLocation(),
            compareVal, tv.GetValue(container), fv.GetValue(container));
    res.SetValue(resVal);
  };
  QuaternarySelectOp(
      container, condA, condB, trueVal, falseVal, litFct, valueFct);
}

void IndexExpr::Min(
    IndexExprContainer &container, SmallVectorImpl<IndexExpr> &vals) {
  F2 litFct = [](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    auto aaa = aa.GetIntLit();
    auto bbb = bb.GetIntLit();
    res.SetIntLiteral((aaa < bbb) ? aaa : bbb);
  };
  Flist affineExprFct = [&](IndexExpr &res, SmallVectorImpl<IndexExpr> &vvals) {
    // Create a list of affine expression
    SmallVector<AffineExpr, 4> affineExprs;
    for (auto vv : vvals) {
      affineExprs.emplace_back(vv.GetAffineExpr(container));
    }
    int dimNum = container.GetDimSize();
    int symNum = container.GetSymbolSize();
    auto context = container.GetRewriter().getContext();

    AffineMap map = AffineMap::get(dimNum, symNum, affineExprs, context);
    SmallVector<Value, 4> dimAndSymList;
    container.DimAndSymbolList(dimAndSymList);
    Value minVal = container.GetRewriter().create<AffineMinOp>(
        container.GetLocation(), map, dimAndSymList);
    res.InitAsValue(container, minVal);
  };
  F2 valueFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    auto compareVal =
        container.GetRewriter().create<CmpIOp>(container.GetLocation(),
            CmpIPredicate::slt, aa.GetValue(container), bb.GetValue(container));
    auto resVal =
        container.GetRewriter().create<SelectOp>(container.GetLocation(),
            compareVal, aa.GetValue(container), bb.GetValue(container));
    res.SetValue(resVal);
  };
  ReductionOp(container, vals, litFct, affineExprFct, valueFct);
}

void IndexExpr::Max(
    IndexExprContainer &container, SmallVectorImpl<IndexExpr> &vals) {
  F2 litFct = [](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    auto aaa = aa.GetIntLit();
    auto bbb = bb.GetIntLit();
    res.SetIntLiteral((aaa > bbb) ? aaa : bbb);
  };
  Flist affineExprFct = [&](IndexExpr &res, SmallVectorImpl<IndexExpr> &vvals) {
    // Create a list of affine expression
    SmallVector<AffineExpr, 4> affineExprs;
    for (auto vv : vvals) {
      affineExprs.emplace_back(vv.GetAffineExpr(container));
    }
    int dimNum = container.GetDimSize();
    int symNum = container.GetSymbolSize();
    auto context = container.GetRewriter().getContext();

    AffineMap map = AffineMap::get(dimNum, symNum, affineExprs, context);
    SmallVector<Value, 4> dimAndSymList;
    container.DimAndSymbolList(dimAndSymList);
    Value minVal = container.GetRewriter().create<AffineMaxOp>(
        container.GetLocation(), map, dimAndSymList);
    res.InitAsValue(container, minVal);
  };
  F2 valueFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    auto compareVal =
        container.GetRewriter().create<CmpIOp>(container.GetLocation(),
            CmpIPredicate::sgt, aa.GetValue(container), bb.GetValue(container));
    auto resVal =
        container.GetRewriter().create<SelectOp>(container.GetLocation(),
            compareVal, aa.GetValue(container), bb.GetValue(container));
    res.SetValue(resVal);
  };
  ReductionOp(container, vals, litFct, affineExprFct, valueFct);
}
