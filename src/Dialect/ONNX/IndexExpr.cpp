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

#include "IndexExpr.hpp"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Support/MathExtras.h"

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

void IndexExprContainer::DimAndSymbolList(SmallVector<Value, 4> &list) const {
  list.clear();
  for (auto dim : dims)
    list.emplace_back(dim);
  for (auto sym : symbols)
    list.emplace_back(sym);
}

ConversionPatternRewriter *IndexExprContainer::GetRewriter() const {
  assert(rewriter);
  return rewriter;
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
  InitAsDimOrIntLit(container, memref, memrefShape, index);
}

void IndexExpr::InitAsIntLit(int64_t val) {
  Init(/*isIntLit*/ true, /*isAffine*/ true, /*isSymbol*/ false,
      /*isDim*/ false, /*isDefined*/ true, val, AffineExpr(nullptr),
      Value(nullptr));
}

void IndexExpr::InitAsDim(Value val) {
  Init(/*isIntLit*/ false, /*isAffine*/ true, /*isSymbol*/ false,
      /*isDim*/ true, /*isDefined*/ true, 0, AffineExpr(nullptr), val);
}

void IndexExpr::InitAsSymbol(Value val) {
  Init(/*isIntLit*/ false, /*isAffine*/ true, /*isSymbol*/ true,
      /*isDim*/ false, /*isDefined*/ true, 0, AffineExpr(nullptr), val);
}

void IndexExpr::InitAsDimOrIntLit(IndexExprContainer &container, Value memref,
    ArrayRef<int64_t> memrefShape, int index) {
  if (memrefShape[index] < 0) {
    // We have a dynamic dimension.
    Value dynVal = container.GetRewriter()->create<DimOp>(
        container.GetLocation(), memref, index);
    InitAsDim(dynVal);
  } else {
    // We have a consant dimension.
    int64_t intVal = memrefShape[index];
    InitAsIntLit(intVal);
  }
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

void IndexExpr::Copy(IndexExpr &a) {
  // If we go to a model like Values & AffineExpr with a pointer to the actual
  // data, we should just make the indirection here. Copy info in the meanwhile.
  *this = a;
}

//===----------------------------------------------------------------------===//
// IndexExpr list querries.
//===----------------------------------------------------------------------===//

  bool IndexExpr::AreAllIntLit(SmallVectorImpl<IndexExpr> &list) {
    for(auto index : list) {
      if (!index.IsIntLit()) return false;
    }
    return true;
  }

  bool IndexExpr::AreAllAffine(SmallVectorImpl<IndexExpr> &list){
    for(auto index : list) {
      if (!index.IsAffine()) return false;
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
    affineExpr = container.GetRewriter()->getAffineConstantExpr(intLit);
  } else if (IsSymbol()) {
    // Create a symbol value expr and register its value in the
    // aresay of symbols.
    assert(HasValue());
    int id = container.AddSymbol(value);
    affineExpr = container.GetRewriter()->getAffineSymbolExpr(id);
  } else if (IsDim()) {
    // Create a dim/index value expr and register its value in the
    // aresay of dims/indices.
    assert(HasValue());
    int id = container.AddDim(value);
    affineExpr = container.GetRewriter()->getAffineDimExpr(id);
  } else {
    assert(HasAffineExpr());
  }
  return affineExpr;
}

Value IndexExpr::GetValue(IndexExprContainer &container) {
  if (IsIntLit()) {
    // Create a litteral constant.
    value = container.GetRewriter()->create<ConstantOp>(container.GetLocation(),
        container.GetRewriter()->getIntegerAttr(
            container.GetRewriter()->getIndexType(), intLit));
  } else if (HasAffineExpr()) {
    // Has an affine expression: need to build a map, and then perform an
    // affine.apply.
    int dimNum = container.GetDimSize();
    int symNum = container.GetSymbolSize();
    AffineMap map = AffineMap::get(
        dimNum, symNum, {affineExpr}, container.GetRewriter()->getContext());
    // We need to concatenate the dims and symbol into a single
    // list, and then use the apply.
    SmallVector<Value, 4> list;
    container.DimAndSymbolList(list);
    value = container.GetRewriter()->create<AffineApplyOp>(
        container.GetLocation(), map, list);
  } else {
    assert(HasValue());
  }
  return value;
}

void IndexExpr::DebugPrint(const std::string &msg) {
#if 1
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
    res.SetValue(
        container.GetRewriter()->create<AddIOp>(container.GetLocation(),
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
    res.SetValue(
        container.GetRewriter()->create<SubIOp>(container.GetLocation(),
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
    res.SetValue(
        container.GetRewriter()->create<MulIOp>(container.GetLocation(),
            aa.GetValue(container), bb.GetValue(container)));
  };
  // Index a must be a literal.
  BinaryOp(container, a, b, true, false, true, litFct, affineExprFct, valueFct);
}

void IndexExpr::FloorDiv(
    IndexExprContainer &container, IndexExpr &a, IndexExpr &b) {
  F2 litFct = [](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    int64_t rval =
        floor((1.0 * aa.GetIntLit()) / (1.0 * bb.GetIntLit()));
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
      res.SetValue(
          container.GetRewriter()->create<SignedDivIOp>(container.GetLocation(),
              aa.GetValue(container), bb.GetValue(container)));
  };
  // Index b must be a literal.
  BinaryOp(container, a, b, false, true, true, litFct, affineExprFct, valueFct);
}

void IndexExpr::CeilDiv(
    IndexExprContainer &container, IndexExpr &a, IndexExpr &b) {
  F2 litFct = [](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    int64_t rval =
        ceil((1.0 * aa.GetIntLit()) / (1.0 * bb.GetIntLit()));
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
      llvm_unreachable(
          "not implemented yet, look at mlir's AffineToStandard.cpp");
      res.SetValue(
          container.GetRewriter()->create<SignedDivIOp>(container.GetLocation(),
              aa.GetValue(container), bb.GetValue(container)));
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
        container.GetRewriter()->create<SignedRemIOp>(container.GetLocation(),
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
        container.GetRewriter()->create<CmpIOp>(container.GetLocation(),
            comparePred, ca.GetValue(container), cb.GetValue(container));
    auto resVal =
        container.GetRewriter()->create<SelectOp>(container.GetLocation(),
            compareVal, tv.GetValue(container), fv.GetValue(container));
    res.SetValue(resVal);
  };
  QuaternarySelectOp(
      container, condA, condB, trueVal, falseVal, litFct, valueFct);
}
