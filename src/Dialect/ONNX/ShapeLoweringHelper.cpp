#include "ShapeLoweringHelper.hpp"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace mlir;

// Constructors.

ShapeValue::ShapeValue(ShapeValue &a)
    : constVal(a.constVal), dynVal(a.dynVal), rewriter(a.rewriter),
      isConst(a.isConst), isAffine(a.isAffine) {}

ShapeValue::ShapeValue(int64_t val, ConversionPatternRewriter *rewriter) {
  Init(val, rewriter);
}

ShapeValue::ShapeValue(
    Value val, bool affine, ConversionPatternRewriter *rewrite) {
  Init(val, affine, rewrite);
}

// Copy.

void ShapeValue::Copy(ShapeValue &a) {
  Init(a);
  constVal = a.constVal;
  dynVal = a.dynVal;
}

// Initializers.

void ShapeValue::Init(int64_t val, ConversionPatternRewriter *convRewriter) {
  constVal = val;
  rewriter = convRewriter;
  isConst = true;
  isAffine = true;
}

void ShapeValue::Init(
    Value val, bool affine, ConversionPatternRewriter *convRewriter) {
  dynVal = val;
  rewriter = convRewriter;
  isConst = false;
  isAffine = affine;
}

void ShapeValue::Init(ShapeValue &a) {
  rewriter = a.rewriter;
  isConst = a.isConst;
  isAffine = a.isAffine;
}

void ShapeValue::Init(ShapeValue &a, ShapeValue &b, bool affineIfBConst) {
  rewriter = a.rewriter;
  isConst = a.isConst & b.isConst;
  isAffine = a.isAffine & b.isAffine;
  if (affineIfBConst && !b.isConst)
    isAffine = false;
}

void ShapeValue::Init(ShapeValue &a, ShapeValue &b, ShapeValue &c) {
  rewriter = a.rewriter;
  isConst = a.isConst & b.isConst & c.isConst;
  isAffine = a.isAffine & b.isAffine & c.isAffine;
}

void ShapeValue::Init(
    ShapeValue &a, ShapeValue &b, ShapeValue &c, ShapeValue &d) {
  rewriter = a.rewriter;
  isConst = a.isConst & b.isConst & c.isConst & d.isConst;
  isAffine = a.isAffine & b.isAffine & c.isAffine & d.isConst;
}

// Getters/Setters.

int64_t ShapeValue::GetConstVal() {
  assert(IsConst());
  return constVal;
}

void ShapeValue::SetConstVal(int64_t val) {
  assert(IsConst());
  constVal = val;
}

Value ShapeValue::GetDynVal(Location loc) {
  assert(rewriter);
  MakeDynamic(loc);
  return dynVal;
}

void ShapeValue::SetDynVal(Value val) {
  assert(rewriter);
  dynVal = val;
}

ConversionPatternRewriter *ShapeValue::GetRewriter() {
  assert(rewriter);
  return rewriter;
}

// Convert from constant value to dynamic.
void ShapeValue::MakeDynamic(Location loc) {
  assert(rewriter);
  if (IsDynamic())
    return;
  isConst = false;
  // Create value from integer.
  rewriter->create<ConstantOp>(
      loc, rewriter->getIntegerAttr(rewriter->getIndexType(), 0));
}

// Operator support.

void ShapeValue::UnaryOp(ShapeValue &a, F1 finteger, F1 fvalue) {
  // Constant/affine if a is const/afine.
  Init(a);
  // Constant, use constant computations.
  if (IsConst())
    finteger(*this, a);
  // Else if we can create ops, use dyn computations.
  else if (rewriter)
    fvalue(*this, a);
  // We have a non constant without being able to create a value;
  // we get a question mark.
  else
    assert(IsQuestionmark());
}

void ShapeValue::BinaryOp(
    ShapeValue &a, ShapeValue &b, F2 finteger, F2 fvalue, bool affineIfBConst) {
  // Constant if a and b are const.
  // Affine if both a and b are affine (and possibly b is also constant)
  Init(a, b, affineIfBConst);
  // Constant, use constant computations.
  if (IsConst())
    finteger(*this, a, b);
  // Else if we can create ops, use dyn computations.
  else if (rewriter)
    fvalue(*this, a, b);
  // We have a non constant without being able to create a value;
  // we get a question mark.
  else
    assert(IsQuestionmark());
}

void ShapeValue::TernaryOp(
    ShapeValue &a, ShapeValue &b, ShapeValue &c, F3 finteger, F3 fvalue) {
  // Constant if a, b, and c are const.
  // Affine if all are affine
  Init(a, b, c);
  // Constant, use constant computations.
  if (IsConst())
    finteger(*this, a, b, c);
  // Else if we can create ops, use dyn computations.
  else if (rewriter)
    fvalue(*this, a, b, c);
  // We have a non constant without being able to create a value;
  // we get a question mark.
  else
    assert(IsQuestionmark());
}

void ShapeValue::QuaternaryOp(ShapeValue &a, ShapeValue &b, ShapeValue &c,
    ShapeValue &d, F4 finteger, F4 fvalue) {
  // Constant if a, b, and c are const.
  // Affine if all are affine
  Init(a, b, c, d);
  // Constant, use constant computations.
  if (IsConst())
    finteger(*this, a, b, c, d);
  // Else if we can create ops, use dyn computations.
  else if (rewriter)
    fvalue(*this, a, b, c, d);
  // We have a non constant without being able to create a value;
  // we get a question mark.
  else
    assert(IsQuestionmark());
}

void ShapeValue::QuaternarySelectOp(ShapeValue &ca, ShapeValue &cb,
    ShapeValue &tv, ShapeValue &fv, F4 finteger, F4 fvalue) {
  // Check first if the test (ca & cb) can be evaluated at compile time.
  if (ca.IsConst() && cb.IsConst()) {
    // Init using a & b.
    Init(ca, cb);
    // Comparison will set the right const/affine depending on the input
    // selected, as the compare can be evaluated at compile time.
    finteger(*this, ca, cb, tv, fv);
  } else {
    // Init with all inputs, and if we can create ops, use dyn computations.
    Init(ca, cb, tv, fv);
    if (rewriter)
      fvalue(*this, ca, cb, tv, fv);
    // We have a non constant without being able to create a value;
    // we get a question mark.
    else
      assert(IsQuestionmark());
  }
}

// Operators
void ShapeValue::Add(ShapeValue &a, ShapeValue &b, Location loc) {
  F2 constAddFct = [](ShapeValue &rr, ShapeValue &aa, ShapeValue &bb) {
    rr.SetConstVal(aa.GetConstVal() + bb.GetConstVal());
  };
  F2 valueAddFct = [&](ShapeValue &rr, ShapeValue &aa, ShapeValue &bb) {
    rr.SetDynVal(rr.GetRewriter()->create<AddIOp>(
        loc, aa.GetDynVal(loc), bb.GetDynVal(loc)));
  };
  BinaryOp(a, b, constAddFct, valueAddFct);
}

void ShapeValue::Sub(ShapeValue &a, ShapeValue &b, Location loc) {
  F2 constSubFct = [](ShapeValue &rr, ShapeValue &aa, ShapeValue &bb) {
    rr.SetConstVal(aa.GetConstVal() - bb.GetConstVal());
  };
  F2 valueSubFct = [&](ShapeValue &rr, ShapeValue &aa, ShapeValue &bb) {
    rr.SetDynVal(rr.GetRewriter()->create<SubIOp>(
        loc, aa.GetDynVal(loc), bb.GetDynVal(loc)));
  };
  BinaryOp(a, b, constSubFct, valueSubFct);
}

void ShapeValue::Inc(ShapeValue &a, Location loc) {
  // since this += a; make first a copy of "this"
  ShapeValue selfCopy(*this);
  Add(selfCopy, a, loc);
}

void ShapeValue::Dec(ShapeValue &a, Location loc) {
  // since this -= a; make first a copy of "this"
  ShapeValue selfCopy(*this);
  Sub(selfCopy, a, loc);
}

void ShapeValue::Select(ShapeValue &condA, ShapeValue &condB,
    CmpIPredicate comparePred, ShapeValue &trueVal, ShapeValue &falseVal,
    Location loc) {
  F4 constCompareFct = [&](ShapeValue &rr, ShapeValue &ca, ShapeValue &cb,
                           ShapeValue &tv, ShapeValue &fv) {
    int64_t sca = ca.GetConstVal();
    int64_t scb = cb.GetConstVal();
    uint64_t uca = (uint64_t)sca;
    uint64_t ucb = (uint64_t)scb;
    switch (comparePred) {
    case CmpIPredicate::eq:
      if (sca == scb)
        rr.Copy(tv);
      else
        rr.Copy(fv);
      break;
    case CmpIPredicate::ne:
      if (sca != scb)
        rr.Copy(tv);
      else
        rr.Copy(fv);
      break;
    case CmpIPredicate::slt:
      if (sca < scb)
        rr.Copy(tv);
      else
        rr.Copy(fv);
      break;
    case CmpIPredicate::sle:
      if (sca <= scb)
        rr.Copy(tv);
      else
        rr.Copy(fv);
      break;
    case CmpIPredicate::sgt:
      if (sca > scb)
        rr.Copy(tv);
      else
        rr.Copy(fv);
      break;
    case CmpIPredicate::sge:
      if (sca >= scb)
        rr.Copy(tv);
      else
        rr.Copy(fv);
      break;
    case CmpIPredicate::ult:
      if (uca < ucb)
        rr.Copy(tv);
      else
        rr.Copy(fv);
      break;
    case CmpIPredicate::ule:
      if (uca <= ucb)
        rr.Copy(tv);
      else
        rr.Copy(fv);
      break;
    case CmpIPredicate::ugt:
      if (uca > ucb)
        rr.Copy(tv);
      else
        rr.Copy(fv);
      break;
    case CmpIPredicate::uge:
      if (uca >= ucb)
        rr.Copy(tv);
      else
        rr.Copy(fv);
      break;
    default:
      llvm_unreachable("unknown compare opeartor");
    }
  };
  F4 dynCompareFct = [&](ShapeValue &rr, ShapeValue &ca, ShapeValue &cb,
                         ShapeValue &tv, ShapeValue &fv) {
    auto compareVal = rr.GetRewriter()->create<CmpIOp>(
        loc, comparePred, ca.GetDynVal(loc), cb.GetDynVal(loc));
    auto resVal = rr.rewriter->create<SelectOp>(
        loc, compareVal, tv.GetDynVal(loc), fv.GetDynVal(loc));
    rr.SetDynVal(resVal);
  };
  QuaternarySelectOp(
      condA, condB, trueVal, falseVal, constCompareFct, constCompareFct);
}

void ShapeValue::Select(ShapeValue &condA, ShapeValue &condB,
    CmpIPredicate comparePred, ShapeValue &trueVal, Location loc) {
  // When there is no false value, we simply recopy the current value on the
  // false path.
  ShapeValue falseVal(*this);
  Select(condA, condB, comparePred, trueVal, falseVal, loc);
}

void ShapeValue::Clip(ShapeValue &min, ShapeValue &max, int64_t minInc,
    int64_t maxInc, Location loc) {
  // Functions below uncoditionally override rr with the clipped value of val.
  F3 constClipFct = [&](ShapeValue &rr, ShapeValue &val, ShapeValue &min,
                        ShapeValue &max) {
    // assume signed compares
    int64_t smin = min.GetConstVal() + minInc;
    int64_t smax = max.GetConstVal() + maxInc;
    int64_t sval = val.GetConstVal();
    if (sval < smin)
      sval = smin;
    if (sval > smax)
      sval = smax;
    rr.SetConstVal(sval);
  };
  F3 dynClipFct = [&](ShapeValue &rr, ShapeValue &val, ShapeValue &min,
                      ShapeValue &max) {
    // Copy because don't want to modify the original min.
    ShapeValue minBound(min); 
    if (minInc != 0) {
      ShapeValue increment(minInc, rr.GetRewriter());
      minBound.Inc(increment, loc);
    }
    ShapeValue minClipped(val);
    minClipped.Select(val, minBound, CmpIPredicate::slt, minBound, loc);
    // Copy because don't want to modify the original max.
    ShapeValue maxBound(max);
    if (maxInc != 0) {
      ShapeValue increment(maxInc, rr.GetRewriter());
      maxBound.Inc(increment, loc);
    }
    rr.Select(minClipped, maxBound, CmpIPredicate::slt, maxBound, loc);
  };
  // Save the original value in a separate ShapeValue
  ShapeValue val(*this);
  TernaryOp(val, min, max, constClipFct, dynClipFct);
}
