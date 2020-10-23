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
// IndexExprContainer utils.
//===----------------------------------------------------------------------===//

IndexExprContainer::IndexExprContainer(
    ConversionPatternRewriter *rewriter, Location loc)
    : rewriter(rewriter), loc(loc), dims(), symbols() {}

IndexExpr IndexExprContainer::CreateUndefinedIndexExpr() {
  IndexExpr res;
  return res.InitAsUndefined();
}

IndexExpr IndexExprContainer::CreateQuestionmarkIndexExpr() {
  IndexExpr res;
  return res.InitAsQuestionmark(this);
}

IndexExpr IndexExprContainer::CreateLiteralIndexExpr(int64_t val) {
  IndexExpr res;
  return res.InitAsIntLit(this, val);
}

IndexExpr IndexExprContainer::CreateDimIndexExpr(Value val) {
  IndexExpr res;
  return res.InitAsDim(this, val);
}

IndexExpr IndexExprContainer::CreateDimIndexExpr(
    Value memref, ArrayRef<int64_t> memrefShape, int index) {
  IndexExpr res;
  return res.InitAsDim(this, memref, memrefShape, index);
}
IndexExpr IndexExprContainer::CreateSymbolIndexExpr(Value val) {
  IndexExpr res;
  return res.InitAsSymbol(this, val);
}

int IndexExprContainer::AddDim(Value value) {
  dims.emplace_back(value);
  return dims.size() - 1;
  ;
}
int IndexExprContainer::AddSymbol(Value value) {
  symbols.emplace_back(value);
  return symbols.size() - 1;
}

void IndexExprContainer::GetDimAndSymbolList(
    SmallVectorImpl<Value> &list) const {
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
    : isDefined(false), isIntLit(false), isAffine(false), isSymbol(false),
      isDim(false), intLit(0), affineExpr(nullptr), value(nullptr),
      container(nullptr) {}

IndexExpr &IndexExpr::InitAsUndefined() {
  return Init(/*container*/ nullptr, /*isDefined*/ false, /*isIntLit*/ false,
      /*isAffine*/ false, /*isSymbol*/ false, /*isDim*/ false, 0,
      AffineExpr(nullptr), Value(nullptr));
}

IndexExpr &IndexExpr::InitAsQuestionmark(IndexExprContainer *newContainer) {
  return Init(newContainer, /*isDefined*/ true, /*isIntLit*/ false,
      /*isAffine*/ true, /*isSymbol*/ false, /*isDim*/ false, 0,
      AffineExpr(nullptr), Value(nullptr));
}

IndexExpr &IndexExpr::InitAsIntLit(
    IndexExprContainer *newContainer, int64_t val) {
  return Init(newContainer, /*isDefined*/ true, /*isIntLit*/ true,
      /*isAffine*/ true, /*isSymbol*/ false, /*isDim*/ false, val,
      AffineExpr(nullptr), Value(nullptr));
}

IndexExpr &IndexExpr::InitAsDim(IndexExprContainer *newContainer, Value val) {
  return InitAsValueOrIntLit(
      newContainer, val, /*isAffine*/ true, /*isSymbol*/ false, /*isDim*/ true);
}

IndexExpr &IndexExpr::InitAsSymbol(
    IndexExprContainer *newContainer, Value val) {
  return InitAsValueOrIntLit(
      newContainer, val, /*isAffine*/ true, /*isSymbol*/ true, /*isDim*/ false);
}

IndexExpr &IndexExpr::InitAsAffineExpr(
    IndexExprContainer *newContainer, AffineExpr val) {
  return Init(newContainer, /*isDefined*/ true, /*isIntLit*/ false,
      /*isAffine*/ true, /*isSymbol*/ false, /*isDim*/ false, 0,
      AffineExpr(val), Value(nullptr));
}

IndexExpr &IndexExpr::InitAsValue(IndexExprContainer *newContainer, Value val) {
  return InitAsValueOrIntLit(newContainer, val, /*isAffine*/ false,
      /*isSymbol*/ false, /*isDim*/ false);
}

IndexExpr &IndexExpr::InitAsDim(IndexExprContainer *newContainer, Value memref,
    ArrayRef<int64_t> memrefShape, int index) {
  if (memrefShape[index] < 0) {
    // We have a dynamic dimension.
    assert(newContainer && "expected a container");
    Value dynVal = newContainer->GetRewriter().create<DimOp>(
        newContainer->GetLocation(), memref, index);
    return InitAsDim(newContainer, dynVal);
  }
  // We have a constant dimension.
  int64_t intVal = memrefShape[index];
  return InitAsIntLit(newContainer, intVal);
}

IndexExpr &IndexExpr::Init(IndexExprContainer *newContainer, bool newIsDefined,
    bool newIsIntLit, bool newIsAffine, bool newIsSymbol, bool newIsDim,
    int newIntLit, AffineExpr newAffineExpr, Value newValue) {
  if (newIsDefined)
    assert(newContainer && "defined expressions need a container");
  container = newContainer;
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

IndexExpr &IndexExpr::InitAsValueOrIntLit(IndexExprContainer *newContainer,
    Value val, bool newIsAfine, bool newIsSymbol, bool newIsDim) {
  // Do we have a literal integer, if we do, handle it now.
  int64_t valIntLit;
  if (getIntegerLiteralFromValue(val, valIntLit)) {
    // We have an integer. No need for symbol or dim. It is by default affine.
    return InitAsIntLit(newContainer, valIntLit);
  }
  // We have a value, check if it is of the right type.
  auto type = val.getType();
  if (type.isa<IntegerType>()) {
    // We need to convert the int into an index, since we are dealing with index
    // expressions.
    assert(newContainer && "defined expressions need a container");
    val = newContainer->GetRewriter().create<IndexCastOp>(
        newContainer->GetLocation(), newContainer->GetRewriter().getIndexType(),
        val);
  } else {
    assert(type.isa<IndexType>() && "unsupported element type");
  }
  // Now record the value. Affine Expr will be created on demand by
  // GetAffineExpr.
  assert(!(newIsDim && newIsSymbol) &&
         "cannot have dim and symbol at the same time");
  return Init(newContainer, /*isDefined*/ true, /*isIntLit*/ false,
      /*isAffine*/ newIsAfine, /*isSymbol*/ newIsSymbol, /*isDim*/ newIsDim, 0,
      AffineExpr(nullptr), val);
}

IndexExpr &IndexExpr::Copy(IndexExpr &a) {
  // If we go to a model like Values & AffineExpr with a pointer to the actual
  // data, we should just make the indirection here. Copy info in the meanwhile.
  *this = a;
  return *this;
}

//===----------------------------------------------------------------------===//
// IndexExpr list querries.
//===----------------------------------------------------------------------===//

bool IndexExpr::IsDefined() const {
  assert(!isDefined || HasContainer());
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
  assert(HasContainer());
  return container->IsShapeInferencePass();
}

bool IndexExpr::HasContainer() const { return container != nullptr; }

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
    affineExpr = container->GetRewriter().getAffineConstantExpr(intLit);
  } else if (IsSymbol()) {
    // Create a symbol value expr and register its value in the
    // array of symbols. Has value because symbols are gen on demand from
    // values.
    assert(HasValue());
    int id = container->AddSymbol(value);
    affineExpr = container->GetRewriter().getAffineSymbolExpr(id);
  } else if (IsDim()) {
    // Create a dim/index value expr and register its value in the
    // array of dims/indices. Has value because dims are gen on demand from
    // values.
    assert(HasValue());
    int id = container->AddDim(value);
    affineExpr = container->GetRewriter().getAffineDimExpr(id);
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
    value = container->GetRewriter().create<ConstantIndexOp>(
        container->GetLocation(), intLit);
  } else if (HasAffineExpr()) {
    // Has an affine expression: need to build a map, and then perform an
    // affine.apply.
    int dimNum = container->GetDimSize();
    int symNum = container->GetSymbolSize();
    AffineMap map = AffineMap::get(
        dimNum, symNum, {affineExpr}, container->GetRewriter().getContext());
    // We need to concatenate the dims and symbol into a single
    // list, and then use the apply.
    SmallVector<Value, 4> list;
    container->GetDimAndSymbolList(list);
    value = container->GetRewriter().create<AffineApplyOp>(
        container->GetLocation(), map, list);
  } else {
    assert(HasValue());
  }
  return value;
}

IndexExprContainer *IndexExpr::GetContainer() const {
  assert(HasContainer());
  return container;
}

Location IndexExpr::GetLocation() const {
  return GetContainer()->GetLocation();
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
  printf(" container(0x%llx)\n", (long long unsigned)container);
#endif
}

//===----------------------------------------------------------------------===//
// IndexExpr Op Support.
//===----------------------------------------------------------------------===//

// Used for Add/Sub/Mult/CeilDiv/FloorDiv
IndexExpr &IndexExpr::BinaryOp(IndexExpr &a, IndexExpr &b, bool affineWithLitA,
    bool affineWithLitB, bool canBeAffine, F2 litFct, F2 affineExprFct,
    F2 valueFct) {
  assert(a.GetContainer() == b.GetContainer() && "incompatible containers");
  // Literal integer if a and b are literals. Affine if canBeAffine is true,
  // both a and b are affine, and possibly a and/or b are also constant.
  bool resIsLit = a.IsLiteral() && b.IsLiteral();
  bool resIsAffine = resIsLit || (canBeAffine && a.IsAffine() && b.IsAffine() &&
                                     (!affineWithLitA || a.IsLiteral()) &&
                                     (!affineWithLitB || b.IsLiteral()));

  // We use now use the result of the above determination on whether the new
  // index is literal and/or affine.
  if (resIsLit) {
    // Constant, use constant computations.
    litFct(*this, a, b);
  } else if (a.IsShapeInferencePass()) {
    // In shape analysis, if not constant: do noting, aka leave Values & Affine
    // expr undefined.
    InitAsQuestionmark(a.GetContainer());
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
  assert(a.GetContainer() == b.GetContainer() &&
         a.GetContainer() == c.GetContainer() && "incompatible containers");
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
    InitAsQuestionmark(a.GetContainer());
  } else {
    // Use values.
    valueFct(*this, a, b, c);
  }
  return *this;
}

IndexExpr &IndexExpr::QuaternarySelectOp(IndexExpr &compA, IndexExpr &compB,
    IndexExpr &trueVal, IndexExpr &falseVal, F4 litFct, F4 valueFct) {
  assert(compA.GetContainer() == compB.GetContainer() &&
         compA.GetContainer() == trueVal.GetContainer() &&
         compA.GetContainer() == falseVal.GetContainer() &&
         "incompatible containers");
  // Check first if the test (ca & cb) can be evaluated at compile time.
  if (compA.IsLiteral() && compB.IsLiteral()) {
    // Comparison will set the right const/affine depending on the input
    // selected, as the compare can be evaluated at compile time.
    litFct(*this, compA, compB, trueVal, falseVal);
  } else if (compA.IsShapeInferencePass()) {
    // Just set as undefined
    InitAsQuestionmark(compA.GetContainer());
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
    assert(vals[0].GetContainer() == vals[i].GetContainer() &&
           "incompatible containers");
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
    InitAsQuestionmark(vals[0].GetContainer());
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
    res.InitAsIntLit(aa.GetContainer(), aa.GetLiteral() + bb.GetLiteral());
  };
  F2 affineExprFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    res.InitAsAffineExpr(
        aa.GetContainer(), aa.GetAffineExpr() + bb.GetAffineExpr());
  };
  F2 valueFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    res.InitAsValue(
        aa.GetContainer(), aa.GetContainer()->GetRewriter().create<AddIOp>(
                               aa.GetLocation(), aa.GetValue(), bb.GetValue()));
  };
  return BinaryOp(a, b, false, false, true, litFct, affineExprFct, valueFct);
}

IndexExpr &IndexExpr::Add(IndexExpr &a, int64_t b) {
  IndexExpr bIndex = a.GetContainer()->CreateLiteralIndexExpr(b);
  return Add(a, bIndex);
}

IndexExpr &IndexExpr::Sub(IndexExpr &a, IndexExpr &b) {
  F2 litFct = [](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    res.InitAsIntLit(aa.GetContainer(), aa.GetLiteral() - bb.GetLiteral());
  };
  F2 affineExprFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    res.InitAsAffineExpr(
        aa.GetContainer(), aa.GetAffineExpr() - bb.GetAffineExpr());
  };
  F2 valueFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    res.InitAsValue(
        aa.GetContainer(), aa.GetContainer()->GetRewriter().create<SubIOp>(
                               aa.GetLocation(), aa.GetValue(), bb.GetValue()));
  };
  return BinaryOp(a, b, false, false, true, litFct, affineExprFct, valueFct);
}

IndexExpr &IndexExpr::Sub(IndexExpr &a, int64_t b) {
  IndexExpr bIndex = a.GetContainer()->CreateLiteralIndexExpr(b);
  return Sub(a, bIndex);
}

IndexExpr &IndexExpr::Sub(int64_t a, IndexExpr &b) {
  IndexExpr aIndex = b.GetContainer()->CreateLiteralIndexExpr(a);
  return Sub(aIndex, b);
}

// If one input is possibly a literal, place it in the first position (a).
IndexExpr &IndexExpr::Mult(IndexExpr &a, IndexExpr &b) {
  F2 litFct = [](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    res.InitAsIntLit(aa.GetContainer(), aa.GetLiteral() * bb.GetLiteral());
  };
  F2 affineExprFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    // Operand aa must be a literal.
    res.InitAsAffineExpr(
        aa.GetContainer(), bb.GetAffineExpr() * aa.GetLiteral());
  };
  F2 valueFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    if (aa.IsLiteral() && aa.GetLiteral() == 1) {
      res.Copy(bb);
    } else if (bb.IsLiteral() && bb.GetLiteral() == 1) {
      res.Copy(aa);
    } else {
      res.InitAsValue(aa.GetContainer(),
          aa.GetContainer()->GetRewriter().create<MulIOp>(
              aa.GetLocation(), aa.GetValue(), bb.GetValue()));
    }
  };
  // Index a must be a literal.
  return BinaryOp(a, b, true, false, true, litFct, affineExprFct, valueFct);
}

IndexExpr &IndexExpr::Mult(int64_t a, IndexExpr &b) {
  IndexExpr aIndex = b.GetContainer()->CreateLiteralIndexExpr(a);
  return Mult(aIndex, b);
}

IndexExpr &IndexExpr::FloorDiv(IndexExpr &a, IndexExpr &b) {
  F2 litFct = [](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    int64_t rval = floor((1.0 * aa.GetLiteral()) / (1.0 * bb.GetLiteral()));
    res.InitAsIntLit(aa.GetContainer(), rval);
  };
  F2 affineExprFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    // Operand bb must be a literal.
    int64_t bval = bb.GetLiteral();
    if (bval == 1) {
      res.Copy(aa);
    } else if (bval > 1) {
      res.InitAsAffineExpr(
          aa.GetContainer(), aa.GetAffineExpr().floorDiv(bval));
    } else {
#if CEIL_FLOOR_IN_STD
      res.InitAsValue(aa.GetContainer(),
          aa.GetContainer()->GetRewriter().create<SignedFloorDivIOp>(
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
      res.InitAsValue(aa.GetContainer(),
          aa.GetContainer()->GetRewriter().create<SignedFloorDivIOp>(
              aa.GetLocation(), aa.GetValue(), bb.GetValue()));
#else
      llvm_unreachable("not implemented yet, wait for the new LLVM/MLIR "
                       "support in std");
#endif
    }
  };
  // Index b must be a literal.
  return BinaryOp(a, b, false, true, true, litFct, affineExprFct, valueFct);
}

IndexExpr &IndexExpr::CeilDiv(IndexExpr &a, IndexExpr &b) {
  F2 litFct = [](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    int64_t rval = ceil((1.0 * aa.GetLiteral()) / (1.0 * bb.GetLiteral()));
    res.InitAsIntLit(aa.GetContainer(), rval);
  };
  F2 affineExprFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    // Operand bb must be a literal.
    int64_t bval = bb.GetLiteral();
    if (bval == 1) {
      res.Copy(aa);
    } else if (bval > 1) {
      res.InitAsAffineExpr(aa.GetContainer(), aa.GetAffineExpr().ceilDiv(bval));
    } else {
#if CEIL_FLOOR_IN_STD
      res.InitAsValue(aa.GetContainer(),
          aa.GetContainer()->GetRewriter().create<SignedCeilDivIOp>(
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
      res.InitAsValue(aa.GetContainer(),
          aa.GetContainer()->GetRewriter().create<SignedCeilDivIOp>(
              aa.GetLocation(), aa.GetValue(), bb.GetValue()));
#else
      llvm_unreachable(
          "not implemented yet, wait for the new LLVM/MLIR support in std");
#endif
    }
  };
  // Index b must be a literal.
  return BinaryOp(a, b, false, true, true, litFct, affineExprFct, valueFct);
}

IndexExpr &IndexExpr::Mod(IndexExpr &a, IndexExpr &b) {
  F2 litFct = [](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    res.InitAsIntLit(aa.GetContainer(), mod(aa.GetLiteral(), bb.GetLiteral()));
  };
  F2 affineExprFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    // Operand bb must be a literal.
    int64_t bval = bb.GetLiteral();
    if (bval >= 0) {
      res.InitAsAffineExpr(aa.GetContainer(), aa.GetAffineExpr() % bval);
    } else {
      res.InitAsValue(aa.GetContainer(),
          aa.GetContainer()->GetRewriter().create<SignedRemIOp>(
              aa.GetLocation(), aa.GetValue(), bb.GetValue()));
    }
  };
  F2 valueFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    res.InitAsValue(aa.GetContainer(),
        aa.GetContainer()->GetRewriter().create<SignedRemIOp>(
            aa.GetLocation(), aa.GetValue(), bb.GetValue()));
  };
  // Index b must be a literal.
  return BinaryOp(a, b, false, true, true, litFct, affineExprFct, valueFct);
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
    res.InitAsIntLit(val.GetContainer(), sval);
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

IndexExpr &IndexExpr::Clamp(IndexExpr &val, int64_t min, IndexExpr &max) {
  IndexExpr minIndex = val.GetContainer()->CreateLiteralIndexExpr(min);
  return Clamp(val, minIndex, max);
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
    Value compare = ca.GetContainer()->GetRewriter().create<CmpIOp>(
        ca.GetLocation(), comparePred, ca.GetValue(), cb.GetValue());
    Value results = ca.GetContainer()->GetRewriter().create<SelectOp>(
        ca.GetLocation(), compare, tv.GetValue(), fv.GetValue());
    res.InitAsValue(ca.GetContainer(), results);
  };
  return QuaternarySelectOp(condA, condB, trueVal, falseVal, litFct, valueFct);
}

IndexExpr &IndexExpr::Select(IndexExpr &condA, CmpIPredicate comparePred,
    int64_t condB, IndexExpr &trueVal, IndexExpr &falseVal) {
  IndexExpr condBIndex = condA.GetContainer()->CreateLiteralIndexExpr(condB);
  return Select(condA, comparePred, condBIndex, trueVal, falseVal);
}

IndexExpr &IndexExpr::Select(IndexExpr &condA, CmpIPredicate comparePred,
    int64_t condB, int64_t trueVal, IndexExpr &falseVal) {
  IndexExpr condBIndex = condA.GetContainer()->CreateLiteralIndexExpr(condB);
  IndexExpr trueValIndex =
      condA.GetContainer()->CreateLiteralIndexExpr(trueVal);
  return Select(condA, comparePred, condBIndex, trueValIndex, falseVal);
}

IndexExpr &IndexExpr::Min(SmallVectorImpl<IndexExpr> &vals) {
  F2 litFct = [](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    auto aaa = aa.GetLiteral();
    auto bbb = bb.GetLiteral();
    res.InitAsIntLit(aa.GetContainer(), (aaa < bbb) ? aaa : bbb);
  };
  Flist affineExprFct = [&](IndexExpr &res, SmallVectorImpl<IndexExpr> &vvals) {
    // Create a list of affine expression
    assert(vvals.size() > 1 && "come here only with 2 or more values");
    SmallVector<AffineExpr, 4> affineExprs;
    for (IndexExpr &vv : vvals) {
      affineExprs.emplace_back(vv.GetAffineExpr());
    }
    // Compute a map including the list of affine expressions.
    IndexExprContainer *currContainer = vvals[0].GetContainer();
    int dimNum = currContainer->GetDimSize();
    int symNum = currContainer->GetSymbolSize();
    auto context = currContainer->GetRewriter().getContext();
    AffineMap map = AffineMap::get(dimNum, symNum, affineExprs, context);
    // Compute the min value out of this map.
    SmallVector<Value, 4> dimAndSymList;
    currContainer->GetDimAndSymbolList(dimAndSymList);
    Value minVal = currContainer->GetRewriter().create<AffineMinOp>(
        vvals[0].GetLocation(), map, dimAndSymList);
    res.InitAsValue(container, minVal);
  };
  F2 valueFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    Value compareVal = aa.GetContainer()->GetRewriter().create<CmpIOp>(
        aa.GetLocation(), CmpIPredicate::slt, aa.GetValue(), bb.GetValue());
    Value resVal = aa.GetContainer()->GetRewriter().create<SelectOp>(
        aa.GetLocation(), compareVal, aa.GetValue(), bb.GetValue());
    res.InitAsValue(aa.GetContainer(), resVal);
  };
  return ReductionOp(vals, litFct, affineExprFct, valueFct);
}

IndexExpr &IndexExpr::Max(SmallVectorImpl<IndexExpr> &vals) {
  F2 litFct = [](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    auto aaa = aa.GetLiteral();
    auto bbb = bb.GetLiteral();
    res.InitAsIntLit(aa.GetContainer(), (aaa > bbb) ? aaa : bbb);
  };
  Flist affineExprFct = [&](IndexExpr &res, SmallVectorImpl<IndexExpr> &vvals) {
    // Create a list of affine expression
    assert(vvals.size() > 1 && "come here only with 2 or more values");
    SmallVector<AffineExpr, 4> affineExprs;
    for (IndexExpr &vv : vvals) {
      affineExprs.emplace_back(vv.GetAffineExpr());
    }
    // Compute a map including the list of affine expressions.
    IndexExprContainer *currContainer = vvals[0].GetContainer();
    int dimNum = currContainer->GetDimSize();
    int symNum = currContainer->GetSymbolSize();
    auto context = currContainer->GetRewriter().getContext();
    AffineMap map = AffineMap::get(dimNum, symNum, affineExprs, context);
    // Compute the min value out of this map.
    SmallVector<Value, 4> dimAndSymList;
    currContainer->GetDimAndSymbolList(dimAndSymList);
    Value minVal = currContainer->GetRewriter().create<AffineMaxOp>(
        vvals[0].GetLocation(), map, dimAndSymList);
    res.InitAsValue(container, minVal);
  };
  F2 valueFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    Value compareVal = aa.GetContainer()->GetRewriter().create<CmpIOp>(
        aa.GetLocation(), CmpIPredicate::sgt, aa.GetValue(), bb.GetValue());
    Value resVal = aa.GetContainer()->GetRewriter().create<SelectOp>(
        aa.GetLocation(), compareVal, aa.GetValue(), bb.GetValue());
    res.InitAsValue(aa.GetContainer(), resVal);
  };
  return ReductionOp(vals, litFct, affineExprFct, valueFct);
}
