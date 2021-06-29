//===- TmpMLIRUtils.hpp - Support migration to MLIR without EDSC --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Code here is lifted from MLIR to facilitate the migration to EDSC-free MLIR.
// TODO: remove file once transition is completed.
//
//===----------------------------------------------------------------------===//

#ifndef TMP_MLIR_DIALECT_STANDARDOPS_UTILS_UTILS_H
#define TMP_MLIR_DIALECT_STANDARDOPS_UTILS_UTILS_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

namespace mlir {

//===----------------------------------------------------------------------===//
// from ImplicitLocObBuilder.h
//===----------------------------------------------------------------------===//

/// ImplicitLocOpBuilder maintains a 'current location', allowing use of the
/// create<> method without specifying the location.  It is otherwise the same
/// as OpBuilder.
class ImplicitLocOpBuilder : public mlir::OpBuilder {
public:
  /// OpBuilder has a bunch of convenience constructors - we support them all
  /// with the additional Location.
  template <typename... T>
  ImplicitLocOpBuilder(Location loc, T &&... operands)
      : OpBuilder(std::forward<T>(operands)...), curLoc(loc) {}

  /// Create a builder and set the insertion point to before the first operation
  /// in the block but still inside the block.
  static ImplicitLocOpBuilder atBlockBegin(
      Location loc, Block *block, Listener *listener = nullptr) {
    return ImplicitLocOpBuilder(loc, block, block->begin(), listener);
  }

  /// Create a builder and set the insertion point to after the last operation
  /// in the block but still inside the block.
  static ImplicitLocOpBuilder atBlockEnd(
      Location loc, Block *block, Listener *listener = nullptr) {
    return ImplicitLocOpBuilder(loc, block, block->end(), listener);
  }

  /// Create a builder and set the insertion point to before the block
  /// terminator.
  static ImplicitLocOpBuilder atBlockTerminator(
      Location loc, Block *block, Listener *listener = nullptr) {
    auto *terminator = block->getTerminator();
    assert(terminator != nullptr && "the block has no terminator");
    return ImplicitLocOpBuilder(
        loc, block, Block::iterator(terminator), listener);
  }

  /// Accessors for the implied location.
  Location getLoc() const { return curLoc; }
  void setLoc(Location loc) { curLoc = loc; }

  // We allow clients to use the explicit-loc version of create as well.
  using OpBuilder::create;
  using OpBuilder::createOrFold;

  /// Create an operation of specific op type at the current insertion point and
  /// location.
  template <typename OpTy, typename... Args>
  OpTy create(Args &&... args) {
    return OpBuilder::create<OpTy>(curLoc, std::forward<Args>(args)...);
  }

  /// Create an operation of specific op type at the current insertion point,
  /// and immediately try to fold it. This functions populates 'results' with
  /// the results after folding the operation.
  template <typename OpTy, typename... Args>
  void createOrFold(llvm::SmallVectorImpl<Value> &results, Args &&... args) {
    OpBuilder::createOrFold<OpTy>(results, curLoc, std::forward<Args>(args)...);
  }

  /// Overload to create or fold a single result operation.
  template <typename OpTy, typename... Args>
  typename std::enable_if<OpTy::template hasTrait<mlir::OpTrait::OneResult>(),
      Value>::type
  createOrFold(Args &&... args) {
    return OpBuilder::createOrFold<OpTy>(curLoc, std::forward<Args>(args)...);
  }

  /// Overload to create or fold a zero result operation.
  template <typename OpTy, typename... Args>
  typename std::enable_if<OpTy::template hasTrait<mlir::OpTrait::ZeroResult>(),
      OpTy>::type
  createOrFold(Args &&... args) {
    return OpBuilder::createOrFold<OpTy>(curLoc, std::forward<Args>(args)...);
  }

  /// This builder can also be used to emit diagnostics to the current location.
  mlir::InFlightDiagnostic emitError(
      const llvm::Twine &message = llvm::Twine()) {
    return mlir::emitError(curLoc, message);
  }
  mlir::InFlightDiagnostic emitWarning(
      const llvm::Twine &message = llvm::Twine()) {
    return mlir::emitWarning(curLoc, message);
  }
  mlir::InFlightDiagnostic emitRemark(
      const llvm::Twine &message = llvm::Twine()) {
    return mlir::emitRemark(curLoc, message);
  }

private:
  Location curLoc;
};

struct DialectBuilder {
  DialectBuilder(OpBuilder &b, Location loc) : b(b), loc(loc) {}
  DialectBuilder(ImplicitLocOpBuilder &lb) : b(lb), loc(lb.getLoc()) {}
  DialectBuilder(DialectBuilder &db) : b(db.b), loc(db.loc) {}

  OpBuilder &getBuilder() { return b; }
  Location getLoc() { return loc; }

protected:
  OpBuilder &b;
  Location loc;
};


//===----------------------------------------------------------------------===//
// from Utils.h
//===----------------------------------------------------------------------===//

/// Helper struct to build simple arithmetic quantities with minimal type
/// inference support.
struct ArithBuilder : DialectBuilder {
  ArithBuilder(OpBuilder &b, Location loc) : DialectBuilder(b, loc) {}
  ArithBuilder(ImplicitLocOpBuilder &lb) : DialectBuilder(lb) {}
  ArithBuilder(DialectBuilder &db) : DialectBuilder(db) {}

  Value _and(Value lhs, Value rhs);
  Value add(Value lhs, Value rhs);
  Value mul(Value lhs, Value rhs);
  Value select(Value cmp, Value lhs, Value rhs);
  Value sgt(Value lhs, Value rhs);
  Value slt(Value lhs, Value rhs);
};

} // namespace mlir
#endif