//===----------------------------------------------------------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
//===----------------------------------------------------------------------===//

#include <numeric>

#include "mlir/Analysis/Verifier.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"

#include "sgir.hpp"

using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

namespace {

class SGIRGenImpl {
public :
  SGIRGenImpl(mlir::MLIRContext &context)
      : context(context), builder(&context) {}

  mlir::ModuleOp mlirGen() {
    theModule = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
    return theModule;
  }

private:
  mlir::MLIRContext &context;
  mlir::ModuleOp theModule;
  mlir::OpBuilder builder;

} ;

} //namespace

namespace onnf {

int SGIRTest() {
  mlir::MLIRContext context;
  
  mlir::OwningModuleRef module = SGIRGenImpl(context).mlirGen();
  if (!module)
    return 1;
  module->dump();
  return 0;
}

} //namespace onnf

