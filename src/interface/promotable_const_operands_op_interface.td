//===------------- promotable_const_operands_op_interface.td --------------===//
//===---- Promotable Const Operands Op Interface TableGen Definition ------===//
//
// Copyright 2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains the TableGen definition of the promotable const operands
// op interface.
//
//===----------------------------------------------------------------------===//

#ifdef PROMOTABLE_CONST_OPERANDS_OP_INTERFACE
#else
#define PROMOTABLE_CONST_OPERANDS_OP_INTERFACE

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

def PromotableConstOperandsOpInterface : OpInterface<"PromotableConstOperandsOpInterface"> {
  let description = [{
    Interface to access a registered method to infer the return types for an
    operation that can be used during type inference.
  }];

  let methods = [
    InterfaceMethod<"Infer and set the output shape for the current operation.",
                    "std::map<std::string, size_t>", "promotableConstOperands">
  ];
}

#endif // PROMOTABLE_CONST_OPERANDS_OP_INTERFACE
