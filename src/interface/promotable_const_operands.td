//=- shape_inference_interface.td - Shape Inference Interface -*- tablegen -==//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// Defines the operations of the Shape Inference Op Interface.
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
